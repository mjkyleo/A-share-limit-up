# -*- coding: utf-8 -*-
"""
回测模块（backtest）v2.5
逻辑：predictions(run_date→target_date) ⋈ daily(target_date 实际数据)
产出：
  1. 整体准确率：Top3/5/10 命中率、平均实际涨幅、基准对比
  2. 清单得分：每张清单独立打分规则（涨停满分/按涨幅分档/下跌扣分）
  3. 单策略表现：每个策略分 vs 实际涨幅 的 Spearman IC（分离评估哪个策略有效）
  4. 滚动历史：多日累计 IC 均值 + CI95 + 样本n + 清单滚动平均得分
  5. 调参建议 make_advice()：基于滚动 IC + 增量IC + 因子相关性 自动生成权重调整建议

本次升级（P0-2/P0-3/P0-1/P0-5/P0-6）：
  - backtest_one 支持 收益口径(caliber) + 双边成本 + 等权净值/绩效 + 无法成交剔除
  - rolling_ic 增 CI95 + 样本n + Bonferroni 显著性
  - dynamic_weight_plan 改 shrinkage(EMA+显著性+λ收缩)，弃硬 kill 归零
  - make_advice 增 因子相关性(五因子) + 正交后增量IC 段落
"""
import json

import numpy as np
import pandas as pd

from core import metrics

STRATEGY_KEYS = ['S1封单强度', 'S2封板质量', 'S3锁仓度', 'S4资金', 'S5股性结构',
                 'A封单强度', 'B封板时间', 'C量能换手', 'D市值', 'E资金']  # 兼容v1命名

# v2 策略键 → 权重键 映射（建议引擎用）
V2_KEYS = ['S1封单强度', 'S2封板质量', 'S3锁仓度', 'S4资金', 'S5股性结构']

# ---------------- 清单打分规则 ----------------
DEFAULT_SCORING = {
    '涨停TopN': [[-99, -80], [-6, -50], [-4, -30], [-2, -10], [0, 20],
                 [2, 40], [4, 60], [6, 75], [9.5, 100]],
    '涨幅4%': [[-99, -60], [-6, -40], [-4, -25], [-2, -10], [0, 15], [2, 40], [4, 100]],
    '连板候选': [[-99, -100], [-6, -70], [-4, -45], [-2, -20], [0, 20], [4, 50], [9.5, 100]],
    '尾盘选股': [[-99, -60], [-4, -35], [-2, -15], [0, 30], [2, 60], [4, 80], [9.5, 100]],
    '抄底清单': [[-99, -80], [-6, -50], [-4, -30], [-2, -15], [0, 20],
                 [2, 60], [4, 80], [6, 95], [9.5, 100]],
    '短线反转': [[-99, -80], [-6, -50], [-4, -30], [-2, -15], [0, 20],
                 [2, 60], [4, 80], [6, 95], [9.5, 100]],
    '防御低波': [[-99, -100], [-6, -60], [-4, -40], [-2, -10], [0, 50],
                 [2, 80], [4, 100], [6, 100], [9.5, 100]],
}

# 每张清单的"达成"口径（计算达成率用）：实际涨幅 ≥ 该值 算达成清单目标
HIT_THRESHOLD = {'涨停TopN': 9.5, '涨幅4%': 4.0, '连板候选': 9.5, '尾盘选股': 2.0,
                 '抄底清单': 2.0, '短线反转': 2.0, '防御低波': 0.0}

LIST_TYPES = ['涨停TopN', '涨幅4%', '连板候选', '尾盘选股']

# ---------------- 合并扩展框架注册的清单类型 ----------------
try:
    from extensions import EXT_LIST_TYPES, EXT_SCORING, EXT_HIT, EXT_IC_KEYS
    if EXT_LIST_TYPES:
        LIST_TYPES = LIST_TYPES + EXT_LIST_TYPES
    if EXT_SCORING:
        DEFAULT_SCORING = {**DEFAULT_SCORING, **EXT_SCORING}
    if EXT_HIT:
        HIT_THRESHOLD = {**HIT_THRESHOLD, **EXT_HIT}
    if EXT_IC_KEYS:
        _seen = set(STRATEGY_KEYS)
        for _k in EXT_IC_KEYS:
            if _k not in _seen:
                STRATEGY_KEYS.append(_k)
                _seen.add(_k)
except Exception:
    pass


# =============================== 计算缓存（性能优化 P0+） ===============================
# 回测的瓶颈在于：rolling_ic / rolling_score / rolling_net_curve / factor_correlation /
# incremental_ic / make_advice 各自都在「遍历所有可回测日」时反复调用 backtest_one，
# 导致 backtest_one 被重复计算 ~13×N 次。这里对 backtest_one 与聚合函数做进程内记忆化，
# 同一 (db, 日期, 清单, 口径) 只算一次，下游聚合全部命中缓存。
# 键使用 db.path（而非 id(db)）以规避对象回收后 id 复用导致的串库风险。
_MEMO = {}


def clear_backtest_cache():
    """清空所有回测/指标缓存。每次用户发起一次完整回测前调用，避免配置变更后读到陈旧结果。"""
    _MEMO.clear()
    try:
        from core import metrics
        metrics._MEMO.clear()
    except Exception:
        pass


def score_row(pct, rules):
    """实际涨幅(百分点) → 得分。rules=[[下限,得分]...] 升序，取满足条件的最后一档"""
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return np.nan
    s = rules[0][1]
    for lo, sc in rules:
        if pct >= lo:
            s = sc
        else:
            break
    return s


def _scoring(cfg, list_type):
    rules = (cfg or {}).get('scoring', {}).get(list_type)
    if rules:
        try:
            return sorted([[float(a), float(b)] for a, b in rules], key=lambda x: x[0])
        except Exception:
            pass
    return DEFAULT_SCORING[list_type]


def _spearman(a, b):
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(a, b)
        return round(float(rho), 3), float(p)
    except Exception:
        return None, None


# =============================== 单日回测 ===============================
def backtest_one(db, target_date, list_type='涨停TopN', cfg=None, caliber=None):
    """单日回测 → (明细df, 汇总dict, 策略IC dict, 绩效dict)

    明细df 新增列：口径收益% , 净收益% , 无法成交(bool) , 双边成本% , 组合净值(累计)
    汇总新增：口径 , 双边成本% , 无法成交比例 , 样本n , 基准净收益% , 净组合收益%
    绩效dict: net_return_curve 结构 + gross_return/bench_gross/bench_return/win_rate/...

    口径(caliber)：
      'close' = T日收盘→T+1收盘（用 daily['涨幅']）
      'open'   = T+1开盘→T+1收盘（用 (1+涨幅/100)/(1+开盘涨幅/100)-1）
      缺开盘涨幅时降级 'close' 并提示。
    """
    cfg = cfg or {}
    bcfg = cfg.get('backtest_cost', {})
    cal = caliber or bcfg.get('caliber', 'close')
    _key = (db.path, target_date, list_type, cal)
    if _key in _MEMO:
        return _MEMO[_key]
    pred = db.get_predictions(target_date)
    pred = pred[pred['list_type'] == list_type]
    if pred.empty:
        _MEMO[_key] = (None, None, None, None)
        return None, None, None, None
    actual = db.get_daily(target_date)
    if actual.empty:
        _MEMO[_key] = (None, None, None, None)
        return None, None, None, None
    merge_cols = ['代码6', '涨幅', '今日涨停', '开盘涨幅', '封板分钟']
    actual_cols = [c for c in merge_cols if c in actual.columns]
    mg = pred.merge(actual[actual_cols], left_on='code', right_on='代码6', how='inner')
    if len(mg) < 3:
        _MEMO[_key] = (None, None, None, None)
        return None, None, None, None
    mg = mg.sort_values('rank').reset_index(drop=True)
    mg = mg.rename(columns={'涨幅': '实际涨幅%', '今日涨停': '实际涨停'})

    # M可买性（来自 detail，用于"无法成交"兜底判定）
    det = mg['detail'].apply(lambda s: json.loads(s) if s else {})
    mg['M可买性'] = det.apply(lambda x: x.get('M可买性'))

    # ---- 口径收益% ----
    if cal == 'open' and '开盘涨幅' in mg.columns and mg['开盘涨幅'].notna().any():
        og = mg['开盘涨幅'].astype(float) / 100.0
        zf = mg['实际涨幅%'].astype(float) / 100.0
        mg['口径收益%'] = ((1.0 + zf) / (1.0 + og) - 1.0)
        caliber_used = 'open(T+1开盘→收盘)'
    else:
        mg['口径收益%'] = mg['实际涨幅%'].astype(float) / 100.0
        caliber_used = 'close(T日收盘→T+1收盘)' if cal != 'open' else 'close(降级:缺开盘涨幅)'

    # ---- 双边成本%（每行只算一次，后续组合/净值/曲线复用）----
    cost = mg['代码6'].apply(lambda c: metrics.per_trade_cost_pct(c, bcfg))
    mg['双边成本%'] = (cost * 100).round(4)
    mg['净收益%'] = (mg['口径收益%'] * 100 - mg['双边成本%']).round(3)

    # ---- 无法成交（向量化）：涨停 且（封板分钟≤0.5 若有效，否则 M可买性<0.3）----
    _is_up = mg['实际涨停'].astype(bool)
    _fm = pd.to_numeric(mg.get('封板分钟'), errors='coerce')
    _mb = pd.to_numeric(mg.get('M可买性'), errors='coerce')
    mg['无法成交'] = _is_up & np.where(_fm.notna(), _fm <= 0.5, _mb < 0.3)

    # ---- 等权组合（仅可成交股）----
    trad = mg[~mg['无法成交']]
    if len(trad) > 0:
        net_i = trad['口径收益%'] - cost[~mg['无法成交']]
        port_gross = float(trad['口径收益%'].mean())
        port_net = float(net_i.mean())
        win_rate = float((net_i > 0).mean())
        rep_code = trad['代码6'].iloc[0]
    else:
        port_gross = 0.0
        port_net = 0.0
        win_rate = 0.0
        rep_code = mg['代码6'].iloc[0] if len(mg) > 0 else '600000'
    cost_pct = float(cost.mean())
    cant_ratio = float(mg['无法成交'].mean())

    # ---- 基准：买全部涨停股（等权）----
    lim = mg[mg['实际涨停']]
    if len(lim) > 0:
        bench_gross = float(lim['口径收益%'].mean())
        bench_in = lim[['代码6', '口径收益%']].copy()
        bench_in['今日涨停'] = True
        bench = metrics.benchmark_curve(bench_in, bcfg)
        bench_net = bench['net_return']
        bench_code = lim['代码6'].iloc[0]
    else:
        bench_gross = 0.0
        bench_net = 0.0
        bench_code = rep_code

    # ---- 得分（按清单规则）----
    rules = _scoring(cfg, list_type)
    mg['得分'] = mg['实际涨幅%'].apply(lambda x: score_row(x, rules))
    hit_th = HIT_THRESHOLD.get(list_type, 9.5)

    summary = {'目标日': target_date, '清单': list_type, '口径': caliber_used,
               '预测数': len(pred), '匹配数': len(mg), '可成交数': len(trad)}
    for n in (3, 5, 10):
        t = mg.head(min(n, len(mg)))
        summary[f'Top{n}涨停率'] = f"{t['实际涨停'].mean():.0%}"
        summary[f'Top{n}平均涨幅'] = f"{t['实际涨幅%'].mean():.2f}%"
    summary['全样本平均涨幅'] = f"{mg['实际涨幅%'].mean():.2f}%"
    summary['平均得分'] = f"{mg['得分'].mean():.1f}"
    summary['总得分'] = f"{mg['得分'].sum():.0f}"
    summary['达成率'] = f"{(mg['实际涨幅%'] >= hit_th).mean():.0%}（涨幅≥{hit_th}%）"
    summary['净组合收益%'] = f"{port_net * 100:.2f}%"
    summary['双边成本%'] = f"{cost_pct * 100:.3f}%"
    summary['无法成交比例'] = f"{cant_ratio:.0%}"
    summary['样本n'] = len(mg)
    summary['基准净收益%'] = f"{bench_net * 100:.2f}%"

    # ---- 单策略 IC（从 detail JSON 拆出策略分）----
    ic = {}
    for key in STRATEGY_KEYS:
        vals = det.apply(lambda d: d.get(key))
        vals = pd.to_numeric(vals, errors='coerce')
        mask = vals.notna() & mg['实际涨幅%'].notna()
        if mask.sum() >= 5 and vals[mask].nunique() > 2:
            rho, p = _spearman(vals[mask], mg.loc[mask, '实际涨幅%'])
            if rho is not None:
                ic[key] = {'IC(ρ)': rho, 'p值': round(p, 4),
                           '评价': '✅ 有效' if rho > 0.4 and p < 0.1
                           else ('⚠️ 弱' if rho > 0 else '❌ 无效/反向')}

    # ---- 绩效 dict ----
    perf = {
        'gross_return': port_gross,
        'net_return': port_net,
        'bench_gross': bench_gross,
        'bench_return': bench_net,
        'bench_code': bench_code,
        'win_rate': win_rate,
        'cost_pct': cost_pct,
        'cant_trade_ratio': cant_ratio,
        'n': len(mg),
        'n_tradable': len(trad),
        'rep_code': rep_code,
        'caliber': caliber_used,
        'curve': metrics.net_return_curve(pd.Series([port_gross]),
                                          pd.Series([rep_code]), bcfg),
    }
    # 组合净值(累计) 列（单日仅一点，便于明细展示）
    mg['组合净值(累计)'] = round((1.0 + mg['口径收益%'] - cost).prod(), 4) \
        if len(mg) > 0 else np.nan
    _MEMO[_key] = (mg, summary, ic, perf)
    return mg, summary, ic, perf


# =============================== 每日 IC 矩阵（供滚动/显著性/动态权重复用） ===============================
def _daily_ic_records(db, list_type='涨停TopN', cfg=None):
    """yield (date, key, ic_rho, p) for each backtestable day & each factor with computable IC."""
    out = []
    for d in db.pending_backtest_dates():
        _, _, ic, _ = backtest_one(db, d, list_type, cfg)
        if not ic:
            continue
        for k, v in ic.items():
            out.append((d, k, float(v['IC(ρ)']), float(v['p值'])))
    return out


def _daily_ic_matrix(db, list_type='涨停TopN', cfg=None):
    """返回 (DataFrame[date×factor→IC], 可用天数)。"""
    recs = _daily_ic_records(db, list_type, cfg)
    if not recs:
        return pd.DataFrame(), 0
    rows, days = {}, set()
    for d, k, ic, _p in recs:
        rows.setdefault(d, {})[k] = ic
        days.add(d)
    return pd.DataFrame.from_dict(rows, orient='index'), len(days)


def rolling_ic(db, list_type='涨停TopN', cfg=None):
    """全部可回测日期的策略 IC 汇总（均值 + CI95 + 样本n + 显著性）→ (DataFrame, 天数)"""
    _key = ('ric', db.path, id(cfg), list_type)
    if _key in _MEMO:
        return _MEMO[_key]
    mat, used = _daily_ic_matrix(db, list_type, cfg)
    if mat.empty:
        _MEMO[_key] = (pd.DataFrame(), 0)
        return pd.DataFrame(), 0
    sig = metrics.factor_significance(mat, alpha=0.05)
    pos = {c: int((mat[c] > 0).sum()) for c in mat.columns}
    ndays = {c: int(mat[c].notna().sum()) for c in mat.columns}
    rows = []
    for _, r in sig.iterrows():
        k = r['策略']
        rows.append({
            '策略': k,
            '平均IC': r['均值IC'],
            'IC>0天数': f"{pos.get(k, 0)}/{ndays.get(k, 0)}",
            'CI95低': r['CI95低'],
            'CI95高': r['CI95高'],
            '样本n': ndays.get(k, 0),
            '评价': '✅ 稳定有效' if r['均值IC'] > 0.3
            else ('⚠️ 不稳定' if r['均值IC'] > 0 else '❌ 建议下调权重'),
        })
    _res = (pd.DataFrame(rows).sort_values('平均IC', ascending=False).reset_index(drop=True), used)
    _MEMO[_key] = _res
    return _res


def rolling_score(db, list_type, cfg=None):
    """清单滚动得分汇总 → (DataFrame, 天数)。列：日期/匹配数/平均得分/总得分/达成率"""
    _key = ('rsc', db.path, id(cfg), list_type)
    if _key in _MEMO:
        return _MEMO[_key]
    rows = []
    for d in db.pending_backtest_dates():
        _, summary, _, _ = backtest_one(db, d, list_type, cfg)
        if not summary:
            continue
        rows.append({'日期': d, '匹配数': summary['匹配数'],
                     '平均得分': float(summary['平均得分']),
                     '总得分': float(summary['总得分']),
                     '达成率': summary['达成率']})
    _res = (pd.DataFrame(rows), len(rows)) if rows else (pd.DataFrame(), 0)
    _MEMO[_key] = _res
    return _res


# =============================== 动态权重（P0-5 shrinkage） ===============================
def _ema(vals, window):
    """指数移动平均（最后一点）。vals 为 IC 序列。"""
    s = pd.Series(vals, dtype='float64')
    if s.empty:
        return float('nan')
    return float(s.ewm(span=max(1, window), adjust=False).mean().iloc[-1])


def dynamic_weight_plan(db, cfg, roll=None, used=None):
    """逐因子动态权重计划 v2（shrinkage + EMA + Bonferroni 显著性）。

    返回 (DataFrame 或 None, used天数)。
    列：策略 | EMA_IC | p值 | 是否显著 | 当前权重% | 建议权重% | 动作

    算法（见 config.dynamic_weights）：
      对每日 IC 做 EMA 平滑(窗口=ema_window) → 单样本 t + Bonferroni(α/因子数) 判显著
      → 不显著因子 shrinkage 至地板(不再归零) → w = (1-λ)·max(EMA_IC,0) + 地板 再归一
    动作取值：✅上调 / ⚠️维持(噪声收缩至地板) / ❌不显著(收缩至地板)
    （不再出现 "归零(kill)"）
    """
    dw = (cfg or {}).get('dynamic_weights', {})
    if not dw.get('enabled', True):
        return None, used if used is not None else 0
    floor = float(dw.get('floor', 0.02))
    min_days = int(dw.get('min_days', 3))
    ema_window = int(dw.get('ema_window', 10))
    lam = float(dw.get('shrinkage_lambda', 0.3))
    alpha = float(dw.get('significance_alpha', 0.05))
    # 仅对显著因子调权(Bonferroni)：保留 auto_kill_negative 键语义，作为显著性门槛开关
    sig_gate = bool(dw.get('auto_kill_negative', True))

    if roll is None or used is None:
        roll, used = rolling_ic(db, '涨停TopN', cfg)
    if used < min_days:
        return None, used

    mat, _ = _daily_ic_matrix(db, '涨停TopN', cfg)
    mat = mat[[c for c in V2_KEYS if c in mat.columns]]
    if mat.empty:
        return None, used

    ema = {k: _ema(mat[k].dropna().tolist(), ema_window) for k in V2_KEYS if k in mat}
    sig = metrics.factor_significance(mat[[k for k in V2_KEYS if k in mat.columns]],
                                      alpha=alpha).set_index('策略')

    cur_w = {k: (cfg or {}).get('weights', {}).get(k, 0) for k in V2_KEYS}

    # shrinkage + 地板
    raw = {}
    for k in V2_KEYS:
        e = ema.get(k, float('nan'))
        significant = bool(sig.loc[k, '是否显著']) if k in sig.index else False
        if pd.isna(e) or e <= 0:
            raw[k] = 0.0
        elif sig_gate and not significant:
            raw[k] = 0.0          # 不显著 → 收缩至地板（不再归零）
        else:
            raw[k] = (1.0 - lam) * max(e, 0.0)
    total = sum(raw.values())
    if total <= 0:
        sug = {k: floor for k in V2_KEYS}
    else:
        sug = {k: floor + raw[k] for k in V2_KEYS}
    ssum = sum(sug.values())
    sug = {k: sug[k] / ssum for k in V2_KEYS}

    rows = []
    for k in V2_KEYS:
        e = ema.get(k, float('nan'))
        p = float(sig.loc[k, 'p']) if k in sig.index else float('nan')
        significant = bool(sig.loc[k, '是否显著']) if k in sig.index else False
        cw = round(float(cur_w.get(k, 0)) * 100)
        sw = round(float(sug[k]) * 100)
        if (not significant) and (not pd.isna(e)):
            action = '❌ 不显著(收缩至地板)'
        elif sw > cw * 1.05:
            action = '✅ 上调'
        else:
            action = '⚠️ 维持(噪声收缩至地板)'
        rows.append({'策略': k, 'EMA_IC': None if pd.isna(e) else round(e, 3),
                     'p值': None if pd.isna(p) else round(p, 4),
                     '是否显著': significant, '当前权重%': cw,
                     '建议权重%': sw, '动作': action})
    return pd.DataFrame(rows), used


# =============================== 市场环境建议（保留） ===============================
def regime_style_advice(cfg=None, timeout=15):
    """基于数据源（默认 AKShare 实时行情）判断当前市场环境，给出风格/清单配置建议。"""
    cfg = cfg or {}
    try:
        from extensions.datasources import get_source
        from extensions.market_regime import MarketRegimeExtension

        ext = MarketRegimeExtension()
        params = ext.effective_params(cfg.get('extensions', {}).get('market_regime', {}))
        ds_key = (cfg.get('extensions', {}).get('market_regime', {})
                  .get('datasource', 'akshare'))
        src = get_source(ds_key)
        if src is None:
            return 'unknown', [f'未找到数据源「{ds_key}」，请检查 extensions/datasources'], []

        class _Ctx:
            pass
        ctx = _Ctx()
        ctx.db = None
        ctx.base_dir = '.'
        ctx.data_dir = '.'
        ctx.cfg = cfg

        df, info = src.fetch(ctx)
        ctx.info = info
        if df is None or df.empty:
            return 'unknown', ['市场环境识别失败：数据源返回空（检查网络/接口可用性）'], []

        res = ext.run(df, params, ctx)
        main = res.get('主表')
        if main is None or main.empty:
            return 'unknown', ['市场环境识别未返回结果'], []
        row = main.iloc[0]
        regime = row['状态']
        note = row['建议']

        if regime == 'risk_on':
            lines = [
                f'【市场状态】risk_on（动量友好） | {note}',
                '【建议配置】',
                '  ✅ 主攻：涨停TopN、连板候选、尾盘选股',
                '  ⚠️  轻仓观察：涨幅4%',
                '  ❌ 暂停：抄底反弹、短线反转、防御低波（风险错配）',
            ]
            focus = ['涨停TopN', '连板候选', '尾盘选股']
        elif regime == 'panic':
            lines = [
                f'【市场状态】panic（恐慌） | {note}',
                '【建议配置】',
                '  ✅ 主攻：抄底反弹、短线反转（急跌修复）',
                '  🛡️  底仓：防御低波',
                '  ❌ 暂停：涨停TopN、连板候选、尾盘选股（动量策略失效）',
            ]
            focus = ['抄底清单', '短线反转', '防御低波']
        elif regime == 'risk_off':
            lines = [
                f'【市场状态】risk_off（反转友好） | {note}',
                '【建议配置】',
                '  ✅ 主攻：短线反转、抄底反弹、防御低波',
                '  ⚠️  观望：涨幅4%',
                '  ❌ 暂停：涨停TopN、连板候选、尾盘选股',
            ]
            focus = ['短线反转', '抄底清单', '防御低波']
        else:  # neutral
            lines = [
                f'【市场状态】neutral（震荡/磨底） | {note}',
                '【建议配置】',
                '  ✅ 轻仓试盘：涨幅4%、短线反转、尾盘选股',
                '  ⚠️  控制仓位，避免追涨杀跌',
            ]
            focus = ['涨幅4%', '短线反转', '尾盘选股']
        return regime, lines, focus
    except Exception as e:
        return 'unknown', [f'市场环境识别失败（{e}），请检查网络或AKShare版本'], []


# =============================== 调参建议（P0-1/P0-5/P0-6） ===============================
def make_advice(db, cfg):
    """调参建议引擎：基于滚动 IC + 因子相关性 + 增量IC，生成文字建议。

    返回 (建议文本list, 建议权重dict or None, 动态权重计划DataFrame or None)。
    新增段落：【策略IC显著性(Bonferroni)】【因子相关性(五因子)】【正交后增量IC】
    严禁"归零/kill/冻结"等自动写盘表述（权重仅用户显式点击时落盘）。
    """
    advice = []
    days = db.pending_backtest_dates()
    n_days = len(days)
    if n_days == 0:
        return ['暂无可回测数据：先「生成预测」，次日收盘数据入库后再回测。'], None, None

    # ---- 市场环境 + 风格建议 ----
    regime, regime_lines, _focus = regime_style_advice(cfg)
    advice.extend(regime_lines)
    advice.append('')

    # ---- 清单级表现 ----
    advice.append(f'【清单表现】（近 {n_days} 个可回测日）')
    for lt in LIST_TYPES:
        df, n = rolling_score(db, lt, cfg)
        if n == 0:
            continue
        avg = df['平均得分'].mean()
        tag = '✅' if avg >= 40 else ('⚠️' if avg >= 25 else '❌')
        advice.append(f'  {tag} {lt}：平均得分 {avg:.1f}（{n} 天）')
        if avg < 25:
            if lt == '涨停TopN':
                advice.append('      → 涨停清单整体偏弱：建议缩小 top_n 聚焦头部，或提高 S1封单强度 权重')
            elif lt == '连板候选':
                advice.append('      → 连板接力亏钱效应明显：建议收紧候选门槛（S1≥0.7），或降低该清单仓位')
            elif lt == '涨幅4%':
                advice.append('      → 低吸清单表现差：建议检查动量区间（3%~8%）是否匹配当前市场情绪')
            elif lt == '尾盘选股':
                advice.append('      → 尾盘溢价不稳定：建议提高「强势」因子权重、避开高位股')

    # ---- 策略级 IC（基于涨停TopN 清单的 detail 拆包）----
    roll, used = rolling_ic(db, '涨停TopN', cfg)
    if roll.empty or used < 1:
        advice.append('【策略IC】样本不足（需预测时含五策略分明细，且至少1天可回测）')
        return advice, None, None
    advice.append(f'【策略IC】（基于涨停TopN清单，{used} 天；含 CI95 与 样本n）')
    cur_w = {k: (cfg or {}).get('weights', {}).get(k, 0) for k in V2_KEYS}
    seen = set()
    for _, r in roll.iterrows():
        k = r['策略']
        if k not in V2_KEYS:
            continue
        seen.add(k)
        w_pct = round(cur_w.get(k, 0) * 100)
        if r['平均IC'] <= 0:
            advice.append(f'  ❌ {k}：平均IC={r["平均IC"]:+.3f}（IC>0 {r["IC>0天数"]}，'
                          f'CI95[{r["CI95低"]:+.2f},{r["CI95高"]:+.2f}]，n={r["样本n"]}）'
                          f'→ 反向拖后腿，建议下调权重')
        elif r['平均IC'] < 0.2:
            advice.append(f'  ⚠️ {k}：平均IC={r["平均IC"]:+.3f}（CI95[{r["CI95低"]:+.2f},'
                          f'{r["CI95高"]:+.2f}]，n={r["样本n"]}）→ 效果不稳，建议维持或小幅下调'
                          f'（当前 {w_pct}%）')
        else:
            advice.append(f'  ✅ {k}：平均IC={r["平均IC"]:+.3f}（CI95[{r["CI95低"]:+.2f},'
                          f'{r["CI95高"]:+.2f}]，n={r["样本n"]}）→ 有效，可适度上调（当前 {w_pct}%）')
    for k in V2_KEYS:
        if k not in seen:
            advice.append(f'  ℹ️ {k}：得分几乎无区分度，IC 无法计算 → 维持现状观察'
                          f'（当前 {round(cur_w.get(k, 0) * 100)}%）')

    # ---- 策略IC 显著性（Bonferroni 多重比较校正）----
    advice.append('')
    advice.append('【策略IC显著性(Bonferroni)】单样本t检验 + α/因子数 校正（防多重比较假显著）')
    mat, _ = _daily_ic_matrix(db, '涨停TopN', cfg)
    mat = mat[[c for c in V2_KEYS if c in mat.columns]]
    if not mat.empty:
        sig = metrics.factor_significance(mat, alpha=0.05)
        for _, r in sig.iterrows():
            mark = '✅显著' if r['是否显著'] else '—不显著'
            advice.append(f'  {r["策略"]}: 均值IC={r["均值IC"]:+.3f}  '
                          f'CI95[{r["CI95低"]:+.2f},{r["CI95高"]:+.2f}]  '
                          f'p={r["p"]:.3f}  α_b={r["Bonferroniα"]}  {mark}  n={r["样本天数n"]}')

    # ---- 因子相关性（五因子 Spearman 相关矩阵，跨日均值）----
    advice.append('')
    advice.append('【因子相关性(五因子)】Spearman 相关矩阵（跨回测日截面均值，共线参考）')
    corr = metrics.factor_correlation(db, cfg, method='spearman')
    if not corr.empty:
        header = '         ' + ''.join(f'{c:>8}' for c in corr.columns)
        advice.append('  ' + header)
        for idx in corr.index:
            row = ''.join(f'{corr.loc[idx, c]:>8.2f}' for c in corr.columns)
            advice.append(f'  {idx:>8}' + row)

    # ---- 正交后增量 IC ----
    advice.append('')
    advice.append('【正交后增量IC】以实际涨幅为因、其余4因子回归得残差，再算该因子 vs 残差 的 Spearman IC')
    inc = metrics.incremental_ic(db, '涨停TopN', cfg)
    if not inc.empty:
        for _, r in inc.iterrows():
            advice.append(f'  {r["因子"]}: 原始IC={r["IC(原始)"]:+.3f}  '
                          f'增量IC={r["增量IC"]:+.3f}  p={r["增量IC_p"]:.3f}  '
                          f'独立增量贡献={r["独立增量贡献%"]:.1f}%')
        advice.append('  （独立增量贡献%：该因子在剔除其余4因子后的增量解释力占比；'
                      '某因子"对角"增量最高说明其信息最独立）')

    # ---- 动态权重计划（shrinkage + EMA + 显著性）----
    advice.append('')
    plan, _ = dynamic_weight_plan(db, cfg, roll, used)
    sug = None
    dw = (cfg or {}).get('dynamic_weights', {})
    if plan is None:
        if not dw.get('enabled', True):
            advice.append('【动态权重】已关闭（config.dynamic_weights.enabled=false）')
        else:
            advice.append(f'【动态权重】样本 {used} 天 < {dw.get("min_days", 3)} 天，'
                          f'暂不出数值建议；继续积累回测天数。')
    else:
        floor = float(dw.get('floor', 0.02))
        lam = float(dw.get('shrinkage_lambda', 0.3))
        sig_gate = bool(dw.get('auto_kill_negative', True))
        sug = {r['策略']: round(r['建议权重%'] / 100, 4) for _, r in plan.iterrows()}
        advice.append(f'【动态权重建议】{used} 天 | 地板{floor*100:.0f}% | '
                      f'λ={lam} shrinkage | '
                      f'{"仅显著因子调权(Bonferroni)" if sig_gate else "全因子按EMA_IC加权"}'
                      f' | 可一键应用到「策略参数」页（须用户显式点击，不自动冻结）')
        for _, r in plan.iterrows():
            advice.append(f'  {r["策略"]}: {r["当前权重%"]}% → 建议 {r["建议权重%"]}%  '
                          f'[{r["动作"]}]  (EMA_IC={r["EMA_IC"]}, p={r["p值"]})')
    return advice, sug, plan
