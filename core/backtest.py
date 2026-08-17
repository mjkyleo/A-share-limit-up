# -*- coding: utf-8 -*-
"""
回测模块（backtest）v2.1
逻辑：predictions(run_date→target_date) ⋈ daily(target_date 实际数据)
产出：
  1. 整体准确率：Top3/5/10 命中率、平均实际涨幅、基准对比
  2. 清单得分：每张清单独立打分规则（涨停满分/按涨幅分档/下跌扣分），
     得分规则见 SCORING_RULES，可在 config.json 的 "scoring" 节覆盖
  3. 单策略表现：每个策略分 vs 实际涨幅 的 Spearman IC（分离评估哪个策略有效）
  4. 滚动历史：多日累计 IC 均值 + 清单滚动平均得分
  5. 调参建议 make_advice()：依据滚动 IC 与清单得分自动生成权重调整建议
"""
import json

import numpy as np
import pandas as pd

STRATEGY_KEYS = ['S1封单强度', 'S2封板质量', 'S3锁仓度', 'S4资金', 'S5股性结构',
                 'A封单强度', 'B封板时间', 'C量能换手', 'D市值', 'E资金']  # 兼容v1命名

# v2 策略键 → 权重键 映射（建议引擎用）
V2_KEYS = ['S1封单强度', 'S2封板质量', 'S3锁仓度', 'S4资金', 'S5股性结构']

# ---------------- 清单打分规则 ----------------
# 规则格式: [[涨幅下限%, 得分], ...] 按下限升序，取满足"实际涨幅 ≥ 下限"的最后一档。
# 设计原则：达成清单目标 → 满分100；接近目标 → 分档给分；下跌 → 按幅度扣分（目标越激进扣得越狠）。
DEFAULT_SCORING = {
    # 目标：次日涨停。涨停100；涨6%+大肉75；小涨微利20~60；跌2%内小亏-10；跌6%+大面-80
    '涨停TopN': [[-99, -80], [-6, -50], [-4, -30], [-2, -10], [0, 20],
                 [2, 40], [4, 60], [6, 75], [9.5, 100]],
    # 目标：次日涨≥4%。达成即100；涨2~4%接近40；下跌扣分（低吸半路风险小于打板，扣分温和）
    '涨幅4%': [[-99, -60], [-6, -40], [-4, -25], [-2, -10], [0, 15], [2, 40], [4, 100]],
    # 目标：连板（已涨停股再涨停）。连板100；冲高未板50；断板亏钱快，扣分最狠
    '连板候选': [[-99, -100], [-6, -70], [-4, -45], [-2, -20], [0, 20], [4, 50], [9.5, 100]],
    # 目标：尾盘买入次日溢价。≥2%即合格60；涨停100；跌2%内-15（尾盘买成本低，扣分温和）
    '尾盘选股': [[-99, -60], [-4, -35], [-2, -15], [0, 30], [2, 60], [4, 80], [9.5, 100]],
    # 新增扩展清单（extensions 亦可声明并覆盖）
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
# 扩展（extensions/）可声明自己的 list_type / scoring / hit_threshold / IC 键，
# 在此自动并入，使「回测评估」下拉框、打分、达成率、单因子 IC 对其同样生效。
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


def backtest_one(db, target_date, list_type='涨停TopN', cfg=None):
    """单日回测 → (明细df, 汇总dict, 策略IC dict)
    明细df 增加 得分 列（按清单规则打分）；汇总含 平均得分/总得分/达成率。"""
    pred = db.get_predictions(target_date)
    pred = pred[pred['list_type'] == list_type]
    if pred.empty:
        return None, None, None
    actual = db.get_daily(target_date)
    if actual.empty:
        return None, None, None
    mg = pred.merge(actual[['代码6', '涨幅', '今日涨停']],
                    left_on='code', right_on='代码6', how='inner')
    if len(mg) < 3:
        return None, None, None
    mg = mg.sort_values('rank').reset_index(drop=True)
    mg = mg.rename(columns={'涨幅': '实际涨幅%', '今日涨停': '实际涨停'})

    rules = _scoring(cfg, list_type)
    mg['得分'] = mg['实际涨幅%'].apply(lambda x: score_row(x, rules))
    hit_th = HIT_THRESHOLD.get(list_type, 9.5)

    summary = {'目标日': target_date, '清单': list_type, '预测数': len(pred), '匹配数': len(mg)}
    for n in (3, 5, 10):
        t = mg.head(min(n, len(mg)))
        summary[f'Top{n}涨停率'] = f"{t['实际涨停'].mean():.0%}"
        summary[f'Top{n}平均涨幅'] = f"{t['实际涨幅%'].mean():.2f}%"
    summary['全样本平均涨幅'] = f"{mg['实际涨幅%'].mean():.2f}%"
    summary['平均得分'] = f"{mg['得分'].mean():.1f}"
    summary['总得分'] = f"{mg['得分'].sum():.0f}"
    summary['达成率'] = f"{(mg['实际涨幅%'] >= hit_th).mean():.0%}（涨幅≥{hit_th}%）"

    # 单策略 IC（从 detail JSON 拆出策略分）
    ic = {}
    details = mg['detail'].apply(lambda s: json.loads(s) if s else {})
    for key in STRATEGY_KEYS:
        vals = details.apply(lambda d: d.get(key))
        vals = pd.to_numeric(vals, errors='coerce')
        mask = vals.notna() & mg['实际涨幅%'].notna()
        if mask.sum() >= 5 and vals[mask].nunique() > 2:
            rho, p = _spearman(vals[mask], mg.loc[mask, '实际涨幅%'])
            if rho is not None:
                ic[key] = {'IC(ρ)': rho, 'p值': round(p, 4),
                           '评价': '✅ 有效' if rho > 0.4 and p < 0.1
                           else ('⚠️ 弱' if rho > 0 else '❌ 无效/反向')}
    return mg, summary, ic


def rolling_ic(db, list_type='涨停TopN', cfg=None):
    """全部可回测日期的策略 IC 汇总（均值+次数）→ (DataFrame, 天数)"""
    acc = {}
    days = db.pending_backtest_dates()
    used = 0
    for d in days:
        _, _, ic = backtest_one(db, d, list_type, cfg)
        if not ic:
            continue
        used += 1
        for k, v in ic.items():
            acc.setdefault(k, []).append(v['IC(ρ)'])
    if not acc:
        return pd.DataFrame(), 0
    rows = [{'策略': k, '平均IC': round(float(np.mean(v)), 3),
             'IC>0天数': f"{sum(1 for x in v if x > 0)}/{len(v)}",
             '评价': '✅ 稳定有效' if np.mean(v) > 0.3 else ('⚠️ 不稳定' if np.mean(v) > 0 else '❌ 建议下调权重')}
            for k, v in acc.items()]
    return pd.DataFrame(rows).sort_values('平均IC', ascending=False), used


def rolling_score(db, list_type, cfg=None):
    """清单滚动得分汇总 → (DataFrame, 天数)。列：日期/匹配数/平均得分/总得分/达成率"""
    rows = []
    for d in db.pending_backtest_dates():
        _, summary, _ = backtest_one(db, d, list_type, cfg)
        if not summary:
            continue
        rows.append({'日期': d, '匹配数': summary['匹配数'],
                     '平均得分': float(summary['平均得分']),
                     '总得分': float(summary['总得分']),
                     '达成率': summary['达成率']})
    if not rows:
        return pd.DataFrame(), 0
    return pd.DataFrame(rows), len(rows)


def dynamic_weight_plan(db, cfg, roll=None, used=None):
    """逐因子动态权重计划。返回 (DataFrame 或 None, used天数)。
    列：策略 | 滚动IC | IC>0天数 | 当前权重% | 建议权重% | 动作

    规则（见 config.dynamic_weights）：
      - enabled=False        → 直接返回 None（不输出建议）
      - 负IC因子 + auto_kill_negative=True → 建议权重=0（硬 kill 反向因子）
      - 否则地板 floor（默认2%）
      - 全负（total_pos<=0） → 空仓信号，全部归零
      - 其余按 max(IC,0) 归一化 + 地板
    注：受 min_days 约束，样本不足也返回 None。
    """
    dw = (cfg or {}).get('dynamic_weights', {})
    if not dw.get('enabled', True):
        return None, used if used is not None else 0
    floor = float(dw.get('floor', 0.02))
    kill_neg = bool(dw.get('auto_kill_negative', True))
    min_days = int(dw.get('min_days', 3))

    if roll is None or used is None:
        roll, used = rolling_ic(db, '涨停TopN', cfg)
    if roll.empty or used < min_days:
        return None, used

    cur_w = {k: (cfg or {}).get('weights', {}).get(k, 0) for k in V2_KEYS}
    mean_ic = {}
    ic_pos_days = {}
    for _, r in roll.iterrows():
        k = r['策略']
        if k not in V2_KEYS:
            continue
        mean_ic[k] = r['平均IC']
        ic_pos_days[k] = r['IC>0天数']

    # 判定硬kill（负IC因子 → 权重 0）+ 原始IC（>=0）
    def _ic(k):
        v = mean_ic.get(k, float('nan'))
        return float('nan') if pd.isna(v) else float(v)

    killed = {k: (kill_neg and (pd.isna(mean_ic.get(k)) or mean_ic.get(k) <= 0))
              for k in V2_KEYS}
    raw = {k: (0.0 if killed[k] else max(_ic(k), 0.0)) for k in V2_KEYS}
    total_raw = sum(raw.values())

    sug = {}
    if total_raw <= 0:
        # 全部反向/无效：空仓信号，权重全 0
        sug = {k: 0.0 for k in V2_KEYS}
    elif kill_neg:
        active = [k for k in V2_KEYS if not killed[k]]
        scale = (1 - floor * len(active)) / total_raw
        for k in V2_KEYS:
            sug[k] = 0.0 if killed[k] else floor + raw[k] * scale
    else:
        # 不硬kill：负IC也给地板，按 max(IC,0) 归一
        scale = (1 - floor * len(V2_KEYS)) / total_raw
        for k in V2_KEYS:
            sug[k] = floor + raw[k] * scale

    rows = []
    for k in V2_KEYS:
        ic = _ic(k)
        if killed[k]:
            action = '❌ 归零(kill)'
        elif ic < 0.2:
            action = '⚠️ 维持/下调'
        else:
            action = '✅ 上调'
        rows.append({
            '策略': k,
            '滚动IC': None if pd.isna(ic) else round(ic, 3),
            'IC>0天数': ic_pos_days.get(k, '-'),
            '当前权重%': round(float(cur_w.get(k, 0)) * 100),
            '建议权重%': round(float(sug[k]) * 100),
            '动作': action,
        })
    return pd.DataFrame(rows), used


def regime_style_advice(cfg=None, timeout=15):
    """基于数据源（默认 AKShare 实时行情）判断当前市场环境，给出风格/清单配置建议。
    走 extensions.datasources 抽象：数据源由
      cfg['extensions']['market_regime']['datasource'] 决定（默认 akshare）。
    返回 (regime_label, text_lines, list_focus)。失败返回 ('unknown', ['无法获取市场状态'], [])。
    """
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

        # 构造最小 ctx 并通过数据源 fetch 获取数据（含市场活跃度）
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

        # 风格配置建议
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


def make_advice(db, cfg):
    """调参建议引擎：基于全部可回测日的滚动 IC + 清单滚动得分，生成文字建议。
    返回 (建议文本list, 建议权重dict or None, 动态权重计划DataFrame or None)。
    规则：
      滚动IC ≤ 0      → 该策略反向/无效，建议下调（给出具体数值）
      0 < IC < 0.2    → 不稳，维持或小降
      IC ≥ 0.2        → 有效，可上调
      建议权重 = max(IC,0) 归一化（地板2%），样本<3天不出数值建议
      清单平均分 < 25 → 该清单整体表现差，给出口径级建议
    """
    advice = []
    days = db.pending_backtest_dates()
    n_days = len(days)
    if n_days == 0:
        return ['暂无可回测数据：先「生成预测」，次日收盘数据入库后再回测。'], None, None

    # ---- 市场环境 + 风格建议（基于 AKShare 实时数据） ----
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

    # ---- 策略级 IC（基于涨停TopN清单的 detail 拆包） ----
    roll, used = rolling_ic(db, '涨停TopN', cfg)
    if roll.empty or used < 1:
        advice.append('【策略IC】样本不足（需预测时含五策略分明细，且至少1天可回测）')
        return advice, None, None
    advice.append(f'【策略IC】（基于涨停TopN清单，{used} 天）')
    cur_w = {k: (cfg or {}).get('weights', {}).get(k, 0) for k in V2_KEYS}
    seen = set()
    for _, r in roll.iterrows():
        k = r['策略']
        if k not in V2_KEYS:
            continue
        seen.add(k)
        w_pct = round(cur_w.get(k, 0) * 100)
        if r['平均IC'] <= 0:
            advice.append(f'  ❌ {k}：滚动IC={r["平均IC"]:+.3f}（IC>0 {r["IC>0天数"]}）'
                          f'→ 反向拖后腿，建议归零/下调')
        elif r['平均IC'] < 0.2:
            advice.append(f'  ⚠️ {k}：滚动IC={r["平均IC"]:+.3f} → 效果不稳，建议维持或小幅下调（当前 {w_pct}%）')
        else:
            advice.append(f'  ✅ {k}：滚动IC={r["平均IC"]:+.3f} → 有效，可适度上调（当前 {w_pct}%）')
    for k in V2_KEYS:
        if k not in seen:
            advice.append(f'  ℹ️ {k}：得分几乎无区分度，IC 无法计算 → 维持现状观察（当前 {round(cur_w.get(k, 0) * 100)}%）')

    # ---- 动态权重计划（基于滚动IC，自动归零负IC因子） ----
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
        kill = dw.get('auto_kill_negative', True)
        floor = float(dw.get('floor', 0.02))
        sug = {r['策略']: round(r['建议权重%'] / 100, 4) for _, r in plan.iterrows()}
        advice.append(f'【动态权重建议】{used} 天滚动IC | 地板{floor*100:.0f}% | '
                      f'负IC{"归零" if kill else f"{floor*100:.0f}%地板"} | 可一键应用到「策略参数」页')
        for _, r in plan.iterrows():
            advice.append(f'  {r["策略"]}: {r["当前权重%"]}% → 建议 {r["建议权重%"]}%  [{r["动作"]}]')
    return advice, sug, plan
