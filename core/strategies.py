# -*- coding: utf-8 -*-
"""
策略引擎（strategies）
所有参数集中在 DEFAULT_CONFIG，GUI 可修改并保存到 config.json。

v2 权重依据（08-13 实盘回测，n=16 一进二组，策略分 vs 次日涨幅 Spearman ρ）：
  封单强度 ρ=+0.598（显著）→ 0.40 | 封板质量 ρ=+0.448 → 0.25
  锁仓度（重构）→ 0.15 | 资金 ρ=-0.207（反向，降权观察）→ 0.05 | 股性结构 → 0.15
"""
import json
import os

import numpy as np
import pandas as pd

DEFAULT_CONFIG = {
    # ---- 涨停预测策略权重（五策略之和自动归一） ----
    'weights': {'S1封单强度': 0.40, 'S2封板质量': 0.25, 'S3锁仓度': 0.15,
                'S4资金': 0.05, 'S5股性结构': 0.15},
    # ---- 先验概率（文献统计基准） ----
    'base_limit_up': 0.15,      # 一进二无条件成功率 12%~19%
    'base_rush': 0.04,          # 非涨停股次日冲板基准
    'base_rise4': 0.10,         # 任意股次日涨幅≥4%基准（经验值，可回测校准）
    # ---- 阈值 ----
    'thresholds': {
        '涨停判定涨幅': 9.5,        # 主板10%口径，创业板/科创板20cm需另行区分
        '封流比_强': 3.0, '封流比_极强': 5.0,
        '封成比_强': 3.0, '封成比_极强': 10.0,
        '锁仓换手_极紧': 3.0, '锁仓换手_紧': 8.0, '锁仓换手_松': 15.0,
        '小市值上限': 80,           # 亿元
        '高位20日涨幅': 60.0,       # 位置风险线 %
        '开板次数_风险': 3,
    },
    # ---- 输出 ----
    'top_n': 20,                # 明日涨停清单长度
    'rise4_n': 20,              # 涨幅≥4%清单长度
    'monday_n': 8,              # 跨日连板候选长度
    'tail_n': 15,               # 尾盘选股清单长度
    # ---- 尾盘选股因子权重（自动归一） ----
    'tail_weights': {
        '强势': 0.35,           # 涨停股=S1封单强度；未涨停=动量×量能合成
        '资金': 0.25,           # 主力净量 + 资金流向
        '量能': 0.20,           # 量比（温和放大最佳）
        '位置': 0.20,           # 20日涨幅低 + 小市值
    },
    # ---- 回测评分规则（次日实际涨幅% → 得分；[涨幅下限, 得分] 升序，满足的最高档生效） ----
    'scoring': {
        # 目标：次日涨停。涨停100；涨6%+大肉75；小涨微利；跌6%+大面-80
        '涨停TopN': [[-99, -80], [-6, -50], [-4, -30], [-2, -10], [0, 20],
                     [2, 40], [4, 60], [6, 75], [9.5, 100]],
        # 目标：次日涨≥4%。达成即100；涨2~4%接近40；下跌扣分温和
        '涨幅4%': [[-99, -60], [-6, -40], [-4, -25], [-2, -10], [0, 15], [2, 40], [4, 100]],
        # 目标：连板（已涨停股再涨停）。连板100；冲高未板50；断板亏钱快扣分最狠
        '连板候选': [[-99, -100], [-6, -70], [-4, -45], [-2, -20], [0, 20], [4, 50], [9.5, 100]],
        # 目标：尾盘买入次日溢价。≥2%合格60；涨停100；跌2%内-15
        '尾盘选股': [[-99, -60], [-4, -35], [-2, -15], [0, 30], [2, 60], [4, 80], [9.5, 100]],
    },
    # ---- 交易日历补充（额外闭市日 ['20261225', ...]） ----
    'extra_holidays': [],
}


def load_config(base_dir):
    path = os.path.join(base_dir, 'config.json')
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path):
        try:
            user = json.load(open(path, encoding='utf-8'))
            for k, v in user.items():
                if isinstance(v, dict) and k in cfg:
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    return cfg


def save_config(base_dir, cfg):
    path = os.path.join(base_dir, 'config.json')
    json.dump(cfg, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


# =============================== 涨停预测五策略 ===============================
def s1_seal(r, th):
    if not r['今日涨停']:
        return 0.0
    s, n = 0.0, 0
    if pd.notna(r.get('封流比')):
        fr = r['封流比']
        s += 1.0 if fr >= th['封流比_极强'] else 0.8 if fr >= th['封流比_强'] \
            else 0.6 if fr >= 1 else 0.2
        n += 1
    if pd.notna(r.get('封成比')):
        fc = r['封成比']
        s += 1.0 if fc >= th['封成比_极强'] else 0.75 if fc >= th['封成比_强'] \
            else 0.4 if fc >= 1 else 0.1
        n += 1
    return s / n if n else 0.0


def s2_quality(r, th):
    if not r['今日涨停']:
        return 0.0
    if pd.isna(r.get('封板分钟')):
        return 0.5                       # 无封板时间字段 → 中性分
    t = r['封板分钟']
    ts = 1.0 if t <= 0 else 0.9 if t <= 5 else 0.7 if t <= 30 \
        else 0.45 if t <= 60 else 0.2 if t <= 270 else 0.05
    if pd.notna(r.get('开板次数')):
        k = r['开板次数']
        ks = 1.0 if k == 0 else 0.6 if k == 1 else 0.3 if k == 2 else 0.1
        return 0.6 * ts + 0.4 * ks
    return ts


def s3_lockup(r, th):
    """锁仓度：换手/振幅/量比越低 → 筹码越稳 → 连板延续越强（回测重构结论）"""
    if not r['今日涨停']:
        return 0.0
    s, n = 0.0, 0
    if pd.notna(r.get('换手')):
        h = r['换手']
        s += 1.0 if h < th['锁仓换手_极紧'] else 0.75 if h < th['锁仓换手_紧'] \
            else 0.5 if h < th['锁仓换手_松'] else 0.25 if h < 25 else 0.05
        n += 1
    if pd.notna(r.get('振幅')):
        a = r['振幅'] * 100 if r['振幅'] < 1 else r['振幅']
        s += 1.0 if a <= 0.5 else 0.8 if a <= 3 else 0.5 if a <= 7 else 0.2
        n += 1
    if pd.notna(r.get('量比')):
        lb = r['量比']
        s += 1.0 if lb < 0.8 else 0.7 if lb < 1.5 else 0.4 if lb < 3 else 0.2
        n += 1
    return s / n if n else 0.0


def s4_money(r, th):
    """资金（回测呈弱反向→低权重观察）：主力净量+增仓占比+机构动向"""
    if not r['今日涨停']:
        return 0.0
    s, n = 0.0, 0
    if pd.notna(r.get('主力净量')):
        z = r['主力净量']
        s += 1.0 if z >= 2 else 0.75 if z >= 1 else 0.5 if z >= 0 else 0.1
        n += 1
    if pd.notna(r.get('增仓占比')):
        z = r['增仓占比']
        s += 1.0 if z >= 2 else 0.7 if z >= 0 else 0.2
        n += 1
    if pd.notna(r.get('机构动向')):
        z = r['机构动向']
        s += 1.0 if z >= 2 else 0.7 if z >= 0 else 0.2
        n += 1
    return s / n if n else 0.0


def s5_structure(r, th):
    """股性结构：小市值 + 年涨停次数(股性) + 位置(20日涨幅) + 次新"""
    s, n = 0.0, 0
    if pd.notna(r.get('流通市值')):
        y = r['流通市值'] / 1e8
        s += 1.0 if y < 30 else 0.85 if y < 50 else 0.65 if y < th['小市值上限'] \
            else 0.4 if y < 150 else 0.2 if y < 300 else 0.05
        n += 1
    if pd.notna(r.get('年涨停数')):
        c = r['年涨停数']
        s += 1.0 if c >= 10 else 0.75 if c >= 5 else 0.5 if c >= 2 else 0.25
        n += 1
    if pd.notna(r.get('20日涨幅')):
        g = r['20日涨幅']
        s += 1.0 if g < 20 else 0.7 if g < 40 else 0.4 if g < th['高位20日涨幅'] else 0.1
        n += 1
    return s / n if n else 0.0


def m_buyability(r, th):
    """可买性乘子：一字/秒板+巨单封死 → 次日开盘买不进"""
    if not r['今日涨停']:
        return 0.9
    b = 1.0
    t = r.get('封板分钟')
    if pd.notna(t):
        b *= 0.10 if t <= 0 else 0.35 if t <= 0.5 else 0.55 if t <= 2 \
            else 0.75 if t <= 5 else 0.9
    fc = r.get('封成比')
    if pd.notna(fc):
        b *= 0.25 if fc > 100 else 0.5 if fc > 30 else 0.75 if fc > 15 else 1.0
    if pd.notna(r.get('开盘涨幅')) and r['开盘涨幅'] > 8:
        b *= 0.8
    if pd.notna(r.get('换手')) and r['换手'] < 2:
        b *= 0.6
    return round(min(b, 1.0), 3)


def risk_penalty(r, th):
    p = 1.0
    if r.get('亏损股', False):
        p *= 0.85
    if pd.notna(r.get('开板次数')) and r['开板次数'] >= th['开板次数_风险']:
        p *= 0.6
    if pd.notna(r.get('换手')) and r['换手'] > 25:
        p *= 0.7
    if pd.notna(r.get('20日涨幅')) and r['20日涨幅'] > th['高位20日涨幅']:
        p *= 0.6
    if 'ST' in str(r.get('名称', '')):
        p *= 0.5
    return round(p, 3)


# =============================== 涨幅≥4% 预测器 ===============================
def rise4_score(r, th):
    """次日涨幅≥4%概率评分（面向全部股票，不要求今日涨停）
    因子：动量(今日涨幅3~8%最佳) + 量能(量比1.5~3) + 资金(流向/净量) +
          竞价强弱 + 位置(5日涨幅不过热) + 小市值"""
    s, n = 0.0, 0
    if pd.notna(r.get('涨幅')):
        z = r['涨幅']
        s += 1.0 if 3 <= z <= 8 else 0.8 if z >= 9.5 else 0.6 if 1 <= z < 3 \
            else 0.3 if 0 <= z < 1 else 0.1
        n += 2                                # 动量权重×2
    if pd.notna(r.get('量比')):
        lb = r['量比']
        s += 1.0 if 1.5 <= lb <= 3 else 0.7 if 1 <= lb < 1.5 else 0.3 if lb < 1 else 0.5
        n += 1
    if pd.notna(r.get('资金流向')):
        s += 1.0 if r['资金流向'] > 0 else 0.2
        n += 1
    if pd.notna(r.get('主力净量')):
        z = r['主力净量']
        s += 1.0 if z >= 1 else 0.6 if z >= 0 else 0.2
        n += 1
    if pd.notna(r.get('开盘涨幅')):
        og = r['开盘涨幅']
        s += 1.0 if 1 <= og <= 5 else 0.6 if 0 <= og < 1 else 0.3
        n += 1
    if pd.notna(r.get('5日涨幅')):
        g = r['5日涨幅']
        s += 1.0 if 0 <= g <= 20 else 0.5 if g < 0 else 0.2   # 过热减分
        n += 1
    if pd.notna(r.get('流通市值')):
        y = r['流通市值'] / 1e8
        s += 1.0 if y < 80 else 0.6 if y < 200 else 0.3
        n += 1
    return s / n if n else 0.0


# =============================== 尾盘选股（T日尾盘买入，博T+1溢价） ===============================
def tail_factors(r, th):
    """尾盘选股四因子（输出均为 [0,1]）。
    双轨候选：
      已涨停  —— 封单稳 → 次日惯性溢价（强势=S1封单强度）
      未涨停  —— 涨4%~9% + 放量 + 资金流入 → 尾盘可能封板 / 次日冲高（强势=动量×量能）
    """
    f = {}
    is_up = bool(r.get('今日涨停', False))
    zf = r.get('涨幅')
    # ---- 强势 ----
    if is_up:
        f['强势'] = s1_seal(r, th)                        # 封单强度直接复用
    elif pd.notna(zf):
        mom = 1.0 if 5 <= zf < 9.5 else 0.75 if 3 <= zf < 5 \
            else 0.5 if 1 <= zf < 3 else 0.2 if 0 <= zf < 1 else 0.0
        lb = r.get('量比')
        vol = (1.0 if 1.5 <= lb <= 4 else 0.6 if 0.8 <= lb < 1.5
               else 0.3 if lb < 0.8 else 0.5) if pd.notna(lb) else 0.5
        f['强势'] = round(mom * 0.7 + vol * 0.3, 3)
    else:
        f['强势'] = 0.0
    # ---- 资金（尾盘更看重当日真金白银） ----
    s, n = 0.0, 0
    if pd.notna(r.get('主力净量')):
        z = r['主力净量']
        s += 1.0 if z >= 1 else 0.7 if z >= 0 else 0.2
        n += 1
    if pd.notna(r.get('资金流向')):
        s += 1.0 if r['资金流向'] > 0 else 0.2
        n += 1
    f['资金'] = round(s / n, 3) if n else 0.5
    # ---- 量能（量比温和放大最佳，爆量见顶减分） ----
    lb = r.get('量比')
    f['量能'] = (1.0 if 1.5 <= lb <= 4 else 0.7 if 0.8 <= lb < 1.5
                 else 0.4 if lb < 0.8 else 0.2) if pd.notna(lb) else 0.5
    # ---- 位置（低位的次日空间大，高位接力风险大） ----
    s, n = 0.0, 0
    if pd.notna(r.get('20日涨幅')):
        g = r['20日涨幅']
        s += 1.0 if g < 20 else 0.7 if g < 40 else 0.3 if g < th['高位20日涨幅'] else 0.05
        n += 1
    if pd.notna(r.get('流通市值')):
        y = r['流通市值'] / 1e8
        s += 1.0 if y < 50 else 0.7 if y < th['小市值上限'] else 0.4 if y < 200 else 0.15
        n += 1
    f['位置'] = round(s / n, 3) if n else 0.5
    return f


def run_tail_session(df, cfg):
    """尾盘选股主流程：盘中快照 → 四因子打分 → ×M可买性 ×R风险 → 清单。
    返回 (清单df, 备注str)。清单带 尾盘评分 / 预估次日溢价概率% / 类型(涨停惯性|强势未封)。
    """
    th = cfg['thresholds']
    df = df.copy()
    if '涨幅' not in df.columns:
        return pd.DataFrame(), '盘中快照缺少「涨幅」列，无法选股'
    if '今日涨停' not in df.columns:
        df['今日涨停'] = df['涨幅'] >= th['涨停判定涨幅']
    # 只保留有上涨动能的候选：已涨停 或 涨幅≥1%（阴跌股无尾盘博弈价值）
    pool = df[(df['今日涨停']) | (df['涨幅'].fillna(-99) >= 1)].copy()
    if pool.empty:
        return pd.DataFrame(), '今日无强势候选（全部弱势），建议空仓观望'
    fac = pool.apply(lambda r: pd.Series(tail_factors(r, th)), axis=1)
    for c in fac.columns:
        pool[c] = fac[c]
    w = pd.Series(cfg['tail_weights'])
    w = w / w.sum()
    pool['尾盘评分'] = (sum(pool[c] * w[c] for c in w.index)).round(3)
    pool['M可买性'] = pool.apply(lambda r: m_buyability(r, th), axis=1)
    pool['R风险系数'] = pool.apply(lambda r: risk_penalty(r, th), axis=1)
    pool['尾盘评分'] = (pool['尾盘评分'] * pool['M可买性'] * pool['R风险系数']).round(3)
    pool['类型'] = np.where(pool['今日涨停'], '涨停惯性', '强势未封')
    pool['预估次日溢价概率%'] = (np.minimum(0.08 + pool['尾盘评分'] * 0.4, 0.55) * 100).round(1)
    res = pool.sort_values('尾盘评分', ascending=False).head(
        cfg.get('tail_n', 15)).reset_index(drop=True)
    res.insert(0, '排名', res.index + 1)
    n_up = int(res['今日涨停'].sum())
    note = f'候选池{len(pool)}只（涨停{int(pool["今日涨停"].sum())}只），入选{len(res)}只：涨停惯性{n_up}只 + 强势未封{len(res) - n_up}只'
    return res, note



STRAT_FNS = {'S1封单强度': s1_seal, 'S2封板质量': s2_quality, 'S3锁仓度': s3_lockup,
             'S4资金': s4_money, 'S5股性结构': s5_structure}


def run_prediction(df, cfg):
    """对合并数据打分 → dict(涨停TopN=df, 涨幅4%清单=df, 连板候选=df, 全量=df)"""
    th = cfg['thresholds']
    df = df.copy()
    if '今日涨停' not in df.columns:
        df['今日涨停'] = df['涨幅'] >= th['涨停判定涨幅']
    for name, fn in STRAT_FNS.items():
        df[name] = df.apply(lambda r: fn(r, th), axis=1).round(3)

    w = pd.Series(cfg['weights'])
    w = w / w.sum()
    # 冲板组用动能替代 S3
    if '涨幅' in df.columns:
        mom = df['涨幅'].clip(lower=0) / 10
        mask = ~df['今日涨停']
        lb = df['量比'] if '量比' in df.columns else pd.Series(1.0, index=df.index)
        df.loc[mask, 'S3锁仓度'] = (mom[mask] * 0.6 +
                                    lb[mask].fillna(1).clip(0, 3) / 3 * 0.4).round(3)
    # P1 加权 / P2 Borda / P3 先验概率
    df['P1加权分'] = sum(df[c] * w[c] for c in w.index).round(4)
    borda = pd.DataFrame(index=df.index)
    for grp, sub in df.groupby('今日涨停'):
        n = len(sub)
        for c in w.index:
            borda.loc[sub.index, c] = sub[c].rank(method='average')
        df.loc[sub.index, 'P2Borda分'] = ((borda.loc[sub.index] * w).sum(axis=1) / n).round(4)
    df['P3概率'] = np.where(df['今日涨停'],
                            np.minimum(cfg['base_limit_up'] + df['P1加权分'] * 0.35, 0.62),
                            np.minimum(cfg['base_rush'] + df['P1加权分'] * 0.16, 0.25))
    df['M可买性'] = df.apply(lambda r: m_buyability(r, th), axis=1)
    df['R风险系数'] = df.apply(lambda r: risk_penalty(r, th), axis=1)
    for c in ['P1加权分', 'P2Borda分', 'P3概率']:
        rng = df[c].max() - df[c].min()
        df[c + '_z'] = (df[c] - df[c].min()) / (rng + 1e-9)
    df['综合分'] = ((df['P1加权分_z'] + df['P2Borda分_z'] + df['P3概率_z']) / 3
                    * df['M可买性'] * df['R风险系数']).round(3)
    df['预估涨停概率%'] = (df['P3概率'] * df['R风险系数'] * 100).round(1)

    res = df.sort_values('综合分', ascending=False).reset_index(drop=True)
    res.insert(0, '排名', res.index + 1)

    # ---- 清单1：明日涨停 TopN ----
    top_limit = res.head(cfg['top_n'])

    # ---- 清单2：明日涨幅≥4% ----
    df['涨4评分'] = df.apply(lambda r: rise4_score(r, th), axis=1).round(3)
    df['预估涨4概率%'] = (np.minimum(
        cfg['base_rise4'] + df['涨4评分'] * 0.35, 0.55) * df['R风险系数'] * 100).round(1)
    rise4 = (df.sort_values(['涨4评分', '预估涨4概率%'], ascending=False)
             .head(cfg['rise4_n']).reset_index(drop=True))
    rise4.insert(0, '排名', rise4.index + 1)

    # ---- 清单3：跨日连板候选（今日涨停且封单强，按 封单+质量+锁仓 排序） ----
    cand = res[res['今日涨停'] & (res['S1封单强度'] >= 0.6)].copy()
    note = ''
    if len(cand) == 0:
        cand = res[res['今日涨停']].copy()
        note = '今日无强封单股，以下为涨停股中相对最优（整体偏弱，注意仓位）'
    cand['连板潜力'] = (cand['S1封单强度'] * 0.5 + cand['S2封板质量'] * 0.3
                        + cand['S3锁仓度'] * 0.2).round(3)
    monday = (cand.sort_values('连板潜力', ascending=False)
              .head(cfg['monday_n']).reset_index(drop=True))

    return {'涨停TopN': top_limit, '涨幅4%': rise4, '连板候选': monday,
            '全量': res, '连板备注': note}


def cross_validate(df):
    """三聚合路径 Spearman 秩相关（涨停组内）"""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    lb = df[df['今日涨停']]
    if len(lb) < 5:
        return None
    out = {}
    for a, b in [('P1加权分', 'P2Borda分'), ('P1加权分', 'P3概率'), ('P2Borda分', 'P3概率')]:
        rho, p = spearmanr(lb[a], lb[b])
        out[f'{a}×{b}'] = (round(rho, 3), p)
    return out