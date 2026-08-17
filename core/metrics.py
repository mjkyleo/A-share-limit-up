# -*- coding: utf-8 -*-
"""
绩效与因子分析纯函数库（core.metrics）
======================================
被 backtest.py 调用，GUI 仅展示。所有函数无副作用、可单测。

集中实现（P0-2 / P0-3 / P0-1 / P0-5 / P0-6）：
  - per_trade_cost_pct      双边交易成本（按板块选滑点）
  - net_return_curve        等权组合净值 / Sharpe / 最大回撤 / 胜率
  - benchmark_curve         "买全部涨停股"基准等权净值
  - rolling_net_curve       跨交易日滚动聚合（净值曲线 / Sharpe / 回撤 / 日胜率）
  - daily_ic_series         每日 × 五策略 的 Spearman IC 矩阵
  - factor_significance     单样本 t 检验 + Bonferroni 多重比较校正
  - factor_correlation      五因子 Spearman 相关矩阵（跨日截面均值）
  - incremental_ic          正交后独立增量 IC（多元回归残差法）
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from core import strategies


# ----------------------------- 交易成本 -----------------------------
def per_trade_cost_pct(code, bcfg: dict) -> float:
    """双边交易成本（小数）。公式 = 2*commission + stamp_tax + 2*slippage。

    滑点按板块选择：创业板/科创板(20cm) 用 slippage_20cm，其余用 slippage_main。
    主板精确合计 = 2*0.00025 + 0.001 + 2*0.001 = 0.0035
    20cm 精确合计 = 2*0.00025 + 0.001 + 2*0.002 = 0.0055
    """
    commission = float(bcfg.get('commission', 0.00025))
    stamp_tax = float(bcfg.get('stamp_tax', 0.001))
    slip_main = float(bcfg.get('slippage_main', 0.001))
    slip_20 = float(bcfg.get('slippage_20cm', 0.002))
    board = strategies.board_of(code)
    slip = slip_20 if board in ('cyb', 'star') else slip_main
    return 2.0 * commission + stamp_tax + 2.0 * slip


# ----------------------------- 净值 / 绩效 -----------------------------
def net_return_curve(gross_ret, codes, bcfg: dict, init_capital: float = 1_000_000) -> dict:
    """等权组合净值（gross_ret 扣双边成本后）。

    参数：
      gross_ret  pd.Series / 序列：每期的等权组合毛收益（小数，单期或多期时间序列）
      codes      pd.Series / 序列：与 gross_ret 对齐的股票代码（用于选滑点计成本）
      bcfg       backtest_cost 配置
    返回：{curve, sharpe, max_dd, win_rate, net_return, mean_daily}
      curve      净值序列（index=['start',0,1,...]）
      sharpe     年化夏普（√252）；样本<2 期为 NaN
      max_dd     最大回撤（小数，≤0）
      win_rate   盈利期占比
      net_return 累计净收益（小数）
      mean_daily 期均净收益（小数）
    """
    gross = pd.Series(gross_ret, dtype='float64').reset_index(drop=True)
    code_s = pd.Series(list(codes)).reset_index(drop=True)
    if len(gross) == 0:
        return {'curve': pd.Series([init_capital], name='净值'),
                'sharpe': float('nan'), 'max_dd': 0.0,
                'win_rate': 0.0, 'net_return': 0.0, 'mean_daily': 0.0}
    cost = code_s.apply(lambda c: per_trade_cost_pct(c, bcfg))
    net = gross - cost
    curve = pd.Series([init_capital] + list(init_capital * (1.0 + net).cumprod()))
    curve.index = ['start'] + list(range(len(net)))
    net_return = float((1.0 + net).prod() - 1.0)
    mean_daily = float(net.mean())
    win_rate = float((net > 0).mean())
    if len(net) >= 2:
        sd = net.std(ddof=1)
        sharpe = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
        wealth = curve.values.astype(float)
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / np.where(peak == 0, 1, peak)
        max_dd = float(dd.min())
    else:
        sharpe = float('nan')
        max_dd = 0.0
    return {'curve': curve, 'sharpe': sharpe, 'max_dd': max_dd,
            'win_rate': win_rate, 'net_return': net_return, 'mean_daily': mean_daily}


def benchmark_curve(actual: pd.DataFrame, bcfg: dict) -> dict:
    """'买全部涨停股'(actual['今日涨停']==True) 等权净值，结构与 net_return_curve 一致。"""
    empty = {'curve': pd.Series([1_000_000], name='净值'), 'sharpe': float('nan'),
             'max_dd': 0.0, 'win_rate': 0.0, 'net_return': 0.0, 'mean_daily': 0.0}
    if actual is None or actual.empty or '今日涨停' not in actual:
        return empty
    sub = actual[actual['今日涨停'] == True].copy()  # noqa: E712
    if sub.empty:
        return empty
    if '口径收益%' in sub.columns:
        gross = pd.Series(sub['口径收益%'].values, dtype='float64')
    elif '涨幅' in sub.columns:
        gross = pd.Series(sub['涨幅'].values, dtype='float64') / 100.0
    else:
        return empty
    codes = sub['代码6'] if '代码6' in sub.columns else sub.get('code')
    return net_return_curve(gross, codes, bcfg)


def rolling_net_curve(db, list_type: str = '涨停TopN', cfg: dict = None,
                      caliber: str = None) -> dict:
    """跨交易日滚动聚合：把每个可回测日的组合净收益排成时间序列，算净值/Sharpe/回撤/日胜率。

    仅 ≥2 个交易日可用时给出曲线与 Sharpe/回撤；否则标注"样本不足"。
    返回：{sufficient, n_days, caliber, portfolio, benchmark, note}
    """
    from core import backtest as bt
    cfg = cfg or {}
    bcfg = cfg.get('backtest_cost', {})
    cal = caliber or bcfg.get('caliber', 'close')
    days = db.pending_backtest_dates()
    gross_port, code_port, gross_bench, code_bench = {}, {}, {}, {}
    for d in days:
        _, _, _, perf = bt.backtest_one(db, d, list_type, cfg, caliber=cal)
        if not perf:
            continue
        gross_port[d] = perf['gross_return']
        code_port[d] = perf['rep_code']
        gross_bench[d] = perf['bench_gross']
        code_bench[d] = perf['bench_code']
    n_days = len(gross_port)
    if n_days == 0:
        return {'sufficient': False, 'n_days': 0, 'caliber': cal,
                'portfolio': _empty_perf(), 'benchmark': _empty_perf(),
                'note': '样本不足（无可回测日）'}
    port = net_return_curve(pd.Series(gross_port), pd.Series(code_port), bcfg)
    bench = net_return_curve(pd.Series(gross_bench), pd.Series(code_bench), bcfg)
    sufficient = n_days >= 2
    note = (f'跨 {n_days} 个交易日等权净值（{cal} 口径，初始100万仅刻度）'
            if sufficient else '样本不足（需 ≥2 个交易日）无法计算 Sharpe/最大回撤')
    return {'sufficient': sufficient, 'n_days': n_days, 'caliber': cal,
            'portfolio': port, 'benchmark': bench, 'note': note}


def _empty_perf() -> dict:
    return {'curve': pd.Series([1_000_000], name='净值'), 'sharpe': float('nan'),
            'max_dd': 0.0, 'win_rate': 0.0, 'net_return': 0.0, 'mean_daily': 0.0}


# ----------------------------- 相关性 / 增量 IC（因子共线 P0-6） -----------------------------
def _spearman(a, b):
    try:
        rho, p = stats.spearmanr(a, b)
        return float(rho), float(p)
    except Exception:
        return None, None


def daily_ic_series(db, list_type: str = '涨停TopN', cfg: dict = None) -> pd.DataFrame:
    """date × 五策略 的每日 Spearman IC（来自 backtest_one 的 ic dict）。"""
    from core import backtest as bt
    recs = []
    for d in db.pending_backtest_dates():
        _, _, ic, _ = bt.backtest_one(db, d, list_type, cfg)
        if not ic:
            continue
        row = {k: v['IC(ρ)'] for k, v in ic.items()}
        row['date'] = d
        recs.append(row)
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs).set_index('date')


def factor_significance(daily_ic: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """对每列做单样本 t 检验(H0: 均值IC=0) + Bonferroni 多重比较校正。

    返回：策略|均值IC|CI95低|CI95高|t|p|Bonferroniα|是否显著|样本天数n
    Bonferroni α_b = alpha / 列数。
    """
    if daily_ic is None or daily_ic.empty:
        return pd.DataFrame(columns=['策略', '均值IC', 'CI95低', 'CI95高', 't',
                                     'p', 'Bonferroniα', '是否显著', '样本天数n'])
    cols = list(daily_ic.columns)
    k = len(cols)
    alpha_b = alpha / k if k > 0 else alpha
    rows = []
    for c in cols:
        v = pd.to_numeric(daily_ic[c], errors='coerce').dropna()
        n = len(v)
        if n < 2:
            rows.append({'策略': c, '均值IC': float('nan'), 'CI95低': float('nan'),
                         'CI95高': float('nan'), 't': float('nan'), 'p': float('nan'),
                         'Bonferroniα': round(alpha_b, 4), '是否显著': False, '样本天数n': n})
            continue
        mean = float(v.mean())
        sd = v.std(ddof=1)
        sem = sd / np.sqrt(n)
        t = mean / sem if sem > 0 else 0.0
        p = 2.0 * (1.0 - stats.t.cdf(abs(t), n - 1))
        tcrit = stats.t.ppf(0.975, n - 1)
        ci_low = mean - tcrit * sem
        ci_high = mean + tcrit * sem
        sig = bool(p < alpha_b)
        rows.append({'策略': c, '均值IC': round(mean, 3), 'CI95低': round(ci_low, 3),
                     'CI95高': round(ci_high, 3), 't': round(t, 3), 'p': round(p, 4),
                     'Bonferroniα': round(alpha_b, 4), '是否显著': sig, '样本天数n': n})
    return pd.DataFrame(rows)


def factor_correlation(db, cfg: dict = None, method: str = 'spearman') -> pd.DataFrame:
    """五因子 Spearman 相关矩阵（每日截面 corr 后跨日取均值）。

    返回 5×5 DataFrame（index=columns=五策略键）。
    """
    from core import backtest as bt
    factors = bt.V2_KEYS
    days = db.pending_backtest_dates()
    mats = []
    for d in days:
        mg, _, _, _ = bt.backtest_one(db, d, '涨停TopN', cfg)
        if mg is None:
            continue
        det = mg['detail'].apply(lambda s: json.loads(s) if s else {})
        row = {f: det.apply(lambda x: x.get(f)) for f in factors}
        df = pd.DataFrame(row).apply(pd.to_numeric, errors='coerce')
        if df.shape[0] >= 5:
            mats.append(df.corr(method=method))
    if not mats:
        return pd.DataFrame(index=factors, columns=factors, dtype='float64')
    out = sum(mats) / len(mats)
    return out.round(3)


def incremental_ic(db, list_type: str = '涨停TopN', cfg: dict = None) -> pd.DataFrame:
    """正交后独立增量 IC（因子共线 P0-6）。

    每日截面：以 实际涨幅 为因、其余 4 因子为自做多元回归得残差，
    再算 该因子 vs 残差 的 Spearman IC；跨日取均值。
    返回：因子|IC(原始)|增量IC|增量IC_p|独立增量贡献%
    """
    from core import backtest as bt
    factors = bt.V2_KEYS
    raw, inc, inc_p = ({f: [] for f in factors} for _ in range(3))
    for d in db.pending_backtest_dates():
        mg, _, _, _ = bt.backtest_one(db, d, list_type, cfg)
        if mg is None:
            continue
        det = mg['detail'].apply(lambda s: json.loads(s) if s else {})
        X = pd.DataFrame({f: pd.to_numeric(det.apply(lambda x: x.get(f)), errors='coerce')
                          for f in factors})
        y = pd.to_numeric(mg['实际涨幅%'], errors='coerce')
        mask = X.notna().all(axis=1) & y.notna()
        if mask.sum() < 10:
            continue
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        for f in factors:
            rho, _ = _spearman(X[f], y)
            raw[f].append(rho if rho is not None else float('nan'))
        for f in factors:
            others = [g for g in factors if g != f]
            Xo = np.column_stack([np.ones(len(X)), X[others].values.astype(float)])
            yo = y.values.astype(float)
            beta, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
            resid = yo - Xo @ beta
            rho, p = _spearman(X[f].values.astype(float), resid)
            inc[f].append(rho if rho is not None else float('nan'))
            inc_p[f].append(p if p is not None else float('nan'))
    tot = sum(abs(np.nanmean(inc[f]) if inc[f] else 0.0) for f in factors) or 1.0
    rows = []
    for f in factors:
        m_raw = float(np.nanmean(raw[f])) if raw[f] else float('nan')
        m_inc = float(np.nanmean(inc[f])) if inc[f] else float('nan')
        m_p = float(np.nanmean(inc_p[f])) if inc_p[f] else float('nan')
        contrib = abs(m_inc) / tot * 100.0 if (tot > 0 and not np.isnan(m_inc)) else 0.0
        rows.append({'因子': f, 'IC(原始)': round(m_raw, 3), '增量IC': round(m_inc, 3),
                     '增量IC_p': round(m_p, 4), '独立增量贡献%': round(contrib, 1)})
    return pd.DataFrame(rows)
