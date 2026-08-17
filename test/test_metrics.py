# -*- coding: utf-8 -*-
"""P0-2/P0-3/P0-1/P0-6 单元测试：core.metrics 纯函数库。

覆盖：
  - per_trade_cost_pct         双边成本（按板块选滑点：主板 0.35% / 20cm 0.55%）
  - net_return_curve          净值 / Sharpe / 最大回撤 / 胜率
  - benchmark_curve           基准净值
  - factor_significance       Bonferroni 多重比较校正
  - factor_correlation        五因子相关矩阵（5×5）
  - incremental_ic            正交后增量 IC（5 行）
  - rolling_net_curve         跨日聚合（样本不足 vs 充足）
  - backtest_one 四元组 + 绩效键（P0-2 集成）
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from core import backtest as bt
from core import metrics
from core import strategies
from core.db import DB

BCFG = {'commission': 0.00025, 'stamp_tax': 0.001,
        'slippage_main': 0.001, 'slippage_20cm': 0.002}


# ---------------------------------------------------------------- 成本
def test_per_trade_cost_pct_by_board():
    # 主板 = 2*0.00025 + 0.001 + 2*0.001 = 0.0035
    assert abs(metrics.per_trade_cost_pct('600000', BCFG) - 0.0035) < 1e-9
    assert abs(metrics.per_trade_cost_pct('000001', BCFG) - 0.0035) < 1e-9
    # 创业板 / 科创板 = 2*0.00025 + 0.001 + 2*0.002 = 0.0055
    assert abs(metrics.per_trade_cost_pct('300001', BCFG) - 0.0055) < 1e-9
    assert abs(metrics.per_trade_cost_pct('688001', BCFG) - 0.0055) < 1e-9
    # 北交所按主板滑点计（仍 0.0035）
    assert abs(metrics.per_trade_cost_pct('920002', BCFG) - 0.0035) < 1e-9


# ---------------------------------------------------------------- 净值曲线
def test_net_return_curve_single_period():
    gross = pd.Series([0.05])
    codes = pd.Series(['600000'])
    r = metrics.net_return_curve(gross, codes, BCFG)
    assert r['net_return'] == pytest.approx(0.05 - 0.0035)
    assert r['win_rate'] == 1.0
    assert np.isnan(r['sharpe'])          # 单期无法算 Sharpe
    assert r['max_dd'] == 0.0


def test_net_return_curve_multi_period():
    gross = pd.Series([0.1, -0.05, 0.02])
    codes = pd.Series(['600000', '600000', '600000'])
    r = metrics.net_return_curve(gross, codes, BCFG)
    costs = [metrics.per_trade_cost_pct('600000', BCFG)] * 3
    net = gross.values - costs
    exp_ret = float((1 + pd.Series(net)).prod() - 1)
    assert r['net_return'] == pytest.approx(exp_ret)
    assert r['win_rate'] == pytest.approx(2 / 3)
    assert pd.notna(r['sharpe'])
    assert r['max_dd'] <= 0


def test_benchmark_curve_empty():
    empty = metrics.benchmark_curve(None, BCFG)
    assert empty['net_return'] == 0.0


# ---------------------------------------------------------------- 显著性 Bonferroni
def test_factor_significance_bonferroni():
    # 确定性数据：每列 30 点均值 0.6、标准差 0.1 → t≈13.7，p≪0.01 → 全部显著（不依赖随机）
    col = np.tile([0.5, 0.7], 15)
    df = pd.DataFrame({f'F{i}': col for i in range(5)})
    sig = metrics.factor_significance(df, alpha=0.05)
    assert list(sig['策略']) == [f'F{i}' for i in range(5)]
    assert abs(sig['Bonferroniα'].iloc[0] - 0.01) < 1e-9     # 0.05 / 5
    assert np.allclose(sig['均值IC'].values, df.mean().values, atol=1e-6)
    assert sig['是否显著'].all()                              # 均值0.6 远大于噪声 → 显著


# ---------------------------------------------------------------- 合成回测库
def _make_backtest_db(base_dir, n_stocks=15, n_days=3, seed=42):
    """构造最小回测库：n_days 天每日 n_stocks 只，predictions 含 S1~S5 因子。"""
    rng = np.random.default_rng(seed)
    db = DB(base_dir)
    days = [f'2026010{i}' for i in range(5, 5 + n_days)]
    codes = [f'{c:06d}' for c in range(600000, 600000 + n_stocks)]
    for d in days:
        rows = []
        for i, code in enumerate(codes):
            zf = float(rng.uniform(-5, 10))
            og = float(rng.uniform(-3, 6))
            rows.append({'代码6': code, '名称': f'股{i:02d}', '涨幅': zf,
                         '今日涨停': zf >= 9.5, '开盘涨幅': og,
                         '封板分钟': 5.0, 'M可买性': 0.8})
        df = pd.DataFrame(rows)
        info = {'hash': f'{d}-h', 'path': f'{d}.xlsx', 'date': d, 'kind': 'close'}
        db.save_snapshot(info, df)
    cfg = strategies.load_config('/nonexistent')
    for d in days:
        recs = []
        for i, code in enumerate(codes):
            recs.append({'代码6': code, '名称': f'股{i:02d}',
                         '预估涨停概率%': float(rng.uniform(0, 1)),
                         '综合分': float(rng.uniform(0, 100)),
                         'S1封单强度': float(rng.normal(0, 1)),
                         'S2封板质量': float(rng.normal(0, 1)),
                         'S3锁仓度': float(rng.normal(0, 1)),
                         'S4资金': float(rng.normal(0, 1)),
                         'S5股性结构': float(rng.normal(0, 1))})
        pdf = pd.DataFrame(recs)
        db.save_predictions(d, d, '涨停TopN', pdf, '预估涨停概率%', '综合分')
    return db, cfg, days


def test_backtest_one_returns_4tuple_and_perf():
    tmp = tempfile.mkdtemp(prefix='metrics_test_')
    try:
        db, cfg, days = _make_backtest_db(tmp, n_days=3)
        mg, summary, ic, perf = bt.backtest_one(db, days[0], '涨停TopN', cfg)
        assert summary is not None
        assert {'net_return', 'bench_return', 'cost_pct', 'cant_trade_ratio',
                'caliber', 'n', 'n_tradable'} <= set(perf)
        assert isinstance(perf['net_return'], float)
        assert '净收益%' in mg.columns and '无法成交' in mg.columns
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_factor_correlation_shape():
    tmp = tempfile.mkdtemp(prefix='metrics_test_')
    try:
        db, cfg, _ = _make_backtest_db(tmp, n_days=3)
        corr = metrics.factor_correlation(db, cfg, method='spearman')
        assert corr.shape == (5, 5)
        assert list(corr.columns) == bt.V2_KEYS
        assert np.allclose(np.diag(corr.values), 1.0, atol=1e-6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_incremental_ic_shape():
    tmp = tempfile.mkdtemp(prefix='metrics_test_')
    try:
        db, cfg, _ = _make_backtest_db(tmp, n_days=3)
        inc = metrics.incremental_ic(db, '涨停TopN', cfg)
        assert len(inc) == 5
        for col in ['因子', 'IC(原始)', '增量IC', '增量IC_p', '独立增量贡献%']:
            assert col in inc.columns
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rolling_net_curve_sufficient_and_insufficient():
    # 充足：3 个交易日
    tmp = tempfile.mkdtemp(prefix='metrics_test_')
    try:
        db, cfg, _ = _make_backtest_db(tmp, n_days=3)
        r = metrics.rolling_net_curve(db, '涨停TopN', cfg, caliber='close')
        assert r['sufficient'] is True
        assert r['n_days'] == 3
        assert 'curve' in r['portfolio']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 不足：单交易日 → 样本不足
    tmp = tempfile.mkdtemp(prefix='metrics_test_')
    try:
        db, cfg, _ = _make_backtest_db(tmp, n_days=1)
        r = metrics.rolling_net_curve(db, '涨停TopN', cfg, caliber='close')
        assert r['sufficient'] is False
        assert r['n_days'] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
