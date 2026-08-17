# -*- coding: utf-8 -*-
"""自上而下选股方法论引擎的离线回归测试（不依赖网络/上证指数）。"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from core.topdown import (timing_from_breadth, sector_strength, main_lines,
                          leader_score, topdown_rank, market_timing)


def _df():
    """构造一个可控的小池子：两个板块，主线板块动量明显更强。"""
    def mk(code, name, ind, zf, z20, mf, mc, seal,涨停):
        return {'代码6': code, '名称': name, '所属行业': ind,
                '涨幅': zf, '5日涨幅': zf, '10日涨幅': z20 / 2.0, '20日涨幅': z20,
                '主力净量': mf, '流通市值': mc, '现价': 10 if 涨停 else 5,
                '市盈': 30, '市净率': 2, '封流比': seal, '封成比': 1.0,
                '今日涨停': 涨停}
    rows = []
    # 主线板块：光学光电子（强动量、有涨停）
    for i, (zf, z20) in enumerate([(8.0, 20.0), (6.0, 18.0), (4.0, 15.0), (2.0, 12.0), (1.0, 10.0)]):
        rows.append(mk(f'10000{i}', f'光{i}', '光学光电子', zf, z20, 1.0, 5e10, 2.0, (zf >= 9.5)))
    # 非主线板块：煤炭（弱动量、无涨停）
    for i, (zf, z20) in enumerate([(0.2, 1.0), (-0.5, 0.5), (0.1, 2.0)]):
        rows.append(mk(f'20000{i}', f'煤{i}', '煤炭', zf, z20, -0.5, 2e10, np.nan, False))
    return pd.DataFrame(rows)


def test_timing_breadth_keys():
    t = timing_from_breadth(_df())
    for k in ('method', 'regime', 'score', 'coeff', 'detail'):
        assert k in t, f"择时缺字段 {k}"
    assert t['method'] == 'breadth'
    assert 0 <= t['coeff'] <= 1


def test_sector_strength_and_main_lines():
    u = _df()
    s = sector_strength(u, min_count=2)
    assert not s.empty
    assert set(['所属行业', '强度分', '板块排名']).issubset(s.columns)
    # 光学光电子板块更强 → 排名第一
    assert s.iloc[0]['所属行业'] == '光学光电子'
    mset, mstr, mrank = main_lines(s, top_n=1)
    assert '光学光电子' in mset


def test_topdown_rank_shape_and_gating():
    u = _df()
    timing, sectors, ranked = topdown_rank(u, ak=None, force_method='breadth',
                                           top_n_sectors=1, min_sector_count=2)
    assert not ranked.empty
    assert '综合得分' in ranked.columns and '择时系数' in ranked.columns
    assert '是否在主线' in ranked.columns
    # 所有行综合得分>0
    assert (ranked['综合得分'] > 0).all()
    # 主线板块（光学）个股综合得分应整体高于非主线（煤炭）
    main = ranked[ranked['是否在主线']]
    nonmain = ranked[~ranked['是否在主线']]
    assert main['综合得分'].mean() > nonmain['综合得分'].mean()


def test_market_timing_force_breadth_ignores_index():
    u = _df()
    t = market_timing(u, ak=None, force_method='breadth')
    assert t['method'] == 'breadth'


def test_extension_run_returns_main_table():
    from extensions.topdown_pick import TopdownPickExtension
    from extensions.base import normalize_market_cols
    ext = TopdownPickExtension()
    df = normalize_market_cols(_df())
    cfg = ext.effective_params({'params': {}}, {})
    ctx = types.SimpleNamespace(db=None, base_dir='.', data_dir='.')
    out = ext.run(df, cfg, ctx)
    assert '主表' in out and 'note' in out and 'tables' in out
    main = out['主表']
    assert not main.empty
    assert ext.score_col in main.columns          # '综合得分'
    assert '代码6' in main.columns and '名称' in main.columns
    # 板块强度子表存在
    assert '板块强度' in out['tables']
