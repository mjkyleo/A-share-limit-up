# -*- coding: utf-8 -*-
"""短线反转策略（Short-term Reversal）
===============================
与涨停接力动量策略正交：今日大跌/急挫但盘中出现承接，且不在长期下跌通道，
博次日情绪修复带来的反弹。

适用场景：risk_off / panic 市场，或个股突发错杀。

数据源：
  - 推荐 akshare（spot 实时行情）
  - 也兼容本地 data/ 文件（只要有 涨幅/最高/最低/现价/换手/量比/60日涨幅 等）
"""
import numpy as np
import pandas as pd

from extensions.base import Extension, Param


class ShortReversalExtension(Extension):
    key = 'short_reversal'
    name = '短线反转'
    description = '今日大跌但有长下影/放量承接，不在长期下跌通道，博次日情绪修复'
    datasource = 'akshare'
    list_type = '短线反转'
    prob_col = '预估反弹概率%'
    score_col = '反转评分'
    # 目标：次日反弹≥2% 算达成；继续跌扣分，涨得多得分高
    scoring = [[-99, -80], [-6, -50], [-4, -30], [-2, -15], [0, 20],
               [2, 60], [4, 80], [6, 95], [9.5, 100]]
    hit_threshold = 2.0
    detail_keys = ['反转评分', '预估反弹概率%', '超跌分', '承接分', '量能分',
                   '趋势闸门分', '质量分', '下影线比']

    params = [
        Param('min_drop', '最小跌幅%(幅度)', 'float', 3.0, 0.5, 10, step=0.5,
              help='今日跌幅至少这么大（如 3 = 跌≥3%），与 max_drop 共同构成跌幅区间'),
        Param('max_drop', '最大跌幅%(幅度)', 'float', 7.0, 1, 20, step=0.5,
              help='跌幅超过此值视为恐慌/跌停区，默认排除'),
        Param('min_amp', '最小振幅%', 'float', 3.0, 0.5, 15, step=0.5),
        Param('min_shadow_ratio', '最小下影线比', 'float', 0.25, 0, 1, step=0.05,
              help='(现价-最低)/(最高-最低)，越大代表盘中承接越强'),
        Param('min_turnover', '最低换手%', 'float', 1.0, 0, 10, step=0.2),
        Param('max_turnover', '最高换手%', 'float', 20.0, 5, 50, step=1),
        Param('min_lb', '最小量比', 'float', 1.0, 0.5, 5, step=0.1),
        Param('min_60d_change', '60日涨幅下限%', 'float', -20.0, -60, 100, step=1,
              help='过滤长期下跌趋势，-20 表示近 60 日跌幅不超过 20%'),
        Param('max_cap', '最大流通市值(亿)', 'float', 300.0, 50, 2000, step=10),
        Param('exclude_st', '排除 ST/亏损股', 'bool', True),
        Param('top_n', '输出数量', 'int', 20, 5, 100, step=5),
    ]

    def run(self, df, cfg, ctx):
        from extensions.base import normalize_market_cols
        d = normalize_market_cols(df)

        # 跌幅幅度：cfg 里填正数，实际跌幅 = -涨幅
        drop = -d['涨幅']
        mask = pd.Series(True, index=d.index)
        mask &= drop.between(cfg['min_drop'], cfg['max_drop'])
        mask &= d['振幅'] >= cfg['min_amp']
        mask &= d['换手'].between(cfg['min_turnover'], cfg['max_turnover'])
        mask &= d['量比'] >= cfg['min_lb']
        mask &= d['60日涨幅'] >= cfg['min_60d_change']
        mask &= (d['流通市值'] / 1e8) <= cfg['max_cap']
        if cfg.get('exclude_st', True):
            name_ok = ~d['名称'].astype(str).str.contains(r'ST|退市', regex=True, na=False)
            pe_ok = d.get('PE_TTM', pd.Series(np.nan, index=d.index)).fillna(-1) > 0
            mask &= (name_ok & pe_ok)
        cand = d[mask].copy()

        if cand.empty:
            return {'主表': pd.DataFrame(),
                    'note': f"无符合短线反转条件的股票（跌幅 {cfg['min_drop']}%~{cfg['max_drop']}%，"
                            f"下影线比≥{cfg['min_shadow_ratio']}）。注意：参数填跌幅幅度（正数）", 'tables': {}}

        # 计算下影线比
        rng = cand['最高'] - cand['最低']
        cand['下影线比'] = np.where(rng > 0, (cand['现价'] - cand['最低']) / rng, 0)
        cand = cand[cand['下影线比'] >= cfg['min_shadow_ratio']]
        if cand.empty:
            return {'主表': pd.DataFrame(), 'note': '过滤后无有效下影线承接股票', 'tables': {}}

        # 因子打分（均 0~1）
        # 1. 超跌分：跌得越深（但不超过 max_drop）分越高。cfg 里填跌幅幅度（正数）
        zf = cand['涨幅'].abs()
        cand['超跌分'] = np.clip((zf - cfg['min_drop']) /
                                 (cfg['max_drop'] - cfg['min_drop'] + 1e-6), 0, 1)
        # 2. 承接分
        cand['承接分'] = np.clip(cand['下影线比'], 0, 1)
        # 3. 量能分：量比适中放大 + 换手在区间内
        lb = cand['量比']
        cand['量能分'] = (np.clip((lb - 1) / 2, 0, 1) * 0.6 +
                         np.clip(1 - (cand['换手'] - 5).abs() / 10, 0, 1) * 0.4)
        # 4. 趋势闸门：60日跌幅越小越好
        g60 = cand['60日涨幅']
        cand['趋势闸门分'] = np.clip((g60 - cfg['min_60d_change']) / 20, 0, 1)
        # 5. 质量分：小市值 + 非亏损
        cap = cand['流通市值'] / 1e8
        cand['质量分'] = np.clip(1 - cap / cfg['max_cap'], 0, 1)

        # 综合评分（等权合成）
        cand['反转评分'] = (cand['超跌分'] * 0.25 + cand['承接分'] * 0.30 +
                          cand['量能分'] * 0.20 + cand['趋势闸门分'] * 0.15 +
                          cand['质量分'] * 0.10)
        cand['预估反弹概率%'] = np.clip(cand['反转评分'] * 100, 5, 95)

        cand = cand.sort_values('反转评分', ascending=False).head(cfg['top_n'])
        cand.insert(0, '排名', range(1, len(cand) + 1))

        cols = ['排名', '代码6', '名称', '今日涨停', '涨幅', '60日涨幅', '振幅',
                '下影线比', '换手', '量比', '反转评分', '预估反弹概率%',
                '超跌分', '承接分', '量能分', '趋势闸门分', '质量分']
        cols = [c for c in cols if c in cand.columns]
        main = cand[cols].reset_index(drop=True)

        note = (f"共 {len(main)} 只短线反转候选 | 平均评分 {main['反转评分'].mean():.2f} | "
                f"跌幅区间 {cfg['min_drop']}% ~ {cfg['max_drop']}%")
        return {'主表': main, 'note': note, 'tables': {}}
