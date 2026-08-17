# -*- coding: utf-8 -*-
"""防御/低波策略（Defensive / Low-Volatility）
=============================================
当市场处于 risk_off / panic 状态时，作为底仓压舱石：
  - 低波动（振幅小、换手低）
  - 低估值 / 盈利稳定（PE/PB 合理）
  - 走势独立于大盘（60日涨幅不过热也不过冷）
  - 大市值或红利风格优先

适用场景：大盘持续走弱、主线缺失、避险情绪重。
"""
import numpy as np
import pandas as pd

from extensions.base import Extension, Param


class DefensiveExtension(Extension):
    key = 'defensive'
    name = '防御低波'
    description = '低波动、低估值、盈利稳定、走势独立，作为 risk_off / panic 时的底仓'
    datasource = 'akshare'
    list_type = '防御低波'
    prob_col = '稳健概率%'
    score_col = '防御评分'
    # 目标：次日不亏（≥0%）算基本达成；小幅上涨即高分
    scoring = [[-99, -100], [-6, -60], [-4, -40], [-2, -10], [0, 50],
               [2, 80], [4, 100], [6, 100], [9.5, 100]]
    hit_threshold = 0.0
    detail_keys = ['防御评分', '稳健概率%', '低波分', '估值分', '盈利分',
                   '趋势稳度分', '市值分']

    params = [
        Param('max_amp', '最大振幅%', 'float', 4.0, 1, 10, step=0.2),
        Param('max_turnover', '最大换手%', 'float', 5.0, 1, 15, step=0.2),
        Param('min_60d_change', '60日涨幅下限%', 'float', -5.0, -20, 10, step=0.5),
        Param('max_60d_change', '60日涨幅上限%', 'float', 15.0, 0, 50, step=1),
        Param('max_pb', '最大市净率PB', 'float', 3.0, 1, 10, step=0.1),
        Param('min_pe', '最小市盈率PE_TTM', 'float', 0, -100, 50, step=1),
        Param('max_pe', '最大市盈率PE_TTM', 'float', 30, 5, 100, step=1),
        Param('min_cap', '最小流通市值(亿)', 'float', 100.0, 30, 1000, step=10),
        Param('max_cap', '最大流通市值(亿)', 'float', 2000.0, 100, 10000, step=50),
        Param('exclude_st', '排除 ST/亏损股', 'bool', True),
        Param('top_n', '输出数量', 'int', 20, 5, 100, step=5),
    ]

    def run(self, df, cfg, ctx):
        from extensions.base import normalize_market_cols
        d = normalize_market_cols(df)

        mask = pd.Series(True, index=d.index)
        mask &= d['振幅'] <= cfg['max_amp']
        mask &= d['换手'] <= cfg['max_turnover']
        mask &= d['60日涨幅'].between(cfg['min_60d_change'], cfg['max_60d_change'])
        mask &= d['PB'] <= cfg['max_pb']
        mask &= d['PE_TTM'].between(cfg['min_pe'], cfg['max_pe'])
        mask &= (d['流通市值'] / 1e8).between(cfg['min_cap'], cfg['max_cap'])
        if cfg.get('exclude_st', True):
            mask &= ~d['名称'].astype(str).str.contains(r'ST|退市', regex=True, na=False)
        cand = d[mask].copy()

        if cand.empty:
            return {'主表': pd.DataFrame(),
                    'note': '无符合防御条件的股票（低波+低估值+盈利+中大市值）',
                    'tables': {}}

        # 因子打分
        amp = cand['振幅']
        cand['低波分'] = np.clip(1 - amp / cfg['max_amp'], 0, 1)

        hs = cand['换手']
        cand['低波分'] += np.clip(1 - hs / cfg['max_turnover'], 0, 1) * 0.5
        cand['低波分'] = np.clip(cand['低波分'] / 1.5, 0, 1)

        pb = cand['PB']
        cand['估值分'] = np.clip(1 - pb / cfg['max_pb'], 0, 1)

        pe = cand['PE_TTM']
        pe_mid = (cfg['min_pe'] + cfg['max_pe']) / 2
        cand['盈利分'] = np.clip(1 - (pe - pe_mid).abs() / (cfg['max_pe'] - pe_mid + 1e-6), 0, 1)

        g60 = cand['60日涨幅']
        mid60 = (cfg['min_60d_change'] + cfg['max_60d_change']) / 2
        cand['趋势稳度分'] = np.clip(1 - (g60 - mid60).abs() /
                                  (cfg['max_60d_change'] - mid60 + 1e-6), 0, 1)

        cap = cand['流通市值'] / 1e8
        cand['市值分'] = np.clip((cap - cfg['min_cap']) /
                                (cfg['max_cap'] - cfg['min_cap'] + 1e-6), 0, 1)

        cand['防御评分'] = (cand['低波分'] * 0.30 + cand['估值分'] * 0.25 +
                          cand['盈利分'] * 0.15 + cand['趋势稳度分'] * 0.15 +
                          cand['市值分'] * 0.15)
        cand['稳健概率%'] = np.clip(cand['防御评分'] * 100, 5, 95)

        cand = cand.sort_values('防御评分', ascending=False).head(cfg['top_n'])
        cand.insert(0, '排名', range(1, len(cand) + 1))

        cols = ['排名', '代码6', '名称', '涨幅', '振幅', '换手', '60日涨幅',
                'PE_TTM', 'PB', '流通市值', '防御评分', '稳健概率%',
                '低波分', '估值分', '盈利分', '趋势稳度分', '市值分']
        cols = [c for c in cols if c in cand.columns]
        main = cand[cols].reset_index(drop=True)
        main['流通市值'] = (main['流通市值'] / 1e8).round(1)

        note = (f"共 {len(main)} 只防御候选 | 平均防御评分 {main['防御评分'].mean():.2f} | "
                f"条件：振幅≤{cfg['max_amp']}% 换手≤{cfg['max_turnover']}% "
                f"PE∈[{cfg['min_pe']},{cfg['max_pe']}] PB≤{cfg['max_pb']}")
        return {'主表': main, 'note': note, 'tables': {}}
