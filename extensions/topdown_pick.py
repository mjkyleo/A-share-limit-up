# -*- coding: utf-8 -*-
"""
自上而下选股扩展（extensions.topdown_pick）
========================================
把用户确立的选股思想「上证指数择时 → 行业板块找主线 → 主线龙头找个股」
做成一个可插拔能力，与抄底/反转/防御/神经网络并列接入框架与回测。

- datasource='local'：作用于已加载的自选池（data/ 收盘快照），无需实时行情；
  择时优先用 AKShare 上证指数，取不到则回退到自选池市场宽度代理。
- run() 返回 主表（含 综合得分/择时系数/市道/所属主线板块/板块强度/龙头评分），
  可直接入库、进回测、在 GUI 展示。
"""
from extensions.base import Extension, Param, normalize_market_cols
from core.topdown import topdown_rank


class TopdownPickExtension(Extension):
    key = 'topdown_pick'
    name = '自上而下选股'
    description = ('思想：上证指数择时定仓位 → 行业板块找主线 → 主线内挑龙头。'
                   '综合得分=(0.6·龙头+0.4·板块强度)·择时系数·(非主线降权)，'
                   '真正体现「择时定方向、板块定主线、龙头定标的」。')
    datasource = 'local'
    list_type = '自上而下'
    prob_col = None                       # 选股框架，不给伪概率
    score_col = '综合得分'
    scoring = [[-99, -80], [-6, -50], [-4, -30], [-2, -15], [0, 20],
               [2, 60], [4, 80], [6, 95], [9.5, 100]]
    hit_threshold = 2.0                  # 次日涨≥2% 视为命中
    detail_keys = ['所属行业', '板块强度', '龙头评分', '择时系数', '市道', '是否在主线']

    params = [
        Param('top_n_sectors', '主线板块数', 'int', 5, 1, 12,
              help='强度排序前 N 的板块视为「主线」'),
        Param('min_sector_count', '板块最小样本', 'int', 3, 2, 20,
              help='样本少于此数的行业不单独评级'),
        Param('timing_method', '择时方式', 'choice', 'auto',
              choices=['auto', 'index', 'breadth', 'synthetic'],
              help='auto=按 config.timing.method(默认 synthetic 合成信号)；synthetic=上证+中证1000+创业板指+宽度 合成；index=仅上证；breadth=宽度代理'),
        Param('weight_sector', '板块权重%', 'int', 40, 0, 80,
              help='综合得分中板块强度的占比'),
        Param('weight_leader', '龙头权重%', 'int', 60, 20, 100,
              help='综合得分中龙头评分的占比'),
        Param('gate_non_main', '非主线降权%', 'int', 70, 0, 95,
              help='非主线个股综合得分再乘 (1 - 该值)，强排到后排'),
        Param('top_n', '入选只数', 'int', 30, 5, 100,
              help='最终清单保留前 N 只'),
    ]

    def run(self, df, cfg, ctx=None):
        if df is None or df.empty or '涨幅' not in df.columns:
            return {'主表': __import__('pandas').DataFrame(),
                    'note': '无可用数据（请先投放当日收盘快照到 data/）', 'tables': {}}
        d = normalize_market_cols(df)
        if '所属行业' not in d.columns or d['所属行业'].isna().all():
            return {'主表': __import__('pandas').DataFrame(),
                    'note': '数据缺少「所属行业」列，无法做板块主线分析', 'tables': {}}

        def num(name, default):
            try:
                return float(cfg.get(name, default))
            except Exception:
                return float(default)

        timing, sector_df, ranked = topdown_rank(
            d,
            ak=None,
            top_n_sectors=int(num('top_n_sectors', 5)),
            min_sector_count=int(num('min_sector_count', 3)),
            force_method=cfg.get('timing_method', 'auto'),
            weight_sector=int(num('weight_sector', 40)),
            weight_leader=int(num('weight_leader', 60)),
            gate_non_main=int(num('gate_non_main', 70)),
        )

        if ranked.empty:
            return {'主表': __import__('pandas').DataFrame(),
                    'note': '板块聚合后无足够样本，无法产出榜单', 'tables': {}}

        top_n = int(num('top_n', 30))
        main = ranked.head(top_n).copy()
        main = main.drop(columns=['排名'], errors='ignore')
        main.insert(0, '排名', range(1, len(main) + 1))

        cols = ['排名', '代码6', '名称', '所属行业', '综合得分', '择时系数', '市道',
                '是否在主线', '板块强度', '龙头评分', '5日涨幅', '10日涨幅',
                '20日涨幅', '主力净量', '流通市值', '现价', '市盈', '市净率']
        cols = [c for c in cols if c in main.columns]
        main = main[cols]

        # 板块主线子表（供 GUI 展开查看）
        if not sector_df.empty:
            sec = sector_df[['板块排名', '所属行业', '样本数', '平均涨幅', '平均20日',
                             '涨停数', '主力净量均值', '强度分']].copy()
            sec = sec.rename(columns={'平均涨幅': '平均涨幅%', '平均20日': '平均20日%'})
        else:
            sec = __import__('pandas').DataFrame()

        lines = "、".join(sector_df.head(int(num('top_n_sectors', 5)))['所属行业'].tolist())
        note = (f"择时：{timing['regime']}（仓位系数{timing['coeff']}，"
                f"{timing['method']}）；主线板块：{lines or '无'}；"
                f"入选 {len(main)} 只（其中主线 {int(main['是否在主线'].sum())} 只）")
        return {'主表': main, 'note': note, 'tables': {'板块强度': sec}}
