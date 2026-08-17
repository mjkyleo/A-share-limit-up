# -*- coding: utf-8 -*-
"""
抄底反弹扩展（extensions/bottom_fishing.py）
============================================
筛选逻辑（三层交集）：
  ① 今日急跌：跌幅落在 [min_drop, max_drop] 区间（默认 7%~9.5%，方向/区间均可调）
  ② 不是大下跌趋势：20日涨幅 ≥ min_20d_change（中期仍向上）+ 5日涨幅 ≥ min_5d_change
       （可选）require_support：收盘远离最低（下影线比>0.4）才算有承接
       （可选）exclude_consecutive_down：用 data.db 历史排除近 N 日连跌（趋势已坏）
  ③ 次日反弹动能：六因子打分 × 风险系数 → 排序 → 清单 + 预估反弹概率

落库：save_predictions(list_type='抄底清单')，次日收盘数据入库后可在「回测评估」同口径评分。
新增其他能力：复制本文件改 run() 与参数即可，框架自动发现、自动建 Tab、自动可回测。
"""
import numpy as np
import pandas as pd

from extensions.base import Extension, Param


class BottomFishing(Extension):
    key = 'bottom_fishing'
    name = '抄底反弹清单'
    description = ('今日急跌但中期未走坏、次日有反弹动能的低吸清单。'
                   '数据源：本地 data/ 文件夹（收盘快照）。')
    datasource = 'local'
    list_type = '抄底清单'
    prob_col = '预估反弹概率%'
    score_col = '抄底评分'
    # 回测打分：次日反弹≥2% 算达成；涨得多得分高，继续跌扣分
    scoring = [[-99, -80], [-6, -50], [-4, -30], [-2, -15], [0, 20],
               [2, 60], [4, 80], [6, 95], [9.5, 100]]
    hit_threshold = 2.0
    detail_keys = ['抄底评分', '超跌度', '量能健康', '资金未逃', '承接力', '位置',
                   '股性', 'R风险系数', '下影线比', '今日涨停']

    params = [
        Param('min_drop', '最小跌幅%(急跌门槛)', 'float', 7.0, 0, 20, step=0.5,
              help='今日跌幅至少这么大才纳入（如 7 = 跌≥7%）。设 0 即覆盖轻微回调'),
        Param('max_drop', '最大跌幅%(排除跌停)', 'float', 9.5, 0, 20, step=0.5,
              help='跌幅超过此值视为跌停/崩盘，默认排除（-9.5）。调大到 20 可纳入跌停'),
        Param('min_20d_change', '20日涨幅下限%', 'float', 0.0, -50, 100, step=1,
              help='中期仍向上闸门：20日涨幅≥此值（默认 0，排除下行趋势）'),
        Param('min_5d_change', '5日涨幅下限%', 'float', -99.0, -50, 100, step=1,
              help='短期不过分弱势（默认 -99 不限制）'),
        Param('require_support', '要求收盘远离最低(有承接)', 'bool', True,
              help='下影线比>0.4 才纳入：盘中跌下去有人接，不是裸跌'),
        Param('exclude_consecutive_down', '排除连续下跌(需历史)', 'bool', False,
              help='开启后用 data.db 历史判断近 N 日是否连跌，连跌则排除'),
        Param('consecutive_days', '连续下跌天数阈值', 'int', 3, 1, 10,
              help='排除连续下跌 ≥ 此天数的股票'),
        Param('bottom_n', '清单长度', 'int', 20, 1, 100),
    ]

    # ---------------- 打分因子（输出均 [0,1]） ----------------
    def _oversold(self, drop):
        """超跌度：跌 7%~9% 最佳（错杀区），接近跌停恐慌加剧，跌停/崩盘不确定性大。"""
        if drop < 7:
            return 0.55
        if drop <= 9:
            return 1.0
        if drop <= 11:
            return 0.8
        return 0.5

    def _volume_health(self, lb, hs):
        """量能健康：温和缩量/正常量跌 = 抛压轻；爆量跌 = 出逃/派发。"""
        s = 1.0
        if pd.notna(lb):
            s *= 1.0 if 0.5 <= lb <= 2 else 0.7 if lb < 0.5 else 0.4 if lb <= 4 else 0.2
        if pd.notna(hs):
            h = hs * 100 if hs < 1 else hs
            s *= 1.0 if h <= 8 else 0.7 if h <= 15 else 0.4
        return s

    def _money(self, net, flow):
        """资金未出逃：主力净量/资金流向不转负。"""
        vals = []
        if pd.notna(net):
            vals.append(1.0 if net >= 0 else 0.2)
        if pd.notna(flow):
            vals.append(1.0 if flow > 0 else 0.2)
        return np.mean(vals) if vals else 0.5

    def _support(self, row):
        """承接力 = 下影线比 = (现价-最低)/(最高-最低)，越高说明收盘远离最低、有支撑。"""
        hi, lo, cur = row.get('最高'), row.get('最低'), row.get('现价')
        if pd.isna(hi) or pd.isna(lo) or pd.isna(cur) or hi <= lo:
            return 0.5
        rng = hi - lo
        if rng <= 0:
            return 0.5
        return float(min(max((cur - lo) / rng, 0), 1))

    def _position(self, g20):
        """位置：中期向上且未过热最佳；高位接力风险大。"""
        if pd.isna(g20):
            return 0.5
        if g20 < 0:
            return 0.3
        if g20 < 40:
            return 1.0
        if g20 < 60:
            return 0.7
        return 0.4

    def _structure(self, mv):
        """股性/市值：小中盘波动大、反弹弹性高。"""
        if pd.isna(mv):
            return 0.5
        y = mv / 1e8
        return 1.0 if y < 50 else 0.7 if y < 80 else 0.4 if y < 200 else 0.2

    def _risk(self, row, cfg):
        """向下风险系数（0~1）：一字跌停/高位/亏损/巨换手/连跌 均减分。"""
        p = 1.0
        if 'ST' in str(row.get('名称', '')):
            p *= 0.5
        if row.get('亏损股', False):
            p *= 0.85
        g20 = row.get('20日涨幅')
        if pd.notna(g20) and g20 > 60:
            p *= 0.7
        hs = row.get('换手')
        if pd.notna(hs):
            h = hs * 100 if hs < 1 else hs
            if h > 25:
                p *= 0.8
        # 一字跌停：开盘≈最低≈收盘 且 封死 → 大概率继续跌
        if pd.notna(row.get('涨幅')) and row['涨幅'] <= -9.5:
            op, lo, hi = row.get('开盘'), row.get('最低'), row.get('最高')
            if pd.notna(op) and pd.notna(lo) and pd.notna(hi) and (hi - lo) > 0:
                if (op - lo) <= (hi - lo) * 0.15:
                    p *= 0.4
        return round(p, 3)

    # ---------------- 主流程 ----------------
    def run(self, df, cfg, ctx):
        if df is None or df.empty or '涨幅' not in df.columns:
            return {'主表': pd.DataFrame(),
                    'note': '无可用数据（请先投放当日收盘快照到 data/）', 'tables': {}}
        from extensions.base import normalize_market_cols
        d = normalize_market_cols(df)

        min_drop = float(cfg['min_drop'])
        max_drop = float(cfg['max_drop'])

        # ① 急跌筛选
        drop = -d['涨幅']
        pool = d[(drop >= min_drop) & (drop <= max_drop)].copy()
        if pool.empty:
            return {'主表': pd.DataFrame(),
                    'note': f'今日无符合「跌幅 {min_drop}%~{max_drop}%」的股票', 'tables': {}}

        # ② 趋势闸门
        if '20日涨幅' in pool.columns:
            pool = pool[pool['20日涨幅'].fillna(-999) >= float(cfg['min_20d_change'])]
        if '5日涨幅' in pool.columns and float(cfg['min_5d_change']) > -99:
            pool = pool[pool['5日涨幅'].fillna(-999) >= float(cfg['min_5d_change'])]
        if pool.empty:
            return {'主表': pd.DataFrame(),
                    'note': '急跌股中无「中期未走坏」者（20日/5日涨幅闸门过滤后为空）', 'tables': {}}

        # ②b 承接力
        sup = pool.apply(self._support, axis=1)
        pool['下影线比'] = sup.round(3)
        if cfg['require_support']:
            pool = pool[sup > 0.4]
            if pool.empty:
                return {'主表': pd.DataFrame(),
                        'note': '急跌股收盘均贴近最低（无承接），暂不建议抄底', 'tables': {}}

        # ②c 排除连续下跌（用 data.db 历史）
        if cfg['exclude_consecutive_down'] and ctx and getattr(ctx, 'db', None):
            pool = self._exclude_consecutive(pool, ctx.db, int(cfg['consecutive_days']))
            if pool.empty:
                return {'主表': pd.DataFrame(),
                        'note': f'急跌股均处于连续下跌（≥{int(cfg["consecutive_days"])}日），趋势已坏',
                        'tables': {}}

        # ③ 六因子打分
        pool['超跌度'] = (-pool['涨幅']).apply(self._oversold).round(3)
        pool['量能健康'] = pool.apply(
            lambda r: self._volume_health(r.get('量比'), r.get('换手')), axis=1).round(3)
        pool['资金未逃'] = pool.apply(
            lambda r: self._money(r.get('主力净量'), r.get('资金流向')), axis=1).round(3)
        pool['承接力'] = pool['下影线比']
        pool['位置'] = pool['20日涨幅'].apply(self._position).round(3)
        pool['股性'] = pool['流通市值'].apply(self._structure).round(3)

        w = {'超跌度': 0.25, '量能健康': 0.15, '资金未逃': 0.25,
             '承接力': 0.20, '位置': 0.10, '股性': 0.05}
        score = sum(pool[k] * v for k, v in w.items())
        pool['R风险系数'] = pool.apply(lambda r: self._risk(r, cfg), axis=1)
        pool['抄底评分'] = (score * pool['R风险系数']).round(3)
        pool['预估反弹概率%'] = (np.minimum(0.15 + pool['抄底评分'] * 0.45, 0.6)
                                 * pool['R风险系数'] * 100).round(1)

        res = (pool.sort_values('抄底评分', ascending=False)
               .head(int(cfg['bottom_n'])).reset_index(drop=True))
        res.insert(0, '排名', res.index + 1)

        cols = ['排名', '代码6', '名称', '今日涨停', '涨幅', '20日涨幅', '5日涨幅',
                '量比', '换手', '主力净量', '资金流向', '下影线比',
                '超跌度', '量能健康', '资金未逃', '承接力', '位置', '股性',
                'R风险系数', '抄底评分', '预估反弹概率%']
        cols = [c for c in cols if c in res.columns]
        note = (f'急跌池（跌{min_drop}%~{max_drop}%）{len(d)}→中期未坏 {len(pool)} 只→'
                f'入选 {len(res)} 只 | 平均预估反弹概率 '
                f'{res["预估反弹概率%"].mean():.1f}%' if len(res) else '无入选')
        return {'主表': res[cols], 'note': note, 'tables': {}}

    def _exclude_consecutive(self, pool, db, n):
        keep = []
        for _, r in pool.iterrows():
            code = str(r.get('代码6', '') or '')
            if not code or code == 'nan':
                keep.append(True)
                continue
            ser = db.get_code_series(code, limit=max(n + 1, 6))
            if ser is None or len(ser) < n:
                keep.append(True)          # 历史不足，不误杀
                continue
            recent = ser['涨幅'].tail(n).tolist()
            keep.append(not all(x < 0 for x in recent))   # 最近 n 日全跌才排除
        return pool[pd.Series(keep, index=pool.index)].copy()
