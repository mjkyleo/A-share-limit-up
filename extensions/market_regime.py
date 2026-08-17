# -*- coding: utf-8 -*-
"""市场环境识别（Market Regime）
================================
基于 AKShare 免费接口判断当前 A 股风险偏好状态，输出建议启用的策略风格。

判定维度：
  - 全市场涨跌比（乐咕乐股 market_activity）
  - 涨跌停家数比（涨停/跌停）
  - 主要指数 20 日/5 日涨幅与波动率
  - 全市场成交额活跃度

输出状态：
  risk_on      -> 动量/涨停接力友好
  neutral      -> 震荡/磨底，观望或轻仓
  risk_off     -> 反转/低吸友好
  panic        -> 恐慌，防御或空仓为主
"""
import numpy as np
import pandas as pd

from extensions.base import Extension, Param


class MarketRegimeExtension(Extension):
    key = 'market_regime'
    name = '市场环境识别'
    description = '判断当前 A 股风险偏好：risk_on/neutral/risk_off/panic，并给出策略建议'
    datasource = 'akshare'
    list_type = ''              # 非选股清单，不入库
    prob_col = ''
    score_col = ''
    detail_keys = []
    params = [
        Param('ma_window', '指数回看窗口(日)', 'int', 20, 5, 120, step=1,
              help='计算市场涨幅与波动率的窗口'),
        Param('vol_window', '波动率窗口(日)', 'int', 20, 5, 120, step=1),
        Param('risk_on_up_ratio', 'risk_on 涨跌比阈值', 'float', 0.55, 0.3, 0.9, step=0.05,
              help='上涨家数占比高于此值为 risk_on'),
        Param('risk_off_up_ratio', 'risk_off 涨跌比阈值', 'float', 0.45, 0.1, 0.7, step=0.05),
        Param('panic_drop_5d', 'panic 5日跌幅阈值%', 'float', -3.0, -10, 0, step=0.5),
    ]

    def run(self, df, cfg, ctx):
        import akshare as ak
        from core import calendar_cn as cal

        act = ctx.info.get('market_activity', {}) if ctx.info else {}
        if not act or 'error' in act:
            act = self._fetch_activity()

        # 解析市场活跃度
        up = self._to_float(act.get('上涨', 0))
        down = self._to_float(act.get('下跌', 0))
        flat = self._to_float(act.get('平盘', 0))
        total = up + down + flat
        up_ratio = up / total if total > 0 else np.nan

        limit_up = self._to_float(act.get('真实涨停', act.get('涨停', 0)))
        limit_down = self._to_float(act.get('真实跌停', act.get('跌停', 0)))
        zt_dt_ratio = limit_up / max(limit_down, 1)

        # 主要指数历史
        idx = self._fetch_index_metrics(cfg)

        # 成交额活跃度：用 spot 全市场成交额 / 近 20 日均额
        market_amount = df['成交额'].sum() if '成交额' in df.columns else np.nan
        amount_ratio = np.nan
        if not np.isnan(market_amount):
            # 这里用指数成交量的同比近似（更准可用 spot 多日累加，但 AKShare 限速）
            amount_ratio = idx.get('amount_ratio', np.nan)

        # 状态判定
        m20 = idx.get('ret_20d', 0)
        m5 = idx.get('ret_5d', 0)
        vol20 = idx.get('vol_20d', 0)

        if (up_ratio >= cfg['risk_on_up_ratio'] and m20 > 0
                and zt_dt_ratio >= 3 and limit_up >= 50):
            regime = 'risk_on'
            advice = '动量/涨停接力友好：可运行涨停TopN、尾盘选股'
        elif m5 <= cfg['panic_drop_5d'] and up_ratio < 0.4 and zt_dt_ratio < 1:
            regime = 'panic'
            advice = '恐慌：建议空仓或只做超跌/反转，禁用涨停接力'
        elif up_ratio <= cfg['risk_off_up_ratio'] or m20 < 0:
            regime = 'risk_off'
            advice = 'risk_off/反转友好：建议运行抄底反弹、短线反转、防御策略'
        else:
            regime = 'neutral'
            advice = '震荡/磨底：轻仓试盘，关注流动性反弹与独立个股'

        out = pd.DataFrame([{
            '状态': regime,
            '建议': advice,
            '上涨家数': int(up) if not np.isnan(up) else None,
            '下跌家数': int(down) if not np.isnan(down) else None,
            '涨跌比': round(up_ratio, 3) if not np.isnan(up_ratio) else None,
            '真实涨停': int(limit_up) if not np.isnan(limit_up) else None,
            '真实跌停': int(limit_down) if not np.isnan(limit_down) else None,
            '涨跌停比': round(zt_dt_ratio, 2),
            '沪深300_20日涨幅%': round(m20, 2),
            '沪深300_5日涨幅%': round(m5, 2),
            '沪深300_20日波动%': round(vol20, 2),
            '全市场成交额(亿)': round(market_amount / 1e8, 1) if not np.isnan(market_amount) else None,
        }])

        note = (f"当前状态：{regime} | {advice} | "
                f"沪深300 20日{m20:.1f}% 5日{m5:.1f}% | "
                f"涨跌比{up_ratio:.1%} | 涨停{int(limit_up) if not np.isnan(limit_up) else '-'}/"
                f"跌停{int(limit_down) if not np.isnan(limit_down) else '-'}")
        return {'主表': out, 'note': note, 'tables': {}}

    def _fetch_activity(self):
        import akshare as ak
        try:
            df = ak.stock_market_activity_legu()
            return dict(zip(df['item'].astype(str), df['value']))
        except Exception as e:
            return {'error': str(e)}

    def _fetch_index_metrics(self, cfg):
        import akshare as ak
        end = pd.Timestamp.now().strftime('%Y%m%d')
        start = (pd.Timestamp.now() - pd.Timedelta(days=180)).strftime('%Y%m%d')

        metrics = {'ret_20d': np.nan, 'ret_5d': np.nan, 'vol_20d': np.nan, 'amount_ratio': np.nan}
        try:
            hist = ak.stock_zh_a_hist(symbol='000300', period='daily',
                                      start_date=start, end_date=end, adjust='')
            if hist.empty or '收盘' not in hist.columns:
                return metrics
            hist['日期'] = pd.to_datetime(hist['日期'])
            hist = hist.sort_values('日期')
            close = hist['收盘'].astype(float)
            amount = hist['成交额'].astype(float) if '成交额' in hist.columns else None

            w = cfg['ma_window']
            if len(close) >= w + 1:
                metrics['ret_20d'] = (close.iloc[-1] / close.iloc[-(w + 1)] - 1) * 100
            if len(close) >= 6:
                metrics['ret_5d'] = (close.iloc[-1] / close.iloc[-6] - 1) * 100
            if len(close) >= w:
                metrics['vol_20d'] = close.iloc[-w:].pct_change().dropna().std() * np.sqrt(252) * 100
            if amount is not None and len(amount) >= w:
                metrics['amount_ratio'] = amount.iloc[-1] / amount.iloc[-w:].mean()
        except Exception:
            pass
        return metrics

    @staticmethod
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return 0.0
