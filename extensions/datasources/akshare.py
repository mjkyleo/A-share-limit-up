# -*- coding: utf-8 -*-
"""AKShare 实时数据源：免费拉取东财/新浪全市场快照，供 extensions 使用。

复用标准列口径：
  代码6 / 名称 / 涨幅 / 现价 / 最高 / 最低 / 开盘 / 昨收 / 换手 /
  量比 / 振幅 / 总市值 / 流通市值 / 60日涨幅 / 亏损股 / 今日涨停 / 数据日期 /
  （可选）5/10/20日涨幅（若接口提供则映射，否则留空由扩展自行退化）

说明：
  - AKShare 的 spot 接口不含 5/10/20日涨幅、主力净量、资金流向、封流比、
    封成比、封板分钟、开板次数、年涨停数 等“涨停板细节”字段。
  - 因此基于本数据源的分析应以“价格/量能/市场状态”类因子为主；
    需要涨停细节的 S1-S5 策略仍建议使用本地 data/ 文件导入。
  - 健壮性：东财接口偶发被反爬中断，内置重试 + 新浪行情兜底；任一失败不致命。
"""
import re
import time
from datetime import datetime

import pandas as pd

from extensions.base import DataSource


def _col_map_spot(df_spot):
    """把东财/新浪现货 DataFrame 映射到项目标准列。返回 (out, 命中字段)。"""
    out = pd.DataFrame()
    # 标准列 ← 候选原始列（按优先级）
    rules = {
        '名称': ['名称', 'name'],
        '现价': ['最新价', '现价', 'close'],
        '涨幅': ['涨跌幅', '涨幅', 'pct_chg'],
        '涨跌额': ['涨跌额', 'price_chg'],
        '成交量': ['成交量'],
        '成交额': ['成交额'],
        '振幅': ['振幅'],
        '最高': ['最高'],
        '最低': ['最低'],
        '开盘': ['今开', '开盘'],
        '昨收': ['昨收'],
        '量比': ['量比'],
        '换手': ['换手率', '换手'],
        'PE_TTM': ['市盈率-动态', '市盈率'],
        'PB': ['市净率'],
        '总市值': ['总市值'],
        '流通市值': ['流通市值'],
        '涨速': ['涨速'],
        '5分钟涨跌': ['5分钟涨跌'],
        '60日涨幅': ['60日涨跌幅', '60日涨幅'],
        '年初至今涨幅': ['年初至今涨跌幅'],
    }
    for std, cands in rules.items():
        for c in cands:
            if c in df_spot.columns:
                if std == '名称':
                    out[std] = df_spot[c].astype(str)
                else:
                    out[std] = pd.to_numeric(df_spot[c], errors='coerce')
                break
    out['代码6'] = df_spot['代码'].astype(str).str.extract(r'(\d{6})')[0] \
        if '代码' in df_spot.columns else df_spot.index.astype(str)
    return out


class AKShareDataSource(DataSource):
    key = 'akshare'
    description = 'AKShare 东财/新浪实时行情（免费，全市场快照 + 市场活跃度）'

    def fetch(self, ctx):
        import akshare as ak

        df, source_used = self._fetch_spot(ak)
        info = {'来源': source_used, 'date': '', '行数': 0 if df is None else len(df)}
        if df is None or df.empty:
            info['error'] = '现货行情拉取失败（东财/新浪均不可用）'
            return pd.DataFrame(), info

        out = df
        out['数据日期'] = self._trade_date()
        out['今日涨停'] = out['涨幅'] >= 9.5
        pe_bad = out.get('PE_TTM', pd.Series(dtype=float)).notna() & (out.get('PE_TTM', pd.Series(dtype=float)) < 0)
        st_bad = out['名称'].astype(str).str.contains(r'ST|退市', regex=True, na=False)
        out['亏损股'] = pe_bad | st_bad

        # 2) 市场活跃度（涨跌家数、涨停/跌停数、活跃度）—— 失败不致命
        try:
            activity = ak.stock_market_activity_legu()
            act = dict(zip(activity['item'].astype(str), activity['value']))
            info['market_activity'] = act
        except Exception as e:
            info['market_activity'] = {'error': str(e)}

        info['date'] = out['数据日期'].iloc[0] if not out.empty else ''
        return out, info

    # ---------------- 内部工具 ----------------

    def _fetch_spot(self, ak, max_retry=3):
        """东财优先，失败退避重试，再退新浪。返回 (df, source_name) 或 (None, msg)。"""
        last_err = ''
        # 1) 东财
        for attempt in range(max_retry):
            try:
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    return _col_map_spot(df), 'AKShare 东财实时行情'
            except Exception as e:
                last_err = repr(e)
                time.sleep(1.5 * (attempt + 1))
        # 2) 新浪兜底
        try:
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                return _col_map_spot(df), 'AKShare 新浪实时行情(兜底)'
        except Exception as e:
            last_err = repr(e)
        return None, f'东财+新浪均失败: {last_err[:80]}'

    def _trade_date(self):
        try:
            from core import calendar_cn as cal
            td = cal.today()
            if td:
                return td.strftime('%Y%m%d')
        except Exception:
            pass
        return datetime.now().strftime('%Y%m%d')
