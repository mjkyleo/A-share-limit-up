# -*- coding: utf-8 -*-
"""
扩展框架基类（extensions.base）
================================
所有「能力」（如抄底反弹）继承 Extension，丢进 extensions/ 即可被自动发现，
无需修改 app.py / strategies.py 的核心逻辑。

所有「数据源」（本地文件夹 / 实时行情接口）继承 DataSource，丢进
extensions/datasources/ 即可被框架切换注入，扩展逻辑完全不感知数据来源。

Extension 把一项能力「所有需要的要求」都表面在这里：
  key / name / description  身份与说明
  params                   可配置参数（GUI 自动渲染控件，不手写界面）
  datasource               依赖的数据源 key（框架据此注入数据）
  list_type / scoring / hit_threshold / detail_keys   回测注册信息
  run(df, cfg, ctx)        核心逻辑：返回 {主表, note, tables}
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def normalize_market_cols(df):
    """将任意数据源的标准化 DataFrame 统一到扩展所需列口径；缺失列自动派生/置 NaN。
    目的：使扩展对「本地 data/ 文件」与「akshare 实时行情」两类数据源都鲁棒
    （akshare 不同版本列名不一致，可能缺失 振幅/60日涨幅 等）。"""
    d = df.copy()
    # 代码6：从 代码 或 index 提取 6 位
    if '代码6' not in d.columns:
        src = d['代码'] if '代码' in d.columns else d.index.astype(str)
        d['代码6'] = src.astype(str).str.extract(r'(\d{6})')[0]
    d['代码6'] = d['代码6'].astype(str).str.extract(r'(\d{6})')[0]
    # 名称
    if '名称' not in d.columns:
        d['名称'] = ''
    d['名称'] = d['名称'].astype(str)
    # 数值核心列：存在的转数值，缺失的置 NaN
    num = ['涨幅', '现价', '昨收', '开盘', '最高', '最低', '换手', '量比', '流通市值',
           '总市值', 'PE_TTM', 'PB', '60日涨幅', '5日涨幅', '10日涨幅', '20日涨幅',
           '振幅', '主力净量', '资金流向', '成交量', '成交额']
    for c in num:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        else:
            d[c] = np.nan
    # 振幅派生：(最高-最低)/昨收*100（akshare 缺 振幅 时）
    need_amp = d['振幅'].isna()
    if need_amp.any() and {'最高', '最低', '昨收'}.issubset(d.columns):
        base = d['昨收'].abs().replace(0, np.nan)
        d.loc[need_amp, '振幅'] = (d.loc[need_amp, '最高'] - d.loc[need_amp, '最低']) / base.loc[need_amp] * 100
    d['振幅'] = d['振幅'].fillna(0)
    # 60日/5/10/20 日涨幅：兼容 akshare 的「x日涨跌幅」命名
    alt_map = {'60日涨幅': '60日涨跌幅', '5日涨幅': '5日涨跌幅',
               '10日涨幅': '10日涨跌幅', '20日涨幅': '20日涨跌幅'}
    for std, alt in alt_map.items():
        if d[std].isna().all() and alt in df.columns:
            d[std] = pd.to_numeric(df[alt], errors='coerce')
    # PB / PE_TTM：兼容本地「市净率 / 市盈」命名
    if d['PB'].isna().all() and '市净率' in df.columns:
        d['PB'] = pd.to_numeric(df['市净率'], errors='coerce')
    if d['PE_TTM'].isna().all() and '市盈' in df.columns:
        d['PE_TTM'] = pd.to_numeric(df['市盈'], errors='coerce')
    # 亏损股
    if '亏损股' not in d.columns:
        pe = d.get('PE_TTM', pd.Series(dtype=float))
        st = d['名称'].str.contains(r'ST|退市', regex=True, na=False)
        d['亏损股'] = (pe.notna() & (pe < 0)) | st
    d['亏损股'] = d['亏损股'].fillna(False).astype(bool)
    # 今日涨停
    if '今日涨停' not in d.columns:
        d['今日涨停'] = d['涨幅'] >= 9.5
    return d


class Param:
    """扩展的可配置参数。GUI 据 kind 自动渲染对应控件。"""

    def __init__(self, name, label, kind='float', default=None,
                 min=None, max=None, choices=None, step=None, help=''):
        self.name = name          # 配置里的键名
        self.label = label        # 界面显示名
        self.kind = kind          # 'float' | 'int' | 'bool' | 'choice'
        self.default = default
        self.min = min
        self.max = max
        self.choices = choices    # kind='choice' 时的候选项
        self.step = step
        self.help = help


class DataSource(ABC):
    """数据源：把任意来源的数据变成与 loader.clean 同口径的标准化 DataFrame。"""

    key = 'local'
    description = ''

    @abstractmethod
    def fetch(self, ctx):
        """返回 (df, info_dict)。
        df 标准列含：代码6 / 名称 / 涨幅 / 20日涨幅 / 5日涨幅 / 量比 / 换手 /
            主力净量 / 资金流向 / 最高 / 最低 / 现价 / 亏损股 / 数据日期 ...
        info_dict 至少含：来源 / date(YYYYMMDD) / 行数
        """
        raise NotImplementedError


class Extension(ABC):
    """一项可插拔的分析能力。子类只需填属性 + 实现 run()。"""

    key = ''                 # 唯一 ID（config.extensions 的键、回测 list_type 同源）
    name = ''
    description = ''
    params = []              # list[Param]
    datasource = 'local'     # 注入的数据源 key（见 DataSource 注册表）
    list_type = ''           # 回测清单类型（save_predictions 用，留空=不参与回测）
    scoring = []             # 回测打分规则 [[涨幅下限%, 得分], ...] 升序
    hit_threshold = 0.0      # 回测「达成」口径：实际涨幅 ≥ 该值
    detail_keys = []         # 存入 predictions.detail 的因子列（供单因子 IC 分析）
    prob_col = '预估反弹概率%'   # 入库概率列名
    score_col = '抄底评分'        # 入库评分列名

    def effective_params(self, cfg_ext, overrides=None):
        """合并三层：扩展默认参数 ← config.extensions[key].params ← 界面控件当前值。"""
        p = {pp.name: pp.default for pp in self.params}
        if cfg_ext and isinstance(cfg_ext.get('params'), dict):
            p.update(cfg_ext['params'])
        if overrides:
            p.update(overrides)
        return p

    @abstractmethod
    def run(self, df, cfg, ctx):
        """核心逻辑。
        df   : 数据源注入的标准化当日数据
        cfg  : 本扩展合并后的参数（effective_params 的结果）
        ctx  : 上下文（含 db / base_dir / data_dir / target_date 等）
        返回 : {'主表': DataFrame, 'note': str, 'tables': {子表名: DataFrame}}
        """
        raise NotImplementedError
