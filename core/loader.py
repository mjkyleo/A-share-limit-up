# -*- coding: utf-8 -*-
"""
数据加载与识别模块（loader）
职责：
  1. 扫描文件夹，自动识别数据文件（真xlsx / csv / 通达信伪xlsx-GBK制表符文本）
  2. 判断文件类型：开盘快照 / 盘中批次快照 / 收盘快照 / 股票池 / 历史预测结果
  3. 从文件名或列名后缀提取数据日期（如 08-13-close.xlsx、涨停封流比[20260813]）
  4. 表头顺序无关的模糊列匹配 + 中文数值清洗（亿/万/%/+/亏损/--）
  5. 输出每个文件的能力评估：可用于"预测"还是"回测"

盘中批次快照（v2.1 新增，用于尾盘选股）：
  命名格式：日期 + 批次号，批次号表示当天第几次导出
    08-14-01.xlsx   当天第 1 次盘中导出（如 10:30）
    08-14-02.xlsx   当天第 2 次盘中导出（如 14:30 尾盘）
    08-14-open.xlsx  开盘快照    08-14-close.xlsx 收盘快照
  同一天多个批次都保留（各自入库），预测时取批次号最大者（最新盘中数据）。
"""
import glob
import hashlib
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

# ---------- 标准字段 ← 候选关键词（表头顺序无关，自动剥离[20260813]类后缀） ----------
COLUMN_MAP = {
    '代码':     ['代码', 'code'],
    '名称':     ['名称', '股票简称', 'name'],
    '涨幅':     ['涨幅', '涨跌幅'],
    '现价':     ['现价', '最新价', '收盘价'],
    '昨收':     ['昨收', '前收盘'],
    '开盘':     ['开盘', '今开'],
    '最高':     ['最高'], '最低': ['最低'],
    '封流比':   ['涨停封流比', '封流比'],
    '封成比':   ['涨停封成比', '封成比'],
    '封单额':   ['封单额_新', '封单额', '封单金额'],
    '封单量':   ['封单量_新', '封单量'],
    '首封时间': ['首次涨停时间', '首封时间', '涨停时间'],
    '开板次数': ['涨停开板次数', '开板次数'],
    '年涨停数': ['涨停次数', '年内涨停次数'],
    '量比':     ['量比'],
    '换手':     ['换手率', '换手'],
    '振幅':     ['振幅'],
    '委比':     ['委比%', '委比'],
    '主力净量': ['主力净量'],
    '增仓占比': ['主力增仓占比', '增仓占比'],
    '资金流向': ['主力资金流向'],
    '主力买入': ['主力买入金额'],
    '机构动向': ['机构动向'],
    '金叉个数': ['金叉个数', '金叉'],
    '总手':     ['总手', '成交量'],
    '笔数':     ['笔数'],
    '内外比':   ['内外比'],
    '流通市值': ['流通市值'],
    '市盈':     ['市盈(动)', '市盈率', 'TTM市盈率', '市盈'],
    '市净率':   ['市净率'],
    '净利增长': ['净利润增长率'],
    '散户数量': ['散户数量'],
    '上市日期': ['上市日期'],
    '细分行业': ['细分行业'],
    '所属行业': ['所属行业'],
    '开盘涨幅': ['开盘涨幅'],
    '竞价换手': ['竞价换手'],
    '竞价评级': ['竞价评级'],
    '异动类型': ['异动类型'],
    '5日涨幅':  ['5日涨幅'],
    '10日涨幅': ['10日涨幅'],
    '20日涨幅': ['20日涨幅'],
}
NUM_FIELDS = ['涨幅', '现价', '昨收', '开盘', '最高', '最低', '封流比', '封成比',
              '封单额', '封单量', '开板次数', '年涨停数', '量比', '换手', '振幅',
              '委比', '主力净量', '增仓占比', '资金流向', '主力买入', '机构动向',
              '金叉个数', '总手', '笔数', '内外比', '流通市值', '市净率', '净利增长',
              '散户数量', '开盘涨幅', '竞价换手', '5日涨幅', '10日涨幅', '20日涨幅']
# 预测关键字段（有这些字段的文件才能驱动对应策略）
SEAL_FIELDS = {'封流比', '封成比', '封单额', '封单量'}       # 封单类
TIME_FIELDS = {'首封时间', '开板次数'}                        # 封板质量类
AUCTION_FIELDS = {'开盘涨幅', '竞价换手'}                     # 竞价类


def norm_header(h):
    h = re.sub(r'\[.*?\]', '', str(h))
    return re.sub(r'\s+', '', h).replace('（', '(').replace('）', ')').lower()


def locate_columns(columns):
    norm_cols = {norm_header(c): c for c in columns}
    mapping, missing = {}, []
    for field, kws in COLUMN_MAP.items():
        hit = None
        for kw in kws:
            k = norm_header(kw)
            if k in norm_cols:
                hit = norm_cols[k]
                break
            cand = [o for n, o in norm_cols.items() if k in n]
            if cand:
                hit = min(cand, key=len)
                break
        (mapping.__setitem__(field, hit) if hit else missing.append(field))
    return mapping, missing


def cn_num(x):
    """'+10.19%'→10.19 | '1.65亿'→1.65e8 | '3328万'→33280000 | '--'/'亏损'/空→NaN"""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(',', '').replace('%', '').replace('+', '').strip()
    if s in ('--', '-', '', '亏损', 'None', 'nan', '无'):
        return np.nan
    m = re.match(r'^(-?[\d.]+)\s*(亿|万)?', s)
    if not m:
        return np.nan
    v = float(m.group(1))
    if m.group(2) == '亿':
        v *= 1e8
    elif m.group(2) == '万':
        v *= 1e4
    return v


def time_to_min(x):
    if pd.isna(x) or str(x).strip() in ('--', ''):
        return np.nan
    try:
        h, m, s = str(x).strip().split(':')
        return (int(h) - 9) * 60 + int(m) - 30 + int(s) / 60
    except Exception:
        return np.nan


def read_any(path):
    """自动识别三种格式：真xlsx / CSV / 通达信伪xlsx(GBK制表符)"""
    if path.lower().endswith(('.xlsx', '.xls')):
        try:
            return pd.read_excel(path, dtype=str)
        except Exception:
            pass  # 伪xlsx，走文本分支
    if path.lower().endswith('.csv'):
        for enc in ('utf-8-sig', 'gbk', 'gb18030'):
            try:
                return pd.read_csv(path, dtype=str, encoding=enc)
            except Exception:
                continue
    for enc in ('gbk', 'gb18030', 'utf-8-sig', 'utf-16'):
        for sep in ('\t', ','):
            try:
                df = pd.read_csv(path, sep=sep, dtype=str, encoding=enc)
                if df.shape[1] > 3:
                    return df
            except Exception:
                continue
    raise ValueError('无法识别的文件格式')


def extract_date(path, columns):
    """数据日期：优先列名后缀 [20260813]，其次文件名里的日期，最后文件修改时间"""
    for c in columns:
        m = re.search(r'\[(\d{8})', str(c))
        if m:
            return m.group(1)
    name = os.path.basename(path)
    m = re.search(r'(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})', name)      # 2026-08-13
    if m:
        return ''.join(m.groups())
    m = re.search(r'(\d{2})[-_.](\d{2})', name)                       # 08-13
    if m:
        return f'{datetime.now():%Y}{m.group(1)}{m.group(2)}'
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d')


def extract_batch(path):
    """盘中快照批次号：文件名"日期+序号"表示当天第几次导出，无批次返回 None。
    08-14-01 → 1 | 08-14-02 → 2 | 2026-08-14-02 → 2 | 08-14-close → None
    注意：含 open/close/收盘 等关键词的文件不是批次快照。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if re.search(r'open|close|开盘|收盘|竞价|早盘|盘后|自选|池', stem, re.I):
        return None
    rest = re.sub(r'^.*?20\d{2}[-_.]\d{2}[-_.]\d{2}', '', stem)       # 先剥完整日期
    if rest == stem:
        rest = re.sub(r'^.*?\d{2}[-_.]\d{2}', '', stem, count=1)      # 再剥短日期
    m = re.match(r'^[-_.](\d{1,2})$', rest)                           # 剩余部分须恰好是序号
    return int(m.group(1)) if m else None


def classify_by_content(df_raw, mapping):
    """内容特征识别快照时刻（文件名无关键词时的兜底）
    开盘快照（9:25-9:30导出）特征：
      ① 涨幅 ≈ 开盘涨幅（开盘后价格几乎没动）
      ② 全天换手 ≈ 竞价换手（成交只有集合竞价一笔）
    返回 'open' / 'close' / None(无法判断)
    """
    def med(field):
        col = mapping.get(field)
        if not col or col not in df_raw.columns:
            return None
        v = df_raw[col].apply(cn_num)
        return v.median(skipna=True) if v.notna().sum() > 5 else None

    zf, kg, hs, jj = med('涨幅'), med('开盘涨幅'), med('换手'), med('竞价换手')
    votes = []
    if zf is not None and kg is not None:
        # 注意：换手列可能是小数口径，涨幅类统一为百分数
        diff = abs(zf - kg)
        votes.append('open' if diff < 0.5 else 'close')
    if hs is not None and jj is not None and jj > 0:
        h = hs * 100 if hs < 1 else hs
        j = jj * 100 if jj < 1 else jj
        votes.append('open' if h <= j * 1.5 else 'close')
    if not votes:
        return None
    return 'open' if votes.count('open') > votes.count('close') else 'close'


def classify_file(path, mapping, date_str, df_raw=None):
    """判断文件类型与用途（三层判定：文件名关键词 → 内容特征 → 修改时间）
    kind: close 收盘 | open 开盘/竞价 | intraday 盘中批次 | pool 股票池(时刻不明) | pred 历史预测
    """
    name = os.path.basename(path).lower()
    fields = set(mapping)
    if re.search(r'排名结果|final|rank|预测|候选|回测|报告', name):
        return 'pred', 'backtest', '历史预测输出（用于回测对照）'
    src = '文件名'
    batch = extract_batch(path)
    if re.search(r'open|开盘|竞价|早盘|auction', name):
        kind = 'open'
    elif re.search(r'close|收盘|盘后', name):
        kind = 'close'
    elif batch is not None:
        kind = 'intraday'
    else:
        kind = None
        # 第二层：内容特征
        if df_raw is not None:
            kind = classify_by_content(df_raw, mapping)
            src = '内容特征' if kind else src
        # 第三层：文件修改时间（弱信号）
        if kind is None:
            hour = os.path.getmtime(path)
            from datetime import datetime as _dt
            hour = _dt.fromtimestamp(hour).hour
            kind = 'open' if hour < 12 else ('close' if hour >= 14 else 'pool')
            src = '修改时间'
        if kind == 'pool':
            src = '默认'
    has_seal = bool(fields & SEAL_FIELDS)
    has_time = bool(fields & TIME_FIELDS)
    has_auction = bool(fields & AUCTION_FIELDS)
    notes = [f'时刻判定:{src}']
    if kind == 'intraday':
        notes.append(f'盘中第{batch}批→可尾盘选股')
    if has_seal:
        notes.append('有封单数据→可预测涨停')
    if has_time:
        notes.append('有封板时间→封板质量可评')
    if has_auction:
        notes.append('有竞价数据→可早盘选股')
    if len(notes) == 1:
        notes.append('基础行情→仅参与综合评分')
    cap = 'predict'
    return kind, cap, '；'.join(notes)


def clean(df):
    """列定位 → 字段统一 → 数值化 → 衍生字段"""
    df.columns = [str(c) for c in df.columns]
    mapping, missing = locate_columns(df.columns)
    out = pd.DataFrame()
    for f, c in mapping.items():
        out[f] = df[c]
    for f in NUM_FIELDS:
        if f in out.columns:
            out[f] = out[f].apply(cn_num)
    if '市盈' in mapping:
        out['亏损股'] = df[mapping['市盈']].astype(str).str.contains('亏损')
    else:
        out['亏损股'] = False
    out['封板分钟'] = out['首封时间'].apply(time_to_min) if '首封时间' in out else np.nan
    if '代码' in out.columns:
        out['代码6'] = out['代码'].astype(str).str.extract(r'(\d{6})')
    out['今日涨停'] = (out['涨幅'] >= 9.5) if '涨幅' in out else False
    if '换手' in out and out['换手'].median(skipna=True) < 1:
        out['换手'] = out['换手'] * 100
    if '封流比' not in out and {'封单额', '流通市值'} <= set(out.columns):
        out['封流比'] = out['封单额'] / out['流通市值'] * 100
    if '封成比' not in out and {'封单量', '总手'} <= set(out.columns):
        out['封成比'] = out['封单量'] / out['总手'].replace(0, np.nan)
    return out, mapping, missing


def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as fp:
        h.update(fp.read(1 << 20))
    return h.hexdigest()


def scan_folder(base_dir):
    """扫描目录 → [文件信息dict]，不导入数据，只做识别与能力评估"""
    results = []
    skip = re.compile(r'(~\$|^排名|^预测|^周一|^回测|config|报告)', re.I)
    for path in sorted(glob.glob(os.path.join(base_dir, '*'))):
        if not path.lower().endswith(('.xlsx', '.xls', '.csv')):
            continue
        name = os.path.basename(path)
        info = {'path': path, '文件': name, '大小KB': round(os.path.getsize(path) / 1024, 1),
                'hash': file_hash(path)}
        if skip.search(name) and 'rank' not in name.lower() and 'final' not in name.lower():
            info.update(类型='输出文件', 日期='-', 行数='-', 能力='-', 说明='程序生成的结果文件，跳过')
            results.append(info)
            continue
        try:
            head = read_any(path)
            date_str = extract_date(path, head.columns)
            mapping, _ = locate_columns([str(c) for c in head.columns])
            kind, cap, note = classify_file(path, mapping, date_str, df_raw=head)
            info.update(类型={'close': '收盘', 'open': '开盘/竞价', 'intraday': '盘中批次',
                             'pool': '股票池', 'pred': '历史预测'}[kind],
                        日期=f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}',
                        行数=len(head), 能力={'predict': '✅ 可预测', 'backtest': '🔁 可回测'}[cap],
                        说明=note, kind=kind, cap=cap, date=date_str,
                        batch=extract_batch(path))
        except Exception as e:
            info.update(类型='无法读取', 日期='-', 行数='-', 能力='❌', 说明=str(e)[:50])
        results.append(info)
    return results


def load_dataset(base_dir, latest_date_only=True):
    """加载全部可预测数据文件，合并去重 → (合并DataFrame, 文件信息list)
    latest_date_only: 只保留最新数据日期（常规预测只用最近一天收盘数据）"""
    infos = [i for i in scan_folder(base_dir) if i.get('cap') == 'predict']
    frames = []
    for i in infos:
        try:
            d, _, _ = clean(read_any(i['path']))
            d['数据日期'] = i['date']
            d['来源文件'] = i['文件']
            d['快照类型'] = i['kind']
            frames.append(d)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(), infos
    all_df = pd.concat(frames, ignore_index=True)
    if latest_date_only:
        all_df = all_df[all_df['数据日期'] == all_df['数据日期'].max()]
    # 同代码+同日期去重：信息更全的行优先；收盘快照优先于开盘快照
    all_df['信息完整度'] = all_df.notna().sum(axis=1)
    all_df['快照优先级'] = all_df['快照类型'].map(
        {'close': 0, 'intraday': 1, 'pool': 2, 'open': 3}).fillna(2)
    all_df = (all_df.sort_values(['快照优先级', '信息完整度'], ascending=[True, False])
              .drop_duplicates(['代码6', '数据日期'], keep='first').reset_index(drop=True))
    return all_df, infos


def load_latest_intraday(base_dir):
    """加载最新日期的盘中批次快照（用于尾盘选股）。
    同一天多个批次（08-14-01 / 08-14-02）取批次号最大者 = 最接近收盘的盘中数据。
    返回 (DataFrame, info) ；无盘中快照时 DataFrame 为空。
    """
    infos = [i for i in scan_folder(base_dir)
             if i.get('cap') == 'predict' and i.get('kind') == 'intraday']
    if not infos:
        return pd.DataFrame(), None
    latest_date = max(i['date'] for i in infos)
    day_infos = [i for i in infos if i['date'] == latest_date]
    best = max(day_infos, key=lambda i: i.get('batch') or 0)
    d, _, _ = clean(read_any(best['path']))
    d['数据日期'] = best['date']
    d['来源文件'] = best['文件']
    d['快照类型'] = 'intraday'
    d['信息完整度'] = d.notna().sum(axis=1)
    d = (d.sort_values('信息完整度', ascending=False)
         .drop_duplicates('代码6', keep='first').reset_index(drop=True))
    return d, best