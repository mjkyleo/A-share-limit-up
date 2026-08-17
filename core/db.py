# -*- coding: utf-8 -*-
"""
SQLite 轻量数据库（db）
表结构：
  files       — 已入库的数据文件（去重靠文件hash）
  daily       — 每日每只股票的清洗后指标（JSON整行存储，字段演进不丢数据）
  predictions — 每次预测输出（run_date=数据日期, target_date=预测目标交易日）
  回测不建表：由 backtest 模块用 predictions ⋈ daily 实时计算，保证口径最新
"""
import json
import os
import sqlite3
from datetime import datetime

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
  hash TEXT PRIMARY KEY, path TEXT, date TEXT, kind TEXT, rows INT,
  imported_at TEXT);
CREATE TABLE IF NOT EXISTS daily(
  date TEXT, code TEXT, name TEXT, json TEXT,
  PRIMARY KEY(date, code));
CREATE TABLE IF NOT EXISTS predictions(
  run_date TEXT, target_date TEXT, list_type TEXT, rank INT,
  code TEXT, name TEXT, prob REAL, score REAL, detail TEXT,
  PRIMARY KEY(run_date, list_type, code));
CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_date);
"""


class DB:
    def __init__(self, base_dir):
        self.path = os.path.join(base_dir, 'data.db')
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------- 文件与每日数据 ----------------
    def file_imported(self, h):
        return self.conn.execute('SELECT 1 FROM files WHERE hash=?', (h,)).fetchone() is not None

    def save_snapshot(self, info, df):
        """入库一个数据文件：files 元数据 + daily 明细。
        daily 表以收盘数据为准：同日已有 close 文件时，open/intraday/pool
        快照只登记 files、不覆盖 daily（防止盘中/开盘数据污染回测基准）。"""
        cur = self.conn.cursor()
        cur.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?)',
                    (info['hash'], info['path'], info.get('date', ''), info.get('kind', ''),
                     len(df), datetime.now().isoformat(timespec='seconds')))
        date_str = info.get('date', '')
        kind = info.get('kind', '')
        if kind != 'close':
            has_close = cur.execute(
                "SELECT 1 FROM files WHERE date=? AND kind='close'",
                (date_str,)).fetchone()
            if has_close:
                self.conn.commit()
                return 0
        n = 0
        for _, r in df.iterrows():
            code = str(r.get('代码6', '') or '')
            if not code or code == 'nan':
                continue
            rec = {k: (None if pd.isna(v) else (v.item() if hasattr(v, 'item') else v))
                   for k, v in r.items() if k not in ('来源文件',)}
            cur.execute('INSERT OR REPLACE INTO daily VALUES(?,?,?,?)',
                        (date_str, code, str(r.get('名称', '')),
                         json.dumps(rec, ensure_ascii=False, default=str)))
            n += 1
        self.conn.commit()
        return n

    def get_daily(self, date_str):
        rows = self.conn.execute(
            'SELECT code, name, json FROM daily WHERE date=?', (date_str,)).fetchall()
        if not rows:
            return pd.DataFrame()
        recs = []
        for code, name, js in rows:
            d = json.loads(js)
            d['代码6'], d['名称'] = code, name
            recs.append(d)
        return pd.DataFrame(recs)

    def available_dates(self):
        return [r[0] for r in self.conn.execute(
            'SELECT DISTINCT date FROM daily ORDER BY date').fetchall()]

    def get_code_series(self, code, limit=30):
        """某只股票跨所有已入库日期的 [date, 涨幅] 序列（升序），供扩展做连跌/趋势判断。
        返回 DataFrame 或 None（无数据）。"""
        rows = self.conn.execute(
            'SELECT date, json FROM daily WHERE code=? ORDER BY date', (code,)).fetchall()
        if not rows:
            return None
        data = []
        for date, js in rows:
            rec = json.loads(js)
            zf = rec.get('涨幅')
            try:
                zf = float(zf) if zf is not None else float('nan')
            except Exception:
                zf = float('nan')
            data.append({'date': date, '涨幅': zf})
        return pd.DataFrame(data).tail(limit).reset_index(drop=True)

    # ---------------- 预测记录 ----------------
    def save_predictions(self, run_date, target_date, list_type, df,
                         prob_col, score_col, detail_keys=None):
        """落库一次预测清单。
        detail_keys=None（默认）：沿用原五策略/尾盘的固定因子键集合；
        detail_keys=list：扩展能力显式指定要存入 detail 的因子列（供单因子 IC 分析）。"""
        cur = self.conn.cursor()
        if detail_keys is None:
            prefix_keys = ('S', 'M', 'R', 'P')
            fixed = ('综合分', '涨4评分', '连板潜力', '尾盘评分', '类型', '今日涨停',
                     '封板分钟', '强势', '资金', '量能', '位置')
            def _keep(k):
                return str(k).startswith(prefix_keys) or k in fixed
        else:
            _keep = lambda k: k in detail_keys
        for i, r in df.iterrows():
            code = str(r.get('代码6', '') or r.get('代码', ''))
            code = ''.join(__import__('re').findall(r'\d{6}', code)) or code
            detail = {k: (None if pd.isna(v) else (v.item() if hasattr(v, 'item') else v))
                      for k, v in r.items() if _keep(k)}
            cur.execute('INSERT OR REPLACE INTO predictions VALUES(?,?,?,?,?,?,?,?,?)',
                        (run_date, target_date, list_type, int(i) + 1, code,
                         str(r.get('名称', '')), float(r.get(prob_col, 0) or 0),
                         float(r.get(score_col, 0) or 0),
                         json.dumps(detail, ensure_ascii=False, default=str)))
        self.conn.commit()

    def get_predictions(self, target_date=None):
        sql = 'SELECT * FROM predictions'
        args = ()
        if target_date:
            sql += ' WHERE target_date=?'
            args = (target_date,)
        df = pd.read_sql(sql, self.conn, params=args)
        return df

    def pending_backtest_dates(self):
        """有预测但目标日已有收盘数据、可立即回测的 target_date 列表"""
        rows = self.conn.execute(
            '''SELECT DISTINCT p.target_date FROM predictions p
               WHERE EXISTS(SELECT 1 FROM daily d WHERE d.date=p.target_date)'''
        ).fetchall()
        return [r[0] for r in rows]

    def close(self):
        self.conn.close()