# -*- coding: utf-8 -*-
"""本地数据源：复用现有 loader 读取 data/ 文件夹里最新日期的收盘快照。"""
from extensions.base import DataSource


class LocalFolderDataSource(DataSource):
    key = 'local'
    description = '本地 data/ 文件夹（通达信导出的 xlsx/csv），复用 loader 加载最新收盘快照'

    def fetch(self, ctx):
        from core import loader
        df, _infos = loader.load_dataset(ctx.data_dir)
        if df is None or df.empty:
            return df, {'来源': '本地数据文件夹', 'date': '', '行数': 0}
        date_str = str(df['数据日期'].max())
        return df, {'来源': '本地数据文件夹', 'date': date_str, '行数': len(df)}
