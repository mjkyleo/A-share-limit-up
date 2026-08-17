# -*- coding: utf-8 -*-
"""
数据源注册表（extensions/datasources）
====================================
自动发现 extensions/datasources/ 下的 DataSource 实现，构建 DATA_SOURCES。
框架据 Extension.datasource 取对应实例并调用 fetch() 注入数据。

以后要接实时行情（akshare / 东方财富 / 通达信实时）：
  1. 在此目录新建一个 .py，继承 DataSource，实现 fetch() 返回同口径 DataFrame；
  2. 在 config 里把某扩展的 datasource 指到它的 key。
扩展逻辑一行都不用改。
"""
import importlib
import pkgutil

from extensions.base import DataSource

DATA_SOURCES = {}


def load_data_sources():
    if DATA_SOURCES:
        return DATA_SOURCES
    for mod in pkgutil.iter_modules(__path__):
        if mod.name in ('__init__', 'base'):
            continue
        m = importlib.import_module(f'{__name__}.{mod.name}')
        for obj in vars(m).values():
            if (isinstance(obj, type) and issubclass(obj, DataSource)
                    and obj is not DataSource):
                DATA_SOURCES[obj.key] = obj()
    return DATA_SOURCES


def get_source(key):
    load_data_sources()
    return DATA_SOURCES.get(key)
