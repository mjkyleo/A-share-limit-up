# -*- coding: utf-8 -*-
"""
扩展框架入口（extensions/__init__）
==================================
导入本包即自动扫描 extensions/ 下的 Extension 子类，构建 REGISTRY，
并把各扩展声明的回测信息导出给 core/backtest.py 合并使用。

app.py 只需：
    from extensions import REGISTRY, enabled_extensions, get_source
即可拿到所有能力，无需为新增能力改动 app.py。
"""
import importlib
import pkgutil

from . import base as _base
from .datasources import load_data_sources, get_source

REGISTRY = {}          # key -> Extension 实例
EXT_LIST_TYPES = []    # 供 backtest.LIST_TYPES 合并
EXT_SCORING = {}       # 供 backtest.DEFAULT_SCORING 合并
EXT_HIT = {}           # 供 backtest.HIT_THRESHOLD 合并
EXT_IC_KEYS = []       # 供 backtest.STRATEGY_KEYS 合并（单因子 IC）


def load_extensions():
    if REGISTRY:
        return REGISTRY
    for mod in pkgutil.iter_modules(__path__):
        if mod.name in ('__init__', 'base', 'datasources'):
            continue
        m = importlib.import_module(f'{__name__}.{mod.name}')
        for obj in vars(m).values():
            if (isinstance(obj, type) and issubclass(obj, _base.Extension)
                    and obj is not _base.Extension and obj.key):
                REGISTRY[obj.key] = obj()
    _build_backtest_exports()
    return REGISTRY


def _build_backtest_exports():
    for ext in REGISTRY.values():
        if not ext.list_type:
            continue
        EXT_LIST_TYPES.append(ext.list_type)
        if ext.scoring:
            EXT_SCORING[ext.list_type] = ext.scoring
        if ext.hit_threshold is not None:
            EXT_HIT[ext.list_type] = ext.hit_threshold
        for k in ext.detail_keys:
            if k not in EXT_IC_KEYS:
                EXT_IC_KEYS.append(k)


def enabled_extensions(cfg):
    """返回 {key: Extension}，按 config.extensions.<key>.enabled 过滤（默认启用）。"""
    ext_cfg = (cfg or {}).get('extensions', {})
    out = {}
    for key, ext in REGISTRY.items():
        ec = ext_cfg.get(key, {})
        if ec.get('enabled', True):
            out[key] = ext
    return out


# 导入即发现（backtest 在 import 时即可拿到合并后的清单类型）
load_extensions()
