# -*- coding: utf-8 -*-
"""P0-7 单元测试：board_of / is_limit_up（涨停判定唯一入口）。

验收点：
  - 主板(60/000/001/002/003) ≥9.5% 判涨停
  - 创业板(300/301)/科创板(688) ≥19.5% 判涨停
  - 北交所(8/4/92/920) ≥29.5% 判涨停
  - 禁止在别处硬编码 9.5：创业板 9.8% 不应判涨停
  - 坏输入（None/NaN/非数字）安全返回 False
"""
import pytest

from core import strategies

TH = strategies.DEFAULT_CONFIG['thresholds']


@pytest.mark.parametrize('code,exp', [
    # 主板
    ('600000', 'main'), ('000001', 'main'), ('001979', 'main'),
    ('002001', 'main'), ('003000', 'main'),
    # 创业板
    ('300001', 'cyb'), ('301000', 'cyb'),
    # 科创板
    ('688001', 'star'),
    # 北交所
    ('830799', 'bse'), ('430047', 'bse'), ('920002', 'bse'),
    # 带交易所后缀 / 空格 / 前缀
    ('SH600519', 'main'), (' SZ000001 ', 'main'), ('BJ830799', 'bse'),
    # 无法识别
    ('xyz', 'other'), ('', 'other'),
])
def test_board_of(code, exp):
    assert strategies.board_of(code) == exp


def test_is_limit_up_main():
    assert strategies.is_limit_up('600000', 10.0, TH) is True
    assert strategies.is_limit_up('600000', 9.5, TH) is True
    assert strategies.is_limit_up('600000', 9.4, TH) is False
    assert strategies.is_limit_up('600000', -1.0, TH) is False


def test_is_limit_up_cyb_star():
    # 创业板 / 科创板 19.5% 线
    assert strategies.is_limit_up('300001', 19.6, TH) is True
    assert strategies.is_limit_up('300001', 19.5, TH) is True
    assert strategies.is_limit_up('300001', 19.4, TH) is False
    assert strategies.is_limit_up('688001', 20.0, TH) is True
    assert strategies.is_limit_up('688001', 19.0, TH) is False


def test_is_limit_up_bse():
    assert strategies.is_limit_up('920002', 29.5, TH) is True
    assert strategies.is_limit_up('920002', 29.4, TH) is False


def test_is_limit_up_bad_input():
    assert strategies.is_limit_up('600000', None, TH) is False
    assert strategies.is_limit_up('600000', float('nan'), TH) is False
    assert strategies.is_limit_up('600000', 'abc', TH) is False


def test_no_hardcoded_ninefive_for_cyb():
    # 创业板必须用 19.5 而非 9.5：9.8% 的创业板不应判涨停
    assert strategies.is_limit_up('300001', 9.8, TH) is False
    assert strategies.is_limit_up('300001', 20.0, TH) is True
