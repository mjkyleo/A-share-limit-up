# -*- coding: utf-8 -*-
"""
A股交易日历（内置版，零依赖）
规则：
  1. 周一~周五为交易日，周六日闭市（含调休上班的周末，股市也不开盘）
  2. 法定节假日闭市，见 HOLIDAYS 表（2024-2026，可在 config.json 的 extra_holidays 补充）
交易时段（北京时间）：
  09:15-09:25 集合竞价 | 09:30-11:30 / 13:00-15:00 连续竞价 | 15:00 收盘
"""
from datetime import date, datetime, time, timedelta

# 法定节假日闭市表（股市口径：周末一律闭市，此处只列工作日闭市日期）
HOLIDAYS = {
    # 2024
    '20240101', '20240212', '20240213', '20240214', '20240215', '20240216',
    '20240404', '20240405', '20240501', '20240502', '20240503', '20240610',
    '20240916', '20240917', '20241001', '20241002', '20241003', '20241004',
    '20241007',
    # 2025
    '20250101', '20250128', '20250129', '20250130', '20250131', '20250203',
    '20250204', '20250404', '20250501', '20250502', '20250505', '20250602',
    '20251001', '20251002', '20251003', '20251006', '20251007', '20251008',
    # 2026（国务院办公厅口径，如遇调整请在 config.json -> extra_holidays 补充）
    '20260101', '20260102', '20260216', '20260217', '20260218', '20260219',
    '20260220', '20260406', '20260501', '20260504', '20260505', '20260619',
    '20260925', '20261001', '20261002', '20261005', '20261006', '20261007',
    '20261008',
}

_EXTRA = set()  # config.json 注入的额外闭市日


def set_extra_holidays(days):
    """days: ['20261225', ...]"""
    global _EXTRA
    _EXTRA = {str(d).replace('-', '') for d in days}


def is_trading_day(d=None):
    """是否A股交易日"""
    d = d or date.today()
    if isinstance(d, datetime):
        d = d.date()
    if d.weekday() >= 5:                       # 周末一律闭市（含调休工作日）
        return False
    return d.strftime('%Y%m%d') not in (HOLIDAYS | _EXTRA)


def next_trading_day(d=None):
    """下一个交易日"""
    d = (d or date.today()) + timedelta(days=1)
    if isinstance(d, datetime):
        d = d.date()
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d=None):
    """上一个交易日"""
    d = (d or date.today()) - timedelta(days=1)
    if isinstance(d, datetime):
        d = d.date()
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days_between(d1, d2):
    """[d1, d2] 之间的交易日列表"""
    if isinstance(d1, datetime):
        d1 = d1.date()
    if isinstance(d2, datetime):
        d2 = d2.date()
    out, cur = [], d1
    while cur <= d2:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def current_session(now=None):
    """当前所处交易时段，返回 (状态码, 中文描述)
    状态码: PRE_OPEN 盘前 | CALL_AUCTION 集合竞价 | MORNING 上午盘 |
            LUNCH 午间休市 | AFTERNOON 下午盘 | CLOSED 已收盘 | OFFDAY 非交易日
    """
    now = now or datetime.now()
    if not is_trading_day(now):
        nd = next_trading_day(now)
        return 'OFFDAY', f'今日非交易日，下一交易日 {nd:%Y-%m-%d}（{"一二三四五六日"[nd.weekday()]}）'
    t = now.time()
    if t < time(9, 15):
        return 'PRE_OPEN', '盘前（9:15 集合竞价开始）'
    if t < time(9, 25):
        return 'CALL_AUCTION', '集合竞价中（9:25 出竞价结果）'
    if t < time(9, 30):
        return 'CALL_AUCTION', '竞价已结束，9:30 开盘'
    if t < time(11, 30):
        return 'MORNING', '上午交易中'
    if t < time(13, 0):
        return 'LUNCH', '午间休市'
    if t < time(15, 0):
        return 'AFTERNOON', '下午交易中'
    return 'CLOSED', f'已收盘，下一交易日 {next_trading_day(now):%Y-%m-%d}'


def describe_file_moment(d):
    """给定数据日期，返回它对应的预测目标日（下一交易日）"""
    return next_trading_day(d)