# -*- coding: utf-8 -*-
"""
自上而下选股方法论引擎（topdown）
================================
指导思想（用户确立的选股原则）：
    用上证指数择时 → 行业板块找主线 → 主线龙头找个股

把这一思想拆成三层、每一层都可量化、可被排序/预测复用：

  L1 择时（Timing）
      以上证指数（sh000001）日线判断市道，给出「仓位系数(0~1)」与「市道标签」。
      无法取到指数时，回退到自选池「市场宽度」代理（上涨比例/涨幅中位数/涨停数）。

  L2 板块主线（Sector rotation）
      按「所属行业」聚合，计算板块强度 = 动量 + 涨停浓度 + 主力净流入。
      强度排序靠前的板块即「主线」，是资金当下聚集的方向。

  L3 龙头个股（Leaders）
      仅在主线板块内挑选领涨个股：板块排名权重 + 个股动量 + 主力净量
      + 流通市值（龙头偏大） + 封板质量（若涨停）。非主线个股强降权。

  L4 综合（Combine）
      综合得分 = (0.6·龙头评分 + 0.4·板块强度) · 择时系数 · 非主线门槛
      —— 真正体现「择时定仓位、板块定方向、龙头定标的」。

本模块不依赖 GUI，可独立调用；扩展层 extensions/topdown_pick.py 负责接入框架。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================== L1 择时
def fetch_sh_index(ak=None, symbol: str = "sh000001", lookback: int = 130):
    """获取上证指数日线，返回 DataFrame[date, close]（close 为 float）。
    取数失败或数据不足返回 None。"""
    if ak is None:
        try:
            import akshare as ak  # noqa
        except Exception:
            return None
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or len(df) == 0:
            return None
        df = df[["date", "close"]].copy()
        df["date"] = df["date"].astype(str)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().tail(lookback).reset_index(drop=True)
        if len(df) < 21:
            return None
        return df
    except Exception:
        return None


def _regime_from_score(score: float):
    """分数(0~100) → (市道标签, 仓位系数)。"""
    if score >= 70:
        return "进攻", 1.0
    if score >= 55:
        return "偏多", 0.8
    if score >= 40:
        return "中性", 0.5
    if score >= 25:
        return "偏空", 0.3
    return "防御", 0.15


def timing_from_index(idx: pd.DataFrame):
    """仅凭上证指数判断市道。idx 来自 fetch_sh_index。"""
    close = idx["close"].astype(float).values
    n = len(close)
    ma20 = pd.Series(close).rolling(20).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values if n >= 60 else ma20
    ret5 = close[-1] / close[-6] - 1 if n >= 6 else 0.0
    ret20 = close[-1] / close[-21] - 1 if n >= 21 else 0.0
    pos_ma20 = close[-1] / ma20[-1] - 1
    pos_ma60 = (close[-1] / ma60[-1] - 1) if n >= 60 and ma60[-1] == ma60[-1] and not np.isnan(ma60[-1]) else pos_ma20

    score = 50.0
    if close[-1] > ma20[-1] and ma20[-1] > ma60[-1]:
        score += 25          # 多头排列
    if ret20 > 0:
        score += 15
    if ret5 > 0:
        score += 10
    if close[-1] < ma20[-1]:
        score -= 20
    if n >= 60 and ma20[-1] < ma60[-1]:
        score -= 15          # 空头排列
    if ret5 < 0:
        score -= 10
    score = float(np.clip(score, 0, 100))

    regime, coeff = _regime_from_score(score)
    detail = (f"上证收{close[-1]:.0f}：站上20日线{'是' if close[-1]>ma20[-1] else '否'}，"
              f"近5日{ret5*100:+.1f}%、近20日{ret20*100:+.1f}%，"
              f"{'多头' if close[-1]>ma20[-1] and ma20[-1]>ma60[-1] else '非多头'}排列")
    return {
        "method": "index", "regime": regime, "score": round(score, 1),
        "coeff": coeff, "detail": detail,
        "index_close": round(float(close[-1]), 2),
        "index_ret5": round(ret5 * 100, 2), "index_ret20": round(ret20 * 100, 2),
    }


def timing_from_breadth(u: pd.DataFrame):
    """无指数时的回退：用自选池「市场宽度」代理择时。"""
    up = (u["涨幅"] > 0).mean() if "涨幅" in u else 0.5
    med = u["涨幅"].median() if "涨幅" in u else 0.0
    lim = int(u["今日涨停"].sum()) if "今日涨停" in u else 0
    ret20_med = u["20日涨幅"].median() if "20日涨幅" in u else 0.0

    score = 50.0
    if up > 0.55:
        score += 20
    if med > 0.3:
        score += 15
    if lim > 0:
        score += 10
    if ret20_med > 0:
        score += 10
    if up < 0.40:
        score -= 25
    if med < 0:
        score -= 15
    score = float(np.clip(score, 0, 100))

    regime, coeff = _regime_from_score(score)
    detail = (f"池内宽度代理：上涨比例{up*100:.0f}%、涨幅中位{med:+.2f}%、"
              f"涨停{lim}只、20日涨幅中位{ret20_med:+.1f}%")
    return {
        "method": "breadth", "regime": regime, "score": round(score, 1),
        "coeff": coeff, "detail": detail,
        "index_close": None, "index_ret5": None, "index_ret20": None,
    }


def market_timing(u: pd.DataFrame, ak=None, force_method: str = "auto"):
    """统一择时入口。force_method ∈ {auto,index,breadth}。"""
    if force_method == "breadth":
        return timing_from_breadth(u)
    if force_method == "index":
        idx = fetch_sh_index(ak)
        return timing_from_index(idx) if idx is not None else timing_from_breadth(u)
    # auto：优先真指数
    idx = fetch_sh_index(ak)
    if idx is not None:
        return timing_from_index(idx)
    return timing_from_breadth(u)


# ============================================================== L2 板块主线
def sector_strength(u: pd.DataFrame, min_count: int = 3):
    """按所属行业聚合，计算板块强度（0~100）。返回按强度降序的 DataFrame。"""
    if "所属行业" not in u or u["所属行业"].isna().all():
        return pd.DataFrame()
    g = u.groupby("所属行业")
    rows = []
    for name, sub in g:
        n = len(sub)
        if n < min_count:
            continue
        rec = {
            "所属行业": name, "样本数": n,
            "平均涨幅": sub["涨幅"].mean() if "涨幅" in sub else np.nan,
            "平均5日": sub["5日涨幅"].mean() if "5日涨幅" in sub else np.nan,
            "平均10日": sub["10日涨幅"].mean() if "10日涨幅" in sub else np.nan,
            "平均20日": sub["20日涨幅"].mean() if "20日涨幅" in sub else np.nan,
            "涨停数": int(sub["今日涨停"].sum()) if "今日涨停" in sub else 0,
            "主力净量均值": sub["主力净量"].mean() if "主力净量" in sub else np.nan,
        }
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    s = pd.DataFrame(rows)

    # 涨停浓度（占板块样本比例）
    s["涨停浓度"] = (s["涨停数"] / s["样本数"]).fillna(0)

    def _z(col):
        v = s[col].astype(float)
        rng = v.max() - v.min()
        return (v - v.min()) / (rng + 1e-9)

    s["强度分"] = (
        0.35 * _z("平均涨幅") +
        0.25 * _z("平均20日") +
        0.20 * _z("涨停浓度") +
        0.20 * _z("主力净量均值")
    ) * 100
    s = s.sort_values("强度分", ascending=False).reset_index(drop=True)
    s.insert(0, "板块排名", s.index + 1)
    return s


def main_lines(sector_df: pd.DataFrame, top_n: int = 5):
    """返回 (主线板块集合, 板块→强度映射, 板块→排名映射)。"""
    if sector_df.empty:
        return set(), {}, {}
    top = sector_df.head(top_n)
    sset = set(top["所属行业"])
    strength = dict(zip(sector_df["所属行业"], sector_df["强度分"]))
    rank = dict(zip(sector_df["所属行业"], sector_df["板块排名"]))
    return sset, strength, rank


# ============================================================== L3 龙头个股
def _safe_z(series: pd.Series):
    v = pd.to_numeric(series, errors="coerce")
    rng = v.max() - v.min()
    return (v - v.min()) / (rng + 1e-9)


def leader_score(u: pd.DataFrame, sector_df: pd.DataFrame,
                 main_set, main_rank, main_strength):
    """板块内龙头打分（0~100）。非主线个股强降权。"""
    if sector_df.empty:
        return pd.DataFrame()
    N = len(sector_df)

    # 个股动量 z（全市场）
    mom = (0.3 * _safe_z(u["5日涨幅"]) if "5日涨幅" in u else 0
           + 0.3 * _safe_z(u["10日涨幅"]) if "10日涨幅" in u else 0
           + 0.4 * _safe_z(u["20日涨幅"]) if "20日涨幅" in u else 0)
    # 主力净量 z（封顶 ±3）
    if "主力净量" in u:
        mf = (pd.to_numeric(u["主力净量"], errors="coerce").clip(-3, 3) + 3) / 6
    else:
        mf = pd.Series(0.5, index=u.index)
    # 流通市值 z（龙头偏大）
    if "流通市值" in u:
        cap = _safe_z(np.log10(pd.to_numeric(u["流通市值"], errors="coerce").clip(lower=1e6)))
    else:
        cap = pd.Series(0.5, index=u.index)
    # 封板质量（仅涨停股）
    if "今日涨停" in u and "封流比" in u:
        seal = pd.to_numeric(u["封流比"], errors="coerce").fillna(0)
        seal = (seal / (seal.max() + 1e-9)).where(u["今日涨停"].fillna(False), 0.5)
    else:
        seal = pd.Series(0.5, index=u.index)

    leader_raw = (0.40 * mom + 0.20 * mf + 0.15 * cap + 0.15 * seal + 0.10 * 0.5)
    leader_raw = leader_raw * 100

    sec_rank_norm = u["所属行业"].map(lambda x: (N - main_rank.get(x, N)) / max(N - 1, 1))
    sec_str_norm = u["所属行业"].map(lambda x: (main_strength.get(x, 0)) / 100.0)
    in_main = u["所属行业"].isin(main_set)

    score = pd.Series(np.where(
        in_main.values,
        0.55 * leader_raw + 0.45 * sec_rank_norm * 100,
        0.55 * leader_raw * 0.6 + 0.45 * sec_str_norm * 100,
    ), index=u.index)
    out = u[["代码6", "名称", "所属行业"]].copy()
    out["龙头评分"] = score.round(1)
    out["是否在主线"] = in_main
    out["板块排名"] = u["所属行业"].map(main_rank).fillna(0).astype(int)
    out["板块强度"] = (u["所属行业"].map(main_strength).fillna(0)).round(1)
    return out


# ============================================================== L4 综合
def topdown_rank(u: pd.DataFrame, ak=None, top_n_sectors: int = 5,
                 min_sector_count: int = 3, force_method: str = "auto",
                 weight_sector: int = 40, weight_leader: int = 60,
                 gate_non_main: int = 70):
    """端到端：择时 → 板块 → 龙头 → 综合。
    返回 (timing_dict, sector_df, ranked_df)。
    ranked_df 含：代码,名称,所属行业,综合得分,择时系数,市道,是否在主线,
                  板块强度,龙头评分,5日涨幅,10日涨幅,20日涨幅,主力净量,流通市值,现价,市盈,市净率,排名
    """
    timing = market_timing(u, ak=ak, force_method=force_method)
    coeff = timing["coeff"]

    sector_df = sector_strength(u, min_count=min_sector_count)
    main_set, main_strength, main_rank = main_lines(sector_df, top_n=top_n_sectors)

    ld = leader_score(u, sector_df, main_set, main_rank, main_strength)
    if ld.empty:
        return timing, sector_df, pd.DataFrame()

    w_s, w_l = weight_sector / 100.0, weight_leader / 100.0
    comb = (w_l * ld["龙头评分"] + w_s * ld["板块强度"]) * coeff
    comb = comb * np.where(ld["是否在主线"].values, 1.0, 1 - gate_non_main / 100.0)
    ld["综合得分"] = comb.round(2)
    ld["择时系数"] = coeff
    ld["市道"] = timing["regime"]

    extra = ["5日涨幅", "10日涨幅", "20日涨幅", "主力净量", "流通市值", "现价", "市盈", "市净率"]
    for c in extra:
        if c in u:
            ld[c] = u[c].values

    cols = ["代码6", "名称", "所属行业", "综合得分", "择时系数", "市道", "是否在主线",
            "板块强度", "龙头评分", "5日涨幅", "10日涨幅", "20日涨幅", "主力净量",
            "流通市值", "现价", "市盈", "市净率"]
    cols = [c for c in cols if c in ld.columns]
    ranked = ld[cols].sort_values("综合得分", ascending=False).reset_index(drop=True)
    ranked.insert(0, "排名", ranked.index + 1)
    return timing, sector_df, ranked
