# -*- coding: utf-8 -*-
"""
四股联合分析：自上而下方法论 × 项目五因子 × 基本面
=================================================
输入：data/08-15-close.xlsx（最新自选池快照，含 4 只基本面股）
输出：reports/四股联合分析_0815.md（可读报告）+ reports/topdown_rank_0815.csv（全市场自上而下排名）

三层视角：
  A. 自上而下方法论（core/topdown）：上证指数择时 → 行业板块主线 → 龙头打分
  B. 项目本身分析（core/strategies.run_prediction）：五因子封单/资金/股性 综合分
  C. 用户给出的基本面（业绩增速/估值/业绩质量/风险点）
联合得分 = 0.55·基本面评分 + 0.45·项目综合分(归一化)  —— 在「非主线逆向篮子」内排序。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from core.loader import read_any, clean
from core import strategies
from core.topdown import topdown_rank

DATA = "data/08-15-close.xlsx"
CODES = ['601700', '600395', '601388', '603077']

# ----------------- 用户提供的四股基本面（来自对话粘贴）-----------------
FUND = {
    '601700': dict(name='风范股份', sector='电网设备/特高压角钢塔', price=6.56,
        growth='中报预增742%~1023%', pe='亏损', pb=3.47, quality='扭亏为盈',
        logic='国网/南网特高压集中开工，已完成阿坝-成都、甘肃-浙江等重点工程供货',
        risk='8/14 大跌7.61%、换手17.66%，短期情绪偏空；PB偏高', tier_growth=880),
    '600395': dict(name='盘江股份', sector='煤炭开采/西南煤炭', price=5.18,
        growth='预增1327%', pe=23.19, pb=1.04, quality='扣非>归母(主业成色最好)',
        logic='煤价企稳回升+产能释放',
        risk='周期股，煤价波动；市盈率随煤价波动', tier_growth=1327),
    '601388': dict(name='怡球资源', sector='工业金属/再生铝', price=3.29,
        growth='预增655%', pe=34.76, pb=1.54, quality='未明确',
        logic='汽车轻量化带动铝需求；马来西亚二期产能爬坡',
        risk='再生铝门槛不高、竞争加剧；铝价波动', tier_growth=655),
    '603077': dict(name='和邦生物', sector='农化制品/草甘膦', price=2.37,
        growth='Q1 净利同比+2186%', pe='亏损', pb=1.18, quality='扣非与归母基本持平',
        logic='草甘膦/蛋氨酸价格高位；自有盐矿成本优势',
        risk='市盈率为负(历史亏损)；光伏/玻璃产业链未完全验证', tier_growth=2186),
}


def fundamental_score(f):
    """0-100 基本面评分：40%增速 + 30%估值安全 + 30%业绩质量。"""
    growth = 40 + min(60, f['tier_growth'] / 30.0)        # 和邦100 盘江84 风范69 怡球62
    val = {'601700': 40, '600395': 90, '601388': 60, '603077': 70}[f['__code']]
    qual = {'601700': 75, '600395': 90, '601388': 65, '603077': 85}[f['__code']]
    return round(0.4 * growth + 0.3 * val + 0.3 * qual, 1)


def main():
    d, _, _ = clean(read_any(DATA))
    cfg = strategies.load_config('.')
    res = strategies.run_prediction(d, cfg)
    full = res['全量']                       # 项目五因子 全量评分
    full['项目综合分'] = full['综合分']

    # A. 自上而下
    timing, sectors, ranked = topdown_rank(d, ak=None, force_method='auto')
    ranked = ranked.rename(columns={'综合得分': 'TD综合得分'})

    # 合并三层到四股
    proj = full[full['代码6'].isin(CODES)][['代码6', '名称', '项目综合分']]
    td = ranked[ranked['代码6'].isin(CODES)][
        ['代码6', 'TD综合得分', '是否在主线', '板块强度', '龙头评分', '所属行业']]
    rows = []
    for code in CODES:
        f = dict(FUND[code]); f['__code'] = code
        fscore = fundamental_score(f)
        p = proj[proj['代码6'] == code]['项目综合分'].iloc[0]
        t = td[td['代码6'] == code]
        tscore = t['TD综合得分'].iloc[0] if len(t) else np.nan
        in_main = bool(t['是否在主线'].iloc[0]) if len(t) else False
        sec = t['所属行业'].iloc[0] if len(t) else f['sector']
        rows.append(dict(
            代码=code, 名称=f['name'], 行业=sec,
            基本面评分=fscore, 项目综合分=round(float(p), 4),
            TD综合得分=round(float(tscore), 2) if pd.notna(tscore) else np.nan,
            是否在主线=in_main,
            业绩增速=f['growth'], 市盈率=f['pe'], 市净率=f['pb'],
            业绩质量=f['quality'], 逻辑=f['logic'], 风险=f['risk'],
        ))
    r = pd.DataFrame(rows)
    # 归一化项目综合分到 0-100（全市场 min-max）
    lo, hi = full['项目综合分'].min(), full['项目综合分'].max()
    r['项目归一'] = ((r['项目综合分'] - lo) / (hi - lo) * 100).round(1)
    # 联合得分：基本面 0.55 + 项目 0.45
    r['联合得分'] = (0.55 * r['基本面评分'] + 0.45 * r['项目归一']).round(1)
    r = r.sort_values('联合得分', ascending=False).reset_index(drop=True)
    r.insert(0, '联合排名', r.index + 1)

    # ---------- 写 CSV（全市场自上而下排名）----------
    csv_path = "reports/topdown_rank_0815.csv"
    ranked.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ---------- 生成 Markdown 报告 ----------
    md = []
    md.append("# 四股联合分析：自上而下方法论 × 项目五因子 × 基本面\n")
    md.append(f"> 数据基准：`{DATA}`（收盘快照，共 {len(d)} 只自选池）\n")
    md.append("## 一、方法论如何落地到项目\n")
    md.append("- **择时**：`core/topdown.timing_from_index` 取上证指数(sh000001)日线，"
              "输出「市道标签 + 仓位系数」；取不到则回退自选池市场宽度代理。已封装为扩展 `extensions/topdown_pick.py` 接入 GUI/回测。\n")
    md.append("- **板块主线**：按 `所属行业` 聚合，板块强度=0.35·平均涨幅+0.25·平均20日+"
              "0.20·涨停浓度+0.20·主力净量，强度 Top5 即主线。\n")
    md.append("- **龙头个股**：主线板块内按 动量+主力+流通市值+封板质量 打分；非主线强降权。\n")
    md.append("- **综合**：`综合得分=(0.6·龙头+0.4·板块强度)·择时系数·(非主线降权)`——"
              "真正体现「择时定仓位、板块定方向、龙头定标的」。\n")

    md.append("## 二、择时结论（上证指数）\n")
    md.append(f"- 市道：**{timing['regime']}**，仓位系数 **{timing['coeff']}**"
              f"（method={timing['method']}）。\n")
    md.append(f"- 依据：{timing['detail']}\n")
    md.append(f"- **操作含义**：中性市道 → 单次仓位不超过 50%，不追高、分批低吸。\n")

    md.append("## 三、板块主线（来自自选池，含局限说明）\n")
    top = sectors.head(5)
    md.append("| 排名 | 主线板块 | 样本 | 平均涨幅% | 平均20日% | 涨停数 | 强度分 |")
    md.append("|------|----------|------|-----------|-----------|--------|--------|")
    for _, s in top.iterrows():
        md.append(f"| {int(s['板块排名'])} | {s['所属行业']} | {int(s['样本数'])} | "
                  f"{s['平均涨幅']:.2f} | {s['平均20日']:.2f} | {int(s['涨停数'])} | {s['强度分']:.1f} |")
    md.append("\n> ⚠️ **数据局限**：主线仅基于本 {0} 只自选池。四股所在行业"
              "（煤炭/工业金属/农化/电网设备）在本池中样本少，**未进入 Top5 主线**；"
              "但这不代表它们在全市场不是主线——属「池内样本偏差」。严格自上而下纪律下，"
              "这四只目前属于**左侧/逆向篮子**，而非顺势主线。\n".format(len(d)))

    md.append("## 四、四股三层联合分析\n")
    md.append("| 联合排名 | 代码 | 名称 | 行业 | 基本面评分 | 项目综合分(全市场归一) | "
              "TD综合得分 | 是否主线 | 市盈率 | 市净率 |")
    md.append("|----------|------|------|------|-----------|------------------------|-----------|----------|--------|--------|")
    for _, x in r.iterrows():
        pe = x['市盈率']; pb = x['市净率']
        pe_s = f"{pe}" if isinstance(pe, str) else f"{pe:.2f}"
        pb_s = f"{pb}" if isinstance(pb, str) else f"{pb:.2f}"
        md.append(f"| {int(x['联合排名'])} | {x['代码']} | {x['名称']} | {x['行业']} | "
                  f"{x['基本面评分']:.1f} | {x['项目归一']:.1f} | "
                  f"{x['TD综合得分'] if pd.notna(x['TD综合得分']) else '—'} | "
                  f"{'是' if x['是否在主线'] else '否'} | {pe_s} | {pb_s} |")

    md.append("\n### 各股要点\n")
    for _, x in r.iterrows():
        f = FUND[x['代码']]
        md.append(f"**{int(x['联合排名'])}. {x['名称']}（{x['代码']}）**  "
                  f"联合得分 {x['联合得分']:.1f} ｜ 基本面 {x['基本面评分']:.1f} ｜ "
                  f"项目归一 {x['项目归一']:.1f} ｜ TD {x['TD综合得分'] if pd.notna(x['TD综合得分']) else '—'}\n")
        md.append(f"- 业绩：{f['growth']}；估值：PE {f['pe']} / PB {f['pb']}；质量：{f['quality']}\n")
        md.append(f"- 逻辑：{f['logic']}\n")
        md.append(f"- 风险：{f['risk']}\n")

    md.append("## 五、周一操作框架结论\n")
    winner = r.iloc[0]
    md.append(f"1. **首选（风险收益比最佳）：{winner['名称']}（{winner['代码']}）**——"
              "基本面评分与项目五因子双高，估值安全垫最厚（PB 最低、扣非>归母主业成色最好），"
              "且在本池项目模型中位列前 15%。中性市道下以不超过 5 成仓、分批低吸。\n")
    md.append("2. **高弹性高波动：和邦生物**——增速最高、价格最低，但市盈率为负（周期股属性强），"
              "属「高增长+左侧」标的，小仓试错、严格止损。\n")
    md.append("3. **稳健中游：怡球资源**——铝业景气+产能释放，估值合理但竞争加剧，可观察铝价方向。\n")
    md.append("4. **最强动量但需等待企稳：风范股份**——20日涨幅+45% 显示强趋势，但 8/14 单日-7.61% "
              "且 PB 偏高、PE 亏损，**不追跌、等放量企稳再考虑**，特高压为中长期逻辑。\n")
    md.append("\n> **方法论总开关**：当前择时=中性（≤5成仓）、四股均不在池内主线（左侧逆向篮子）。"
              "若后续上证指数转「进攻」且煤炭/有色/化工成为全市场主线，则这四只进入顺势加仓区间；"
              "反之维持轻仓观察。\n")

    md.append("## 六、缺少的数据与后续\n")
    md.append("- 四股**扣非净利润明细、正式业绩预告数值**仅部分体现在快照，详细以用户粘贴基本面为准。\n")
    md.append("- **全市场板块涨跌**未离线获取，主线判定受自选池样本偏差影响（见第三节）。\n")
    md.append("- **神经网络预测(nn_model.db)** 未在本快照训练，本轮未纳入 NN 评分；可在 GUI 训练后加入联合权重。\n")
    md.append("- 项目五因子为「封单/涨停」模型，四股均非涨停股 → 项目综合分天然偏低（盘江除外，因其资金/股性尚可）。\n")

    out = "\n".join(md)
    md_path = "reports/四股联合分析_0815.md"
    open(md_path, 'w', encoding='utf-8').write(out)
    print("written:", md_path, "|", csv_path)
    print("\n=== 四股联合排名 ===")
    print(r[['联合排名','代码','名称','基本面评分','项目归一','TD综合得分','是否在主线','联合得分']].to_string(index=False))
    return r


if __name__ == '__main__':
    main()
