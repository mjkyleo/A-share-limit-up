# -*- coding: utf-8 -*-
"""端到端验证脚本（非交付物，仅测试用）：
1. 构造通达信伪xlsx（GBK制表符）模拟数据：08-13 收盘/开盘/盘中两批 + 08-14 收盘
2. loader：识别 close/open/intraday(批次) + load_latest_intraday 取最新批
3. strategies：常规三清单 + 尾盘选股清单
4. db：入库（含 close 优先级防污染）
5. backtest：分清单评分 / 滚动IC / 滚动得分 / 调参建议
"""
import os
import shutil
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from core import backtest as bt
from core import loader, strategies
from core.db import DB

TMP = tempfile.mkdtemp(prefix='limitup_test_')
DATA = os.path.join(TMP, 'data')
os.makedirs(DATA)

HEADER = ['代码', '名称', '涨幅', '开盘涨幅', '量比', '换手', '振幅', '竞价换手',
          '涨停封流比[20260813]', '涨停封成比[20260813]', '首次涨停时间[20260813]',
          '涨停开板次数[20260813]', '流通市值', '主力净量', '资金流向',
          '5日涨幅', '20日涨幅', '市盈(动)', '总手']

# (代码, 名称, 涨幅, 开盘涨幅, 量比, 换手, 振幅, 竞价换手, 封流比, 封成比, 首封, 开板, 市值亿, 主力净量, 资金流向万, 5日, 20日, 市盈, 总手)
CLOSE_0813 = [
    ('SH600111', '强封A', '+10.02%', '+2.1%', '2.5', '6.5', '8.2', '0.8', '5.2', '12.0', '09:31:05', '0', '45亿', '1.8', '5200万', '8', '15', '25', '35万'),
    ('SH600222', '强封B', '+10.01%', '+1.5%', '1.8', '2.1', '3.1', '0.5', '6.8', '25.0', '09:30:30', '0', '28亿', '2.2', '8100万', '12', '22', '31', '20万'),
    ('SZ300333', '烂板C', '+9.98%', '+4.2%', '5.5', '18.5', '12.5', '1.2', '1.2', '2.5', '14:20:11', '4', '95亿', '-0.5', '-2300万', '25', '45', '亏损', '80万'),
    ('SZ000444', '中阳D', '+5.5%', '+1.0%', '2.2', '8.5', '6.5', '0.6', '--', '--', '--', '--', '60亿', '0.8', '1500万', '6', '10', '18', '50万'),
    ('SZ000555', '小阳E', '+2.5%', '+0.5%', '1.2', '4.5', '4.0', '0.4', '--', '--', '--', '--', '35亿', '0.3', '500万', '3', '8', '22', '30万'),
    ('SH600666', '微跌F', '-1.5%', '-0.3%', '0.8', '3.2', '3.0', '0.3', '--', '--', '--', '--', '120亿', '-0.2', '-800万', '-2', '5', '15', '40万'),
    ('SZ000777', '大阳G', '+7.8%', '+2.8%', '3.2', '12.0', '9.5', '0.9', '--', '--', '--', '--', '55亿', '1.5', '3200万', '10', '18', '35', '65万'),
    ('SH600888', '高位H', '+10.03%', '+5.5%', '4.5', '15.0', '10.2', '1.5', '3.5', '5.0', '10:15:22', '2', '40亿', '0.5', '800万', '35', '75', '48', '55万'),
]

OPEN_0813 = [  # 开盘快照：涨幅≈开盘涨幅，换手≈竞价换手
    (c, n, og, og, '1.0', jj, '0.5', jj, '--', '--', '--', '--', mv, zl, zj, d5, d20, pe, '1万')
    for (c, n, _, og, _, _, _, jj, _, _, _, _, mv, zl, zj, d5, d20, pe, _) in CLOSE_0813
]

def _pct(x, ratio):
    v = float(str(x).replace('%', '').replace('+', '')) * ratio
    return f'{v:+.1f}%'


INTRA1_0813 = [  # 盘中第1批 10:30（涨幅约为收盘一半，换手约四成）
    (c, n, _pct(zf, 0.5), og, lb, f'{float(hs) * 0.4:.1f}', zf2, jj, *rest)
    for (c, n, zf, og, lb, hs, zf2, jj, *rest) in CLOSE_0813
]

INTRA2_0813 = [  # 盘中第2批 14:30 尾盘（涨幅接近收盘但略低；强封A/B已封板）
    ('SH600111', '强封A', '+10.02%', '+2.1%', '2.3', '6.0', '8.0', '0.8', '5.2', '12.0', '09:31:05', '0', '45亿', '1.8', '5000万', '8', '15', '25', '32万'),
    ('SH600222', '强封B', '+10.01%', '+1.5%', '1.7', '2.0', '3.0', '0.5', '6.8', '25.0', '09:30:30', '0', '28亿', '2.2', '8000万', '12', '22', '31', '18万'),
    ('SZ300333', '烂板C', '+8.5%', '+4.2%', '4.8', '16.0', '11.5', '1.2', '--', '--', '--', '--', '95亿', '-0.5', '-2100万', '25', '45', '亏损', '72万'),
    ('SZ000444', '中阳D', '+6.2%', '+1.0%', '2.5', '9.0', '7.0', '0.6', '--', '--', '--', '--', '60亿', '1.2', '2100万', '6', '10', '18', '55万'),
    ('SZ000555', '小阳E', '+1.8%', '+0.5%', '1.1', '4.0', '3.5', '0.4', '--', '--', '--', '--', '35亿', '0.2', '300万', '3', '8', '22', '28万'),
    ('SH600666', '微跌F', '-0.8%', '-0.3%', '0.7', '3.0', '2.8', '0.3', '--', '--', '--', '--', '120亿', '-0.3', '-900万', '-2', '5', '15', '38万'),
    ('SZ000777', '大阳G', '+8.5%', '+2.8%', '3.5', '13.0', '10.0', '0.9', '--', '--', '--', '--', '55亿', '1.8', '3800万', '10', '18', '35', '70万'),
    ('SH600888', '高位H', '+10.03%', '+5.5%', '4.2', '14.0', '10.0', '1.5', '3.5', '5.0', '10:15:22', '2', '40亿', '0.5', '700万', '35', '75', '48', '50万'),
]

CLOSE_0814 = [  # 次日实际收盘（用于回测验证）
    ('SH600111', '强封A', '+10.00%', '+3.0%', '2.0', '5.5', '7.5', '0.7', '4.5', '9.0', '09:35:10', '1', '45亿', '1.2', '3000万', '18', '25', '25', '30万'),
    ('SH600222', '强封B', '+9.99%', '+2.2%', '1.5', '1.8', '2.5', '0.4', '5.5', '18.0', '09:31:00', '0', '28亿', '1.5', '5000万', '22', '32', '31', '15万'),
    ('SZ300333', '烂板C', '-6.5%', '-2.0%', '3.5', '15.0', '10.5', '1.0', '--', '--', '--', '--', '95亿', '-1.5', '-5000万', '17', '36', '亏损', '70万'),
    ('SZ000444', '中阳D', '+4.5%', '+1.2%', '2.0', '7.5', '5.5', '0.5', '--', '--', '--', '--', '60亿', '0.9', '1800万', '10', '14', '18', '45万'),
    ('SZ000555', '小阳E', '+1.0%', '+0.2%', '0.9', '3.5', '2.5', '0.3', '--', '--', '--', '--', '35亿', '0.1', '100万', '4', '9', '22', '25万'),
    ('SH600666', '微跌F', '-3.5%', '-1.0%', '1.2', '4.5', '4.5', '0.4', '--', '--', '--', '--', '120亿', '-0.8', '-2000万', '-5', '1', '15', '45万'),
    ('SZ000777', '大阳G', '+5.2%', '+1.5%', '2.8', '10.0', '8.0', '0.8', '--', '--', '--', '--', '55亿', '1.0', '2500万', '15', '23', '35', '60万'),
    ('SH600888', '高位H', '-4.5%', '-3.5%', '5.0', '18.0', '12.0', '1.6', '--', '--', '--', '--', '40亿', '-1.2', '-3500万', '30', '68', '48', '75万'),
]


def write_tdx(path, rows, header=None):
    """写通达信伪xlsx：GBK + 制表符"""
    with open(path, 'w', encoding='gbk', newline='') as f:
        f.write('\t'.join(header or HEADER) + '\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')


ok = []


def check(name, cond, extra=''):
    ok.append((name, bool(cond)))
    print(f"  {'✅' if cond else '❌'} {name} {extra}")


print('== 1. 写模拟数据 ==')
write_tdx(os.path.join(DATA, '08-13-close.xlsx'), CLOSE_0813)
write_tdx(os.path.join(DATA, '08-13-open.xlsx'), OPEN_0813)
write_tdx(os.path.join(DATA, '08-13-01.xlsx'), INTRA1_0813)
write_tdx(os.path.join(DATA, '08-13-02.xlsx'), INTRA2_0813)
print('  已写 4 个文件')

print('== 2. loader 识别 ==')
infos = loader.scan_folder(DATA)
kinds = {i['文件']: (i.get('kind'), i.get('batch')) for i in infos}
print('  ', kinds)
check('close 识别', kinds.get('08-13-close.xlsx', (None,))[0] == 'close')
check('open 识别', kinds.get('08-13-open.xlsx', (None,))[0] == 'open')
check('intraday-01 识别', kinds.get('08-13-01.xlsx') == ('intraday', 1))
check('intraday-02 识别', kinds.get('08-13-02.xlsx') == ('intraday', 2))

print('== 3. load_latest_intraday 取最新批 ==')
intra, info = loader.load_latest_intraday(DATA)
check('取到第2批', info and info['文件'] == '08-13-02.xlsx', f"got={info and info['文件']}")
check('尾盘数据是中阳D 6.2%', abs(intra[intra['名称'] == '中阳D']['涨幅'].iloc[0] - 6.2) < 0.01)

print('== 4. 常规预测三清单 ==')
cfg = strategies.load_config(BASE)
data, _ = loader.load_dataset(DATA)
check('合并后最新日=20260813', data['数据日期'].max() == '20260813')
check('收盘优先（烂板C收盘9.98而非盘中8.5）',
      abs(data[data['名称'] == '烂板C']['涨幅'].iloc[0] - 9.98) < 0.01)
res = strategies.run_prediction(data, cfg)
check('涨停TopN 非空', len(res['涨停TopN']) > 0)
check('涨幅4% 非空', len(res['涨幅4%']) > 0)
check('连板候选 非空', len(res['连板候选']) > 0)
top1 = res['连板候选'].iloc[0]
print(f"   连板候选Top1: {top1['名称']} 连板潜力={top1['连板潜力']}")
check('连板Top1 是强封B(封成比25+一封到底+换手2%)', top1['名称'] == '强封B')

print('== 5. 尾盘选股清单 ==')
tail, note = strategies.run_tail_session(intra, cfg)
print('  ', note)
check('尾盘清单非空', len(tail) > 0)
check('微跌F被过滤（阴跌无博弈价值）', '微跌F' not in tail['名称'].values)
check('清单含类型列', '类型' in tail.columns)
print(tail[['排名', '名称', '类型', '涨幅', '尾盘评分']].to_string(index=False))

print('== 6. 入库 + close 优先级防污染 ==')
db = DB(TMP)
for i in infos:
    if i.get('cap') == 'predict' and not db.file_imported(i['hash']):
        d, _, _ = loader.clean(loader.read_any(i['path']))
        db.save_snapshot(i, d)
d14_info = None
write_tdx(os.path.join(DATA, '08-14-close.xlsx'), CLOSE_0814,
          [h.replace('20260813', '20260814') for h in HEADER])
for i in loader.scan_folder(DATA):
    if i.get('cap') == 'predict' and not db.file_imported(i['hash']):
        d, _, _ = loader.clean(loader.read_any(i['path']))
        db.save_snapshot(i, d)
daily13 = db.get_daily('20260813')
check('daily 0813 是收盘口径（烂板C 9.98）',
      abs(float(daily13[daily13['名称'] == '烂板C']['涨幅'].iloc[0]) - 9.98) < 0.01)

print('== 7. 预测入库（三清单 + 尾盘） ==')
db.save_predictions('20260813', '20260814', '涨停TopN', res['涨停TopN'], '预估涨停概率%', '综合分')
db.save_predictions('20260813', '20260814', '涨幅4%', res['涨幅4%'], '预估涨4概率%', '涨4评分')
db.save_predictions('20260813', '20260814', '连板候选', res['连板候选'], '预估涨停概率%', '连板潜力')
db.save_predictions('20260813', '20260814', '尾盘选股', tail, '预估次日溢价概率%', '尾盘评分')
check('pending_backtest_dates 含 20260814', '20260814' in db.pending_backtest_dates())

print('== 8. 分清单回测评分 ==')
for lt in bt.LIST_TYPES:
    mg, summary, ic = bt.backtest_one(db, '20260814', lt, cfg)
    if summary is None:
        print(f'   {lt}: 样本不足')
        continue
    print(f"   {lt}: 平均得分={summary['平均得分']} 总得分={summary['总得分']} 达成率={summary['达成率']}")
    check(f'{lt} 有得分列', mg is not None and '得分' in mg.columns)
# 抽查：连板候选里 烂板C 次日 -6.5% → 触发 [-99,-100] 档（<-6 即-100）
mg, _, _ = bt.backtest_one(db, '20260814', '连板候选', cfg)
if mg is not None and '烂板C' in mg['name'].values:
    sc = mg[mg['name'] == '烂板C']['得分'].iloc[0]
    check('烂板C(-6.5%) 连板规则扣100分', sc == -100, f'got={sc}')
mg2, _, _ = bt.backtest_one(db, '20260814', '涨停TopN', cfg)
if mg2 is not None and '强封B' in mg2['name'].values:
    sc = mg2[mg2['name'] == '强封B']['得分'].iloc[0]
    check('强封B(涨停) 涨停规则得100分', sc == 100, f'got={sc}')

print('== 9. 滚动 + 调参建议 ==')
roll, n = bt.rolling_ic(db, '涨停TopN', cfg)
print(f'   滚动IC {n} 天')
sc, ns = bt.rolling_score(db, '涨停TopN', cfg)
check('滚动得分表非空', not sc.empty)
advice, sug, _ = bt.make_advice(db, cfg)
print('   --- 建议 ---')
for line in advice:
    print('   ' + line)
check('建议文本非空', len(advice) > 0)

print()
fails = [n for n, c in ok if not c]
print(f'== 结果：{len(ok) - len(fails)}/{len(ok)} 通过 ==' + (f' 失败: {fails}' if fails else ''))
db.close()
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
