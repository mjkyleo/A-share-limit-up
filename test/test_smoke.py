# -*- coding: utf-8 -*-
"""补充测试：① 多日(3天)回测 → 建议权重数值输出 ② GUI offscreen 全链路冒烟"""
import json
import os
import random
import shutil
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from core import backtest as bt
from core import strategies
from core.db import DB

TMP = tempfile.mkdtemp(prefix='limitup_smoke_')
db = DB(TMP)

# ================= ① 3天回测 → 建议权重 =================
random.seed(7)
days = [('20260810', '20260811'), ('20260811', '20260812'), ('20260812', '20260813')]
for run_d, tgt_d in days:
    rows = []
    for i in range(12):
        s1, s4 = random.random(), random.random()
        actual = 6 * s1 - 4 * s4 + random.uniform(-2, 2)   # S1正相关 / S4负相关
        rows.append({'code': f'600{i:03d}', 'name': f'股{i}', 's1': s1, 's4': s4,
                     'actual': round(actual, 2)})
    cur = db.conn.cursor()
    for r in rows:
        cur.execute('INSERT OR REPLACE INTO daily VALUES(?,?,?,?)',
                    (tgt_d, r['code'], r['name'],
                     json.dumps({'代码6': r['code'], '名称': r['name'], '涨幅': r['actual'],
                                 '今日涨停': r['actual'] >= 9.5}, ensure_ascii=False)))
    import pandas as pd
    df = pd.DataFrame([{
        '代码6': r['code'], '名称': r['name'], '预估涨停概率%': 20.0, '综合分': r['s1'],
        'S1封单强度': r['s1'], 'S2封板质量': 0.5, 'S3锁仓度': 0.5,
        'S4资金': r['s4'], 'S5股性结构': 0.5} for r in rows])
    db.save_predictions(run_d, tgt_d, '涨停TopN', df, '预估涨停概率%', '综合分')
db.conn.commit()

cfg = strategies.load_config(BASE)
advice, sug, _ = bt.make_advice(db, cfg)
print('== 3天回测的调参建议 ==')
for line in advice:
    print(line)
assert sug is not None, '≥3天应给出建议权重'
assert abs(sum(sug.values()) - 1.0) < 0.05, f'建议权重应归一: {sug}'
assert sug['S1封单强度'] > sug['S4资金'], 'S1应显著高于S4'
print(f'\n建议权重校验通过: {sug}')
db.close()

# ================= ② GUI offscreen 全链路冒烟 =================
print('\n== GUI offscreen 冒烟 ==')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication, QMessageBox
# 屏蔽模态弹窗（offscreen 无人点击会阻塞）
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.critical = staticmethod(lambda *a, **k: None)

import app as gui
from core import loader

G_TMP = tempfile.mkdtemp(prefix='limitup_gui_')
G_DATA = os.path.join(G_TMP, 'data')
os.makedirs(G_DATA)
gui.BASE_DIR = G_TMP          # DB/config 写入临时目录，不污染项目目录
gui.DATA_DIR = G_DATA

HEADER = ['代码', '名称', '涨幅', '开盘涨幅', '量比', '换手', '振幅', '竞价换手',
          '涨停封流比[20260813]', '涨停封成比[20260813]', '首次涨停时间[20260813]',
          '涨停开板次数[20260813]', '流通市值', '主力净量', '资金流向',
          '5日涨幅', '20日涨幅', '市盈(动)', '总手']
ROWS13 = [
    ('SH600111', '强封A', '+10.02%', '+2.1%', '2.5', '6.5', '8.2', '0.8', '5.2', '12.0', '09:31:05', '0', '45亿', '1.8', '5200万', '8', '15', '25', '35万'),
    ('SH600222', '强封B', '+10.01%', '+1.5%', '1.8', '2.1', '3.1', '0.5', '6.8', '25.0', '09:30:30', '0', '28亿', '2.2', '8100万', '12', '22', '31', '20万'),
    ('SZ300333', '烂板C', '+9.98%', '+4.2%', '5.5', '18.5', '12.5', '1.2', '1.2', '2.5', '14:20:11', '4', '95亿', '-0.5', '-2300万', '25', '45', '亏损', '80万'),
    ('SZ000444', '中阳D', '+5.5%', '+1.0%', '2.2', '8.5', '6.5', '0.6', '--', '--', '--', '--', '60亿', '0.8', '1500万', '6', '10', '18', '50万'),
    ('SZ000555', '小阳E', '+2.5%', '+0.5%', '1.2', '4.5', '4.0', '0.4', '--', '--', '--', '--', '35亿', '0.3', '500万', '3', '8', '22', '30万'),
    ('SZ000777', '大阳G', '+7.8%', '+2.8%', '3.2', '12.0', '9.5', '0.9', '--', '--', '--', '--', '55亿', '1.5', '3200万', '10', '18', '35', '65万'),
]
INTRA = [  # 08-13-02 尾盘批
    ('SH600111', '强封A', '+10.02%', '+2.1%', '2.3', '6.0', '8.0', '0.8', '5.2', '12.0', '09:31:05', '0', '45亿', '1.8', '5000万', '8', '15', '25', '32万'),
    ('SH600222', '强封B', '+10.01%', '+1.5%', '1.7', '2.0', '3.0', '0.5', '6.8', '25.0', '09:30:30', '0', '28亿', '2.2', '8000万', '12', '22', '31', '18万'),
    ('SZ300333', '烂板C', '+8.5%', '+4.2%', '4.8', '16.0', '11.5', '1.2', '--', '--', '--', '--', '95亿', '-0.5', '-2100万', '25', '45', '亏损', '72万'),
    ('SZ000444', '中阳D', '+6.2%', '+1.0%', '2.5', '9.0', '7.0', '0.6', '--', '--', '--', '--', '60亿', '1.2', '2100万', '6', '10', '18', '55万'),
    ('SZ000555', '小阳E', '+1.8%', '+0.5%', '1.1', '4.0', '3.5', '0.4', '--', '--', '--', '--', '35亿', '0.2', '300万', '3', '8', '22', '28万'),
    ('SZ000777', '大阳G', '+8.5%', '+2.8%', '3.5', '13.0', '10.0', '0.9', '--', '--', '--', '--', '55亿', '1.8', '3800万', '10', '18', '35', '70万'),
]
ROWS14 = [  # 次日实际收盘
    ('SH600111', '强封A', '+10.00%'), ('SH600222', '强封B', '+9.99%'),
    ('SZ300333', '烂板C', '-6.5%'), ('SZ000444', '中阳D', '+4.5%'),
    ('SZ000555', '小阳E', '+1.0%'), ('SZ000777', '大阳G', '+5.2%'),
]
ROWS14 = [(c, n, z, '+1.0%', '2.0', '6.0', '6.0', '0.6', '3.0', '4.0', '10:00:00', '1',
           '50亿', '1.0', '1000万', '8', '15', '25', '30万') for c, n, z in ROWS14]


def write_tdx(path, rows, header):
    with open(path, 'w', encoding='gbk', newline='') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')


write_tdx(os.path.join(G_DATA, '08-13-close.xlsx'), ROWS13, HEADER)
write_tdx(os.path.join(G_DATA, '08-13-02.xlsx'), INTRA, HEADER)
write_tdx(os.path.join(G_DATA, '08-14-close.xlsx'), ROWS14,
          [h.replace('20260813', '20260814') for h in HEADER])

qapp = QApplication([])
win = gui.MainWindow()                       # 启动自动 scan + 入库
assert win.tbl_files.rowCount() == 3, f'扫描表格应有3文件: {win.tbl_files.rowCount()}'
win.predict()                                # 常规三清单
assert win.result is not None, 'predict 未产出结果'
assert win.tbl_limit.rowCount() > 0 and win.tbl_monday.rowCount() > 0
win.tail_select()                            # 尾盘选股
assert win.tbl_tail.rowCount() > 0, '尾盘清单为空'
# 回测：逐清单跑一遍（扩展清单未在此测试落库，会"样本不足"，故校验"任一清单"有得分）
summaries = []
for i in range(win.cmb_list.count()):
    win.cmb_list.setCurrentIndex(i)
    win.run_backtest()
    summaries.append(win.lbl_bt_summary.text())
assert any('平均得分' in s for s in summaries), '回测未产出任何清单得分: ' + ' | '.join(summaries)
assert len(win.txt_advice.toPlainText()) > 10, '建议文本为空'
win.save_cfg()                               # 参数保存
win._suggested_weights = {'S1封单强度': 0.5, 'S2封板质量': 0.2, 'S3锁仓度': 0.15,
                          'S4资金': 0.03, 'S5股性结构': 0.12}
win.apply_suggested_weights()                # 一键应用建议权重
cfg2 = json.load(open(os.path.join(G_TMP, 'config.json'), encoding='utf-8'))
assert abs(cfg2['weights']['S1封单强度'] - 0.5) < 1e-9, cfg2['weights']
win.close()
print('GUI 全链路冒烟通过：扫描→预测→尾盘→四清单回测→建议→保存→应用权重')
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(G_TMP, ignore_errors=True)
print('\n全部补充测试通过 ✅')
