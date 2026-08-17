# -*- coding: utf-8 -*-
"""
================================================================================
涨停预测工具 v2.0  —— VSCode 打开本文件按 F5 直接运行
================================================================================
依赖安装（一次性）:
    pip install PySide6 pandas numpy scipy openpyxl

功能：
  [数据] 自动扫描本程序所在文件夹的 xlsx/csv，识别 开盘/盘中批次/收盘/池 快照并入库(SQLite)
  [预测] 一键生成：明日涨停TopN / 明日涨幅≥4% / 跨日连板候选 三张清单
  [尾盘] 盘中批次快照（08-14-01=当天第1次导出…取最新批）尾盘选股，博次日溢价
  [回测] 自动配对"历史预测 vs 实际收盘"：整体准确率 + 分清单评分（涨停满分/分档给分/下跌扣分）
         + 单策略IC + 滚动得分 + 自动调参建议（可一键应用建议权重）
  [参数] 策略权重/阈值可视化编辑，保存到 config.json 立即生效
================================================================================
"""
import json
import os
import sys
import traceback
from datetime import datetime

import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
                               QProgressDialog, QSpinBox, QTabWidget, QTableWidget,
                               QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')     # 数据文件统一放这里（自动创建）
os.makedirs(DATA_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

from core import backtest as bt
from core import calendar_cn as cal
from core import loader, strategies
from core.db import DB
from extensions import REGISTRY, enabled_extensions, get_source


def df_to_table(table, df, max_rows=200):
    """DataFrame → QTableWidget"""
    df = df.head(max_rows)
    table.setColumnCount(len(df.columns))
    table.setRowCount(len(df))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for i, (_, row) in enumerate(df.iterrows()):
        for j, v in enumerate(row):
            if pd.isna(v):
                s = ''
            elif isinstance(v, float):
                s = f'{v:.3f}'.rstrip('0').rstrip('.')
            else:
                s = str(v)
            item = QTableWidgetItem(s)
            if isinstance(v, (int, float)) and not pd.isna(v):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(i, j, item)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


class ExtContext:
    """传给扩展 run() 的上下文。框架注入，扩展据需取用。"""

    def __init__(self, db, cfg, base_dir, data_dir):
        self.db = db
        self.cfg = cfg
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.target_date = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('涨停预测工具 v2.0')
        self.resize(1280, 800)
        self.db = DB(BASE_DIR)
        self.cfg = strategies.load_config(BASE_DIR)
        cal.set_extra_holidays(self.cfg.get('extra_holidays', []))
        self.data = None          # 合并后的当日数据
        self.result = None        # 预测结果

        # ---------- 顶部状态栏 ----------
        top = QHBoxLayout()
        code, desc = cal.current_session()
        self.lbl_session = QLabel(f'📅 {datetime.now():%Y-%m-%d %A} | {desc}')
        self.lbl_session.setStyleSheet('font-weight:bold;padding:4px')
        btn_scan = QPushButton('🔄 扫描数据文件夹')
        btn_scan.clicked.connect(self.scan)
        btn_predict = QPushButton('🚀 生成预测')
        btn_predict.clicked.connect(self.predict)
        btn_tail = QPushButton('🌆 尾盘选股')
        btn_tail.clicked.connect(self.tail_select)
        btn_backtest = QPushButton('📊 运行回测')
        btn_backtest.clicked.connect(self.run_backtest)
        for b in (btn_scan, btn_predict, btn_tail, btn_backtest):
            b.setMinimumHeight(32)
        top.addWidget(self.lbl_session, 1)
        top.addWidget(btn_scan)
        top.addWidget(btn_predict)
        top.addWidget(btn_tail)
        top.addWidget(btn_backtest)

        # ---------- 主区 Tabs ----------
        self.tabs = QTabWidget()
        self.tab_files = self._build_files_tab()
        self.tab_pred = self._build_pred_tab()
        self.tab_tail = self._build_tail_tab()
        self.tab_bt = self._build_backtest_tab()
        self.tab_cfg = self._build_config_tab()
        self.tab_log = self._build_log_tab()
        self.ext_widgets = {}      # key -> {param_name: widget, '_info', '_res_tabs', '_tab'}
        self.ext_tabs = {}         # key -> tab widget
        for name, w in [('📁 数据文件', self.tab_files), ('🚀 预测结果', self.tab_pred),
                        ('🌆 尾盘选股', self.tab_tail), ('📊 回测评估', self.tab_bt),
                        ('⚙️ 策略参数', self.tab_cfg), ('📜 日志', self.tab_log)]:
            self.tabs.addTab(w, name)

        # ---------- 扩展能力 Tabs（自动发现，无需为新增能力改这里） ----------
        for key, ext in enabled_extensions(self.cfg).items():
            tab, widgets = self._build_extension_tab(ext)
            self.ext_widgets[key] = widgets
            self.ext_tabs[key] = tab
            self.tabs.addTab(tab, f'🔌 {ext.name}')

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addLayout(top)
        lay.addWidget(self.tabs)
        self.setCentralWidget(root)
        self.scan()               # 启动自动扫描

    # ================= Tabs =================
    def _build_files_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.tbl_files = QTableWidget()
        lay.addWidget(QLabel('程序自动扫描 data/ 文件夹的 xlsx/csv，识别每个文件的日期、类型和用途：'))
        lay.addWidget(self.tbl_files)
        return w

    def _build_pred_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_pred_info = QLabel('点击顶部「生成预测」')
        self.pred_tabs = QTabWidget()
        self.tbl_limit = QTableWidget()
        self.tbl_rise4 = QTableWidget()
        self.tbl_monday = QTableWidget()
        self.pred_tabs.addTab(self.tbl_limit, '明日涨停 TopN')
        self.pred_tabs.addTab(self.tbl_rise4, '明日涨幅≥4%')
        self.pred_tabs.addTab(self.tbl_monday, '跨日连板候选')
        lay.addWidget(self.lbl_pred_info)
        lay.addWidget(self.pred_tabs)
        return w

    def _build_tail_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_tail_info = QLabel(
            '盘中把批次快照放进 data/（命名：日期+批次号，如 08-14-01.xlsx=当天第1次导出、'
            '08-14-02.xlsx=第2次），点顶部「🌆 尾盘选股」自动取最新批次选股。')
        self.lbl_tail_info.setWordWrap(True)
        self.tbl_tail = QTableWidget()
        lay.addWidget(self.lbl_tail_info)
        lay.addWidget(self.tbl_tail)
        return w

    def _build_backtest_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel('回测目标日:'))
        self.cmb_bt_date = QComboBox()
        row.addWidget(self.cmb_bt_date)
        row.addWidget(QLabel('清单:'))
        self.cmb_list = QComboBox()
        self.cmb_list.addItems(bt.LIST_TYPES)
        row.addWidget(self.cmb_list)
        self.lbl_bt_summary = QLabel('—')
        row.addWidget(self.lbl_bt_summary, 1)
        lay.addLayout(row)
        self.bt_tabs = QTabWidget()
        self.tbl_bt_detail = QTableWidget()
        self.tbl_bt_ic = QTableWidget()
        self.tbl_bt_rolling = QTableWidget()
        self.tbl_bt_score = QTableWidget()
        self.bt_tabs.addTab(self.tbl_bt_detail, '单日明细')
        self.bt_tabs.addTab(self.tbl_bt_ic, '单策略IC(当日)')
        self.bt_tabs.addTab(self.tbl_bt_rolling, '策略IC滚动汇总')
        self.bt_tabs.addTab(self.tbl_bt_score, '清单得分滚动')
        advice_w = QWidget()
        al = QVBoxLayout(advice_w)
        # 动态权重开关
        dw_cfg = self.cfg.get('dynamic_weights', {})
        self.chk_auto_kill = QCheckBox('负IC因子自动归零（0%，硬kill反向因子）')
        self.chk_auto_kill.setChecked(bool(dw_cfg.get('auto_kill_negative', True)))
        self.chk_auto_kill.stateChanged.connect(
            lambda s: self._set_dw('auto_kill_negative', s == 2))
        self.chk_auto_apply = QCheckBox('回测时自动应用动态权重（写入配置并刷新参数页）')
        self.chk_auto_apply.setChecked(bool(dw_cfg.get('auto_apply', False)))
        self.chk_auto_apply.stateChanged.connect(
            lambda s: self._set_dw('auto_apply', s == 2))
        al.addWidget(self.chk_auto_kill)
        al.addWidget(self.chk_auto_apply)
        self.btn_apply_w = QPushButton('✅ 一键应用建议权重（覆盖当前五策略权重并保存）')
        self.btn_apply_w.clicked.connect(self.apply_suggested_weights)
        al.addWidget(self.btn_apply_w)
        self.txt_advice = QTextEdit()
        self.txt_advice.setReadOnly(True)
        al.addWidget(self.txt_advice)
        al.addWidget(QLabel('动态权重计划（基于滚动IC，自动归零负IC因子）：'))
        self.tbl_dyn_w = QTableWidget()
        al.addWidget(self.tbl_dyn_w)
        self.bt_tabs.addTab(advice_w, '💡 调参建议')
        lay.addWidget(self.bt_tabs)
        self._suggested_weights = None
        return w

    def _build_config_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel('策略权重（保存后立即生效，自动归一化）：'))
        self.tbl_weights = QTableWidget()
        lay.addWidget(self.tbl_weights)
        lay.addWidget(QLabel('关键阈值与参数：'))
        self.form = QFormLayout()
        self.spin = {}
        for key, label, lo, hi, step in [
                ('base_limit_up', '一进二基准概率', 0.01, 0.5, 0.01),
                ('base_rush', '冲板基准概率', 0.01, 0.3, 0.01),
                ('base_rise4', '涨4%基准概率', 0.01, 0.4, 0.01)]:
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(3)
            sp.setValue(self.cfg.get(key, 0.15))
            self.spin[key] = sp
            self.form.addRow(label, sp)
        for key in ['涨停判定涨幅', '封流比_强', '封成比_极强', '锁仓换手_紧',
                    '小市值上限', '高位20日涨幅', '开板次数_风险']:
            sp = QDoubleSpinBox()
            sp.setRange(0, 9999)
            sp.setValue(float(self.cfg['thresholds'].get(key, 0)))
            self.spin['th_' + key] = sp
            self.form.addRow(key, sp)
        for key, label in [('top_n', '涨停清单长度'), ('rise4_n', '涨4%清单长度'),
                           ('monday_n', '连板候选长度'),
                           ('tail_n', '尾盘选股长度')]:
            sp = QDoubleSpinBox()
            sp.setRange(1, 100)
            sp.setValue(float(self.cfg.get(key, 20)))
            self.spin[key] = sp
            self.form.addRow(label, sp)
        lay.addLayout(self.form)
        row = QHBoxLayout()
        b1 = QPushButton('💾 保存参数')
        b1.clicked.connect(self.save_cfg)
        b2 = QPushButton('↩️ 恢复默认')
        b2.clicked.connect(self.reset_cfg)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch(1)
        lay.addLayout(row)
        self._load_weights_table()
        return w

    def _build_log_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        lay.addWidget(self.txt_log)
        return w

    # ================= 扩展能力 Tabs（通用，自动渲染） =================
    def _build_extension_tab(self, ext):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel(ext.description))
        info = QLabel('点击「运行」基于数据源生成清单（结果自动入库，次日可回测）')
        info.setWordWrap(True)
        lay.addWidget(info)
        # 参数自动渲染
        form = QFormLayout()
        widgets = {}
        for p in ext.params:
            if p.kind == 'bool':
                ctl = QCheckBox()
                ctl.setChecked(bool(p.default))
                widgets[p.name] = ctl
                form.addRow(p.label, ctl)
            elif p.kind == 'choice':
                ctl = QComboBox()
                ctl.addItems(p.choices or [])
                if p.default in (p.choices or []):
                    ctl.setCurrentText(p.default)
                widgets[p.name] = ctl
                form.addRow(p.label, ctl)
            elif p.kind == 'str':
                ctl = QLineEdit()
                ctl.setText(str(p.default) if p.default is not None else '')
                widgets[p.name] = ctl
                form.addRow(p.label, ctl)
            else:
                ctl = QDoubleSpinBox() if p.kind == 'float' else QSpinBox()
                if p.min is not None:
                    ctl.setRange(p.min, p.max if p.max is not None else 999999)
                ctl.setValue(p.default if p.default is not None else 0)
                if p.kind == 'float':
                    ctl.setDecimals(2)
                if p.step:
                    ctl.setSingleStep(p.step)
                widgets[p.name] = ctl
                form.addRow(p.label, ctl)
        lay.addLayout(form)
        btn = QPushButton(f'🚀 运行「{ext.name}」')
        btn.clicked.connect(lambda _, e=ext: self.run_extension(e))
        lay.addWidget(btn)
        # 支持训练的能力（如神经网络）额外显示「训练/更新模型」按钮
        if getattr(ext, 'supports_training', False):
            btn_tr = QPushButton(f'🧠 训练/更新模型「{ext.name}」')
            btn_tr.clicked.connect(lambda _, e=ext: self.train_extension(e))
            btn_tr.setEnabled(getattr(ext, 'datasource', 'local') == 'akshare')
            if btn_tr.isEnabled():
                btn_tr.setToolTip('从 AKShare 拉历史行情自举训练（首次较慢，结果缓存到 nn_model.db）')
            else:
                btn_tr.setToolTip('该能力数据源非 akshare，无需训练')
            lay.addWidget(btn_tr)
            widgets['_btn_train'] = btn_tr
        res_tabs = QTabWidget()
        lay.addWidget(res_tabs, 1)
        widgets['_info'] = info
        widgets['_res_tabs'] = res_tabs
        return w, widgets

    def run_extension(self, ext):
        try:
            src = get_source(ext.datasource)
            if src is None:
                QMessageBox.warning(self, '无数据源', f'数据源 {ext.datasource} 未注册')
                return
            ctx = ExtContext(self.db, self.cfg, BASE_DIR, DATA_DIR)
            df, info = src.fetch(ctx)
            # 读取界面参数控件当前值（覆盖 config 与默认）
            widgets = self.ext_widgets[ext.key]
            overrides = {}
            for p in ext.params:
                ctl = widgets.get(p.name)
                if ctl is None:
                    continue
                if p.kind == 'bool':
                    overrides[p.name] = ctl.isChecked()
                elif p.kind == 'choice':
                    overrides[p.name] = ctl.currentText()
                elif p.kind == 'float':
                    overrides[p.name] = float(ctl.value())
                elif p.kind == 'str':
                    overrides[p.name] = ctl.text()
                else:  # int
                    overrides[p.name] = int(ctl.value())
            cfg = ext.effective_params(
                self.cfg.get('extensions', {}).get(ext.key, {}), overrides)
            result = ext.run(df, cfg, ctx)
            main = result.get('主表')
            note = result.get('note', '')
            widgets = self.ext_widgets[ext.key]
            widgets['_info'].setText(
                f'数据源：{info.get("来源", "?")} | 数据日期 {info.get("date", "?")} | {note}')
            rt = widgets['_res_tabs']
            rt.clear()
            if main is not None and not main.empty and ext.list_type:
                t = QTableWidget()
                df_to_table(t, main)
                rt.addTab(t, '主表')
                for name, sub in (result.get('tables') or {}).items():
                    tt = QTableWidget()
                    df_to_table(tt, sub)
                    rt.addTab(tt, name)
                # 落库（与现有清单同口径，次日收盘数据到达后可回测）
                date_str = str(info.get('date', ''))
                target = (cal.next_trading_day(
                    datetime.strptime(date_str, '%Y%m%d').date())
                    if date_str else None)
                if target:
                    self.db.save_predictions(
                        date_str, target.strftime('%Y%m%d'), ext.list_type,
                        main, ext.prob_col, ext.score_col,
                        detail_keys=ext.detail_keys)
                    self._refresh_bt_dates()
                    self.log(f'扩展「{ext.name}」完成：{len(main)} 只 → {ext.list_type}，'
                             f'目标日 {target}，已入库')
            else:
                self.log(f'扩展「{ext.name}」：{note}')
            self.tabs.setCurrentWidget(self.ext_tabs[ext.key])
        except Exception:
            self.log(f'扩展「{ext.name}」出错:\n' + traceback.format_exc())
            QMessageBox.critical(self, '错误', traceback.format_exc()[-500:])

    # ================= 逻辑 =================
    def log(self, msg):
        self.txt_log.append(f'[{datetime.now():%H:%M:%S}] {msg}')

    # ================= 神经网络等需联网自举扩展的训练 =================
    def train_extension(self, ext):
        """对支持训练的能力（如神经网络）执行训练/更新模型。后台线程避免 GUI 卡死。"""
        widgets = self.ext_widgets.get(ext.key, {})
        overrides = {}
        for p in ext.params:
            ctl = widgets.get(p.name)
            if ctl is None:
                continue
            if p.kind == 'bool':
                overrides[p.name] = ctl.isChecked()
            elif p.kind == 'choice':
                overrides[p.name] = ctl.currentText()
            elif p.kind == 'float':
                overrides[p.name] = float(ctl.value())
            else:
                overrides[p.name] = int(ctl.value())
        cfg = ext.effective_params(self.cfg.get('extensions', {}).get(ext.key, {}), overrides)
        ctx = ExtContext(self.db, self.cfg, BASE_DIR, DATA_DIR)
        self._train_thread = TrainWorker(ext, ctx, cfg)
        dlg = QProgressDialog(f'训练「{ext.name}」…（拉取历史行情并训练，首次较慢）', None, 0, 0, self)
        dlg.setWindowTitle('模型训练')
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        self._train_thread.progress.connect(dlg.setLabelText)
        self._train_thread.finished.connect(
            lambda ok, m: self._on_train_done(ext, ok, m, dlg))
        self._train_thread.start()

    def _on_train_done(self, ext, ok, msg, dlg):
        dlg.close()
        self.log(f'训练「{ext.name}」: {"成功" if ok else "失败"} - {msg}')
        QMessageBox.information(self, '训练完成' if ok else '训练失败', str(msg)[:800])
        self.tabs.setCurrentWidget(self.ext_tabs[ext.key])

    def scan(self):
        try:
            infos = loader.scan_folder(DATA_DIR)
            df = pd.DataFrame([{k: v for k, v in i.items()
                                if k not in ('path', 'hash', 'kind', 'cap', 'date', 'batch')}
                               for i in infos])
            df_to_table(self.tbl_files, df)
            n_pred = sum(1 for i in infos if i.get('cap') == 'predict')
            n_bt = sum(1 for i in infos if i.get('cap') == 'backtest')
            self.log(f'扫描目录 data/ 完成：{len(infos)} 个文件，其中 {n_pred} 个可用于预测，{n_bt} 个可用于回测')
            # 新文件自动入库（非收盘文件在当天已有收盘数据时只登记、不覆盖 daily）
            for i in infos:
                if i.get('cap') == 'predict' and not self.db.file_imported(i['hash']):
                    d, _, _ = loader.clean(loader.read_any(i['path']))
                    n = self.db.save_snapshot(i, d)
                    if n == 0:
                        self.log(f"登记：{i['文件']}（当日已有收盘数据，快照不入 daily 防污染）")
                    else:
                        self.log(f"入库：{i['文件']}（{i['日期']}，{n} 行）")
            self._refresh_bt_dates()
        except Exception:
            self.log('扫描出错:\n' + traceback.format_exc())

    def predict(self):
        try:
            data, infos = loader.load_dataset(DATA_DIR)
            if data.empty:
                QMessageBox.warning(self, '无数据', '未找到可用数据文件')
                return
            self.data = data
            self.result = strategies.run_prediction(data, self.cfg)
            date_str = data['数据日期'].max()
            target = cal.next_trading_day(datetime.strptime(date_str, '%Y%m%d').date())

            cols = ['排名', '代码', '名称', '今日涨停', 'S1封单强度', 'S2封板质量',
                    'S3锁仓度', 'S4资金', 'S5股性结构', 'M可买性', 'R风险系数',
                    '预估涨停概率%', '综合分']
            cols = [c for c in cols if c in data.columns or c in self.result['涨停TopN'].columns]
            df_to_table(self.tbl_limit, self.result['涨停TopN'][cols])
            r4cols = ['排名', '代码', '名称', '涨幅', '量比', '资金流向', '涨4评分', '预估涨4概率%']
            df_to_table(self.tbl_rise4,
                        self.result['涨幅4%'][[c for c in r4cols if c in self.result['涨幅4%'].columns]])
            mcols = cols + ['连板潜力']
            df_to_table(self.tbl_monday,
                        self.result['连板候选'][[c for c in mcols if c in self.result['连板候选'].columns]])

            cv = strategies.cross_validate(self.result['全量'])
            cv_txt = ' | '.join(f'{k}: ρ={v[0]}' for k, v in cv.items()) if cv else '样本不足'
            self.lbl_pred_info.setText(
                f"数据日期 {date_str} → 预测目标日 {target:%Y-%m-%d} | "
                f"涨停股 {int(data['今日涨停'].sum())} 只 | 交叉验证 {cv_txt} | {self.result['连板备注']}")
            # 预测入库（供未来回测）
            self.db.save_predictions(date_str, target.strftime('%Y%m%d'), '涨停TopN',
                                     self.result['涨停TopN'], '预估涨停概率%', '综合分')
            self.db.save_predictions(date_str, target.strftime('%Y%m%d'), '涨幅4%',
                                     self.result['涨幅4%'], '预估涨4概率%', '涨4评分')
            self.db.save_predictions(date_str, target.strftime('%Y%m%d'), '连板候选',
                                     self.result['连板候选'], '预估涨停概率%', '连板潜力')
            self.log(f'预测完成：目标日 {target}，三张清单已入库（明日收盘数据导入后可回测）')
            self.tabs.setCurrentWidget(self.tab_pred)
        except Exception:
            self.log('预测出错:\n' + traceback.format_exc())
            QMessageBox.critical(self, '错误', traceback.format_exc()[-500:])

    def tail_select(self):
        """尾盘选股：取最新日期的最新盘中批次快照 → 尾盘清单 → 入库待回测"""
        try:
            data, info = loader.load_latest_intraday(DATA_DIR)
            if data.empty:
                QMessageBox.warning(
                    self, '无盘中数据',
                    '未找到盘中批次快照。\n\n命名格式：日期+批次号（当天第几次导出）\n'
                    '  08-14-01.xlsx  第1次导出（如 10:30）\n'
                    '  08-14-02.xlsx  第2次导出（如 14:30 尾盘）\n'
                    '自动取当天批次号最大的文件。')
                return
            res, note = strategies.run_tail_session(data, self.cfg)
            if res.empty:
                self.lbl_tail_info.setText(note)
                self.tbl_tail.setRowCount(0)
                return
            date_str = data['数据日期'].max()
            target = cal.next_trading_day(datetime.strptime(date_str, '%Y%m%d').date())
            cols = ['排名', '代码', '名称', '类型', '涨幅', '量比', '换手', '封流比',
                    '主力净量', '强势', '资金', '量能', '位置',
                    '尾盘评分', 'M可买性', 'R风险系数', '预估次日溢价概率%']
            df_to_table(self.tbl_tail, res[[c for c in cols if c in res.columns]])
            _, desc = cal.current_session()
            self.lbl_tail_info.setText(
                f"📄 {info['文件']}（第{info.get('batch') or '?'}批，{len(data)}只）→ "
                f"目标日 {target:%Y-%m-%d} | {note} | 当前时段：{desc}")
            self.db.save_predictions(date_str, target.strftime('%Y%m%d'), '尾盘选股',
                                     res, '预估次日溢价概率%', '尾盘评分')
            self._refresh_bt_dates()
            self.log(f'尾盘选股完成：{info["文件"]} → {len(res)} 只，目标日 {target}，已入库待回测')
            self.tabs.setCurrentWidget(self.tab_tail)
        except Exception:
            self.log('尾盘选股出错:\n' + traceback.format_exc())
            QMessageBox.critical(self, '错误', traceback.format_exc()[-500:])

    def run_backtest(self):
        try:
            d = self.cmb_bt_date.currentText()
            if not d:
                QMessageBox.information(self, '提示', '没有可回测的日期。\n回测需要：先有预测记录，再导入目标日收盘数据。')
                return
            lt = self.cmb_list.currentText()
            mg, summary, ic = bt.backtest_one(self.db, d, lt, self.cfg)
            if summary is None:
                self.lbl_bt_summary.setText('该清单在此日匹配样本不足')
                return
            self.lbl_bt_summary.setText(' | '.join(f'{k}:{v}' for k, v in summary.items()))
            if mg is not None:
                df_to_table(self.tbl_bt_detail,
                            mg[[c for c in ['rank', 'code', 'name', 'prob', '实际涨幅%', '实际涨停', '得分']
                                if c in mg.columns]])
            if ic:
                df_to_table(self.tbl_bt_ic, pd.DataFrame(ic).T.reset_index().rename(
                    columns={'index': '策略'}))
            roll, n_days = bt.rolling_ic(self.db, lt, self.cfg)
            if not roll.empty:
                df_to_table(self.tbl_bt_rolling, roll)
            sc, _ = bt.rolling_score(self.db, lt, self.cfg)
            if not sc.empty:
                df_to_table(self.tbl_bt_score, sc)
            advice, sug, plan = bt.make_advice(self.db, self.cfg)
            self.txt_advice.setPlainText('\n'.join(advice))
            self._suggested_weights = sug
            if plan is not None:
                df_to_table(self.tbl_dyn_w, plan)
            else:
                self.tbl_dyn_w.setRowCount(0)
            # 自动应用动态权重（不打断用户，仅记录日志）
            if self.chk_auto_apply.isChecked() and sug:
                self.cfg['weights'].update(sug)
                strategies.save_config(BASE_DIR, self.cfg)
                self._load_weights_table()
                self.log(f'已自动应用动态权重：{sug}')
            self.log(f'回测完成：{d} / {lt}，累计可回测 {n_days} 天，调参建议已更新')
            self.tabs.setCurrentWidget(self.tab_bt)
        except Exception:
            self.log('回测出错:\n' + traceback.format_exc())

    def apply_suggested_weights(self):
        if not self._suggested_weights:
            QMessageBox.information(self, '提示', '当前没有可应用的建议权重。\n先在「回测评估」跑一次回测，且累计可回测天数 ≥3。')
            return
        self.cfg['weights'].update(self._suggested_weights)
        strategies.save_config(BASE_DIR, self.cfg)
        self._load_weights_table()
        self.log(f'已应用建议权重：{self._suggested_weights}')
        QMessageBox.information(self, '已应用', '建议权重已写入 config.json 并刷新到参数页，下次预测生效')

    def _set_dw(self, key, val):
        """动态权重开关状态变更 → 写入 config.dynamic_weights 并落盘。"""
        self.cfg.setdefault('dynamic_weights', {})[key] = val
        strategies.save_config(BASE_DIR, self.cfg)
        self.log(f'动态权重设置 {key} = {val}')

    def _refresh_bt_dates(self):
        self.cmb_bt_date.clear()
        for d in self.db.pending_backtest_dates():
            self.cmb_bt_date.addItem(d)

    def _load_weights_table(self):
        w = self.cfg['weights']
        self.tbl_weights.setColumnCount(2)
        self.tbl_weights.setRowCount(len(w))
        self.tbl_weights.setHorizontalHeaderLabels(['策略', '权重'])
        for i, (k, v) in enumerate(w.items()):
            self.tbl_weights.setItem(i, 0, QTableWidgetItem(k))
            self.tbl_weights.setItem(i, 1, QTableWidgetItem(str(v)))
        self.tbl_weights.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def save_cfg(self):
        try:
            for i in range(self.tbl_weights.rowCount()):
                k = self.tbl_weights.item(i, 0).text()
                self.cfg['weights'][k] = float(self.tbl_weights.item(i, 1).text())
            for key, sp in self.spin.items():
                if key.startswith('th_'):
                    self.cfg['thresholds'][key[3:]] = sp.value()
                elif key in ('top_n', 'rise4_n', 'monday_n', 'tail_n'):
                    self.cfg[key] = int(sp.value())
                else:
                    self.cfg[key] = sp.value()
            strategies.save_config(BASE_DIR, self.cfg)
            self.log('参数已保存到 config.json')
            QMessageBox.information(self, '保存成功', '参数已保存，下次预测立即生效')
        except Exception as e:
            QMessageBox.critical(self, '保存失败', str(e))

    def reset_cfg(self):
        self.cfg = strategies.load_config('/nonexistent')  # 拿默认值
        strategies.save_config(BASE_DIR, self.cfg)
        self._load_weights_table()
        self.log('已恢复默认参数')

    def closeEvent(self, e):
        self.db.close()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# 后台训练线程（神经网络等需联网自举的扩展）—— 模块级，避免打断 MainWindow 类体
# ---------------------------------------------------------------------------
class TrainWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, ext, ctx, cfg):
        super().__init__()
        self.ext = ext
        self.ctx = ctx
        self.cfg = cfg

    def run(self):
        try:
            ok, msg = self.ext.train(self.ctx, self.cfg,
                                     progress=lambda d, t, c: self.progress.emit(f'拉历史 {d}/{t} {c}'))
            self.finished.emit(ok, msg)
        except Exception:
            self.finished.emit(False, '训练异常:\n' + traceback.format_exc()[-500:])