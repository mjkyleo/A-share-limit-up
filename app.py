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
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
                               QProgressDialog, QRadioButton, QSpinBox, QTabWidget,
                               QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')     # 数据文件统一放这里（自动创建）
os.makedirs(DATA_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

from core import backtest as bt
from core import metrics
from core import calendar_cn as cal
from core import loader, strategies
from core.db import DB
from extensions import REGISTRY, enabled_extensions, get_source

# P0-4：research screen 显示层重命名（仅改显示，内部列名/入库字段不变）
PROB_ALIASES = {
    '预估涨停概率%': '排序分（非真实概率）',
    '预估涨4概率%': '排序分（非真实概率）',
    '预估次日溢价概率%': '排序分（非真实概率）',
}
RESEARCH_NOTE = ('⚠️ 研究屏幕：以上"排序分（非真实概率）"仅用于排序参考，'
                 '非真实概率、非收益承诺。')


def df_to_table(table, df, max_rows=200, aliases=None):
    """DataFrame → QTableWidget。

    aliases: 列名显示层重命名映射（P0-4：仅改显示，不动内部列名/入库字段）。
    """
    df = df.head(max_rows)
    if aliases:
        df = df.rename(columns=aliases)
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
        self.lbl_pred_foot = QLabel(RESEARCH_NOTE)
        self.lbl_pred_foot.setWordWrap(True)
        self.lbl_pred_foot.setStyleSheet('color:#b8860b;font-size:11px')
        lay.addWidget(self.lbl_pred_foot)
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
        self.lbl_tail_foot = QLabel(RESEARCH_NOTE)
        self.lbl_tail_foot.setWordWrap(True)
        self.lbl_tail_foot.setStyleSheet('color:#b8860b;font-size:11px')
        lay.addWidget(self.lbl_tail_foot)
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
        row.addWidget(QLabel('收益口径:'))
        self.radio_close = QRadioButton('close(T日收→T+1收)')
        self.radio_open = QRadioButton('open(T+1开→T+1收)')
        self.radio_close.setChecked(True)
        row.addWidget(self.radio_close)
        row.addWidget(self.radio_open)
        self.lbl_bt_summary = QLabel('—')
        row.addWidget(self.lbl_bt_summary, 1)
        lay.addLayout(row)
        self.bt_tabs = QTabWidget()
        self.tbl_bt_detail = QTableWidget()
        self.tbl_bt_ic = QTableWidget()
        self.tbl_bt_rolling = QTableWidget()
        self.tbl_bt_score = QTableWidget()
        self.tbl_bt_perf = QTableWidget()
        self.tbl_bt_corr = QTableWidget()
        self.tbl_bt_incic = QTableWidget()
        self.bt_tabs.addTab(self.tbl_bt_detail, '单日明细')
        self.bt_tabs.addTab(self.tbl_bt_ic, '单策略IC(当日)')
        self.bt_tabs.addTab(self.tbl_bt_rolling, '策略IC滚动汇总')
        self.bt_tabs.addTab(self.tbl_bt_score, '清单得分滚动')
        self.bt_tabs.addTab(self.tbl_bt_perf, '净值/绩效')
        self.bt_tabs.addTab(self.tbl_bt_corr, '因子相关性')
        self.bt_tabs.addTab(self.tbl_bt_incic, '增量IC')
        advice_w = QWidget()
        al = QVBoxLayout(advice_w)
        # 动态权重开关
        dw_cfg = self.cfg.get('dynamic_weights', {})
        self.chk_sig_only = QCheckBox('仅对显著因子调权（Bonferroni 多重比较校正）')
        self.chk_sig_only.setChecked(bool(dw_cfg.get('auto_kill_negative', True)))
        self.chk_sig_only.setToolTip('取消勾选则全部因子按 EMA_IC 加权；勾选则不显著因子收缩至地板（不再归零）')
        self.chk_sig_only.stateChanged.connect(
            lambda s: self._set_dw('auto_kill_negative', s == 2))
        self.chk_auto_apply = QCheckBox('回测时自动应用动态权重（写入配置并刷新参数页）')
        self.chk_auto_apply.setChecked(bool(dw_cfg.get('auto_apply', False)))
        self.chk_auto_apply.stateChanged.connect(
            lambda s: self._set_dw('auto_apply', s == 2))
        al.addWidget(self.chk_sig_only)
        al.addWidget(self.chk_auto_apply)
        self.btn_apply_w = QPushButton('✅ 一键应用建议权重（覆盖当前五策略权重并保存）')
        self.btn_apply_w.clicked.connect(self.apply_suggested_weights)
        al.addWidget(self.btn_apply_w)
        self.txt_advice = QTextEdit()
        self.txt_advice.setReadOnly(True)
        al.addWidget(self.txt_advice)
        al.addWidget(QLabel('动态权重计划（shrinkage：EMA平滑+显著性+λ收缩，不再归零）：'))
        self.tbl_dyn_w = QTableWidget()
        al.addWidget(self.tbl_dyn_w)
        self.bt_tabs.addTab(advice_w, '💡 调参建议')
        lay.addWidget(self.bt_tabs)
        self.lbl_bt_foot = QLabel(
            '⚠️ 研究屏幕：回测为历史对照参考，非收益承诺。'
            '单日(n=1)无法计算 Sharpe/最大回撤，将在"净值/绩效"页标注"样本不足"；'
            '净值曲线等权、每日再平衡、初始100万仅作刻度。')
        self.lbl_bt_foot.setWordWrap(True)
        self.lbl_bt_foot.setStyleSheet('color:#b8860b;font-size:11px')
        lay.addWidget(self.lbl_bt_foot)
        self._suggested_weights = None
        return w

    def _build_config_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel('策略权重（保存后立即生效，自动归一化）：'))
        self.tbl_weights = QTableWidget()
        lay.addWidget(self.tbl_weights)
        # P0-1：待验证标签（权重/历史表现均基于有限样本，仅研究参考）
        lbl_verify = QLabel('⚠️ 待验证（回测样本有限）：本工具权重与历史表现均基于有限样本，'
                            '仅作研究参考、非收益承诺、请勿直接用于实盘决策。')
        lbl_verify.setWordWrap(True)
        lbl_verify.setStyleSheet('color:#b8860b;font-size:11px')
        lay.addWidget(lbl_verify)
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

        # T9（P0-1）：回测成本与口径 / 动态权重（新增可编辑，向后兼容）
        self.cfg_widgets = {}
        bcfg = self.cfg.get('backtest_cost', {})
        dw = self.cfg.get('dynamic_weights', {})

        def add_spin(parent, key, label, val, lo, hi, step, dec):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            sp.setValue(val)
            parent.addRow(label, sp)
            self.cfg_widgets[key] = sp

        gb_bc = QGroupBox('回测成本与口径 (backtest_cost)')
        bc_lay = QFormLayout(gb_bc)
        add_spin(bc_lay, 'bc_commission', '佣金(单边)', float(bcfg.get('commission', 0.00025)), 0, 0.01, 0.00005, 5)
        add_spin(bc_lay, 'bc_stamp_tax', '印花税(卖出)', float(bcfg.get('stamp_tax', 0.001)), 0, 0.01, 0.0005, 5)
        add_spin(bc_lay, 'bc_slippage_main', '滑点(主板单边)', float(bcfg.get('slippage_main', 0.001)), 0, 0.02, 0.0005, 5)
        add_spin(bc_lay, 'bc_slippage_20cm', '滑点(20cm单边)', float(bcfg.get('slippage_20cm', 0.002)), 0, 0.02, 0.0005, 5)
        cb_cal = QComboBox()
        cb_cal.addItems(['close', 'open'])
        cb_cal.setCurrentText(str(bcfg.get('caliber', 'close')))
        bc_lay.addRow('收益口径(caliber)', cb_cal)
        self.cfg_widgets['bc_caliber'] = cb_cal
        chk_csv = QCheckBox('导出回测CSV报告到 reports/')
        chk_csv.setChecked(bool(bcfg.get('report_csv', False)))
        bc_lay.addRow('', chk_csv)
        self.cfg_widgets['bc_report_csv'] = chk_csv
        lay.addWidget(gb_bc)

        gb_dw = QGroupBox('动态权重 (dynamic_weights · shrinkage 收缩，不再归零)')
        dw_lay = QFormLayout(gb_dw)
        add_spin(dw_lay, 'dw_ema_window', 'EMA窗口', float(dw.get('ema_window', 10)), 1, 60, 1, 0)
        add_spin(dw_lay, 'dw_shrinkage_lambda', '收缩系数λ', float(dw.get('shrinkage_lambda', 0.3)), 0, 1, 0.05, 2)
        add_spin(dw_lay, 'dw_significance_alpha', '显著性α(Bonferroni)', float(dw.get('significance_alpha', 0.05)), 0.001, 0.5, 0.005, 3)
        chk_sig = QCheckBox('仅对显著因子调权(Bonferroni 多重比较校正)')
        chk_sig.setChecked(bool(dw.get('auto_kill_negative', True)))
        dw_lay.addRow('', chk_sig)
        self.cfg_widgets['dw_auto_kill_negative'] = chk_sig
        chk_dw_en = QCheckBox('启用动态权重')
        chk_dw_en.setChecked(bool(dw.get('enabled', True)))
        dw_lay.addRow('', chk_dw_en)
        self.cfg_widgets['dw_enabled'] = chk_dw_en
        lay.addWidget(gb_dw)

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
            df_to_table(self.tbl_limit, self.result['涨停TopN'][cols], aliases=PROB_ALIASES)
            r4cols = ['排名', '代码', '名称', '涨幅', '量比', '资金流向', '涨4评分', '预估涨4概率%']
            df_to_table(self.tbl_rise4,
                        self.result['涨幅4%'][[c for c in r4cols if c in self.result['涨幅4%'].columns]],
                        aliases=PROB_ALIASES)
            mcols = cols + ['连板潜力']
            df_to_table(self.tbl_monday,
                        self.result['连板候选'][[c for c in mcols if c in self.result['连板候选'].columns]],
                        aliases=PROB_ALIASES)

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
            df_to_table(self.tbl_tail, res[[c for c in cols if c in res.columns]],
                        aliases=PROB_ALIASES)
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
            caliber = 'open' if self.radio_open.isChecked() else 'close'
            mg, summary, ic, perf = bt.backtest_one(self.db, d, lt, self.cfg, caliber=caliber)
            if summary is None:
                self.lbl_bt_summary.setText('该清单在此日匹配样本不足')
                return
            self.lbl_bt_summary.setText(' | '.join(f'{k}:{v}' for k, v in summary.items()))
            if mg is not None:
                detail_cols = [c for c in ['rank', 'code', 'name', 'prob', '实际涨幅%', '实际涨停',
                                           '得分', '口径收益%', '净收益%', '双边成本%', '无法成交']
                               if c in mg.columns]
                df_to_table(self.tbl_bt_detail, mg[detail_cols])
            if ic:
                df_to_table(self.tbl_bt_ic, pd.DataFrame(ic).T.reset_index().rename(
                    columns={'index': '策略'}))
            roll, n_days = bt.rolling_ic(self.db, lt, self.cfg)
            if not roll.empty:
                df_to_table(self.tbl_bt_rolling, roll)
            sc, _ = bt.rolling_score(self.db, lt, self.cfg)
            if not sc.empty:
                df_to_table(self.tbl_bt_score, sc)
            # ---- 净值/绩效（P0-2/P0-3）----
            curve = metrics.rolling_net_curve(self.db, lt, self.cfg, caliber=caliber)
            self._render_perf(lt, caliber, perf, curve)
            # ---- 因子相关性 + 增量IC（P0-6）----
            self._render_factor_analysis()
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
            # ---- CSV 落盘（T10，受 backtest_cost.report_csv 控制）----
            self._export_backtest_csv(d, lt, caliber, mg, summary, ic, perf, roll, n_days, curve)
            self.log(f'回测完成：{d} / {lt}（{caliber}），累计可回测 {n_days} 天，调参建议已更新')
            self.tabs.setCurrentWidget(self.tab_bt)
        except Exception:
            self.log('回测出错:\n' + traceback.format_exc())

    def _render_perf(self, list_type, caliber, perf, curve):
        """渲染净值/绩效 Tab（P0-2/P0-3）：净组合收益/Sharpe/最大回撤/胜率/基准对照/无法成交/双边成本。"""
        if perf is None:
            self.tbl_bt_perf.setRowCount(0)
            return
        rows = [
            ('口径', perf.get('caliber', caliber)),
            ('样本n', perf.get('n', 0)),
            ('可成交数', perf.get('n_tradable', 0)),
            ('净组合收益%', f"{perf.get('net_return', 0.0) * 100:.2f}%"),
            ('毛组合收益%', f"{perf.get('gross_return', 0.0) * 100:.2f}%"),
            ('基准净收益%', f"{perf.get('bench_return', 0.0) * 100:.2f}%"),
            ('基准毛收益%', f"{perf.get('bench_gross', 0.0) * 100:.2f}%"),
            ('双边成本%', f"{perf.get('cost_pct', 0.0) * 100:.3f}%"),
            ('无法成交比例', f"{perf.get('cant_trade_ratio', 0.0):.0%}"),
            ('本日胜率', f"{perf.get('win_rate', 0.0):.0%}"),
        ]
        port = (curve or {}).get('portfolio', {})
        if (curve or {}).get('sufficient'):
            rows += [
                ('累计净收益%(跨日)', f"{port.get('net_return', 0.0) * 100:.2f}%"),
                ('年化Sharpe(跨日)', '样本不足' if pd.isna(port.get('sharpe', float('nan'))) else f"{port.get('sharpe', 0.0):.2f}"),
                ('最大回撤(跨日)', f"{port.get('max_dd', 0.0) * 100:.2f}%"),
                ('跨日胜率', f"{port.get('win_rate', 0.0):.0%}"),
                ('可回测交易日', (curve or {}).get('n_days', 0)),
            ]
        else:
            rows.append(('跨日净值', (curve or {}).get('note', '样本不足（需 ≥2 个交易日）')))
        df_to_table(self.tbl_bt_perf, pd.DataFrame(rows, columns=['指标', '数值']))

    def _render_factor_analysis(self):
        """渲染因子相关性(彩色热力图) + 增量IC 两个 Tab（P0-6）。"""
        corr = metrics.factor_correlation(self.db, self.cfg, method='spearman')
        self._render_corr_table(corr)
        inc = metrics.incremental_ic(self.db, '涨停TopN', self.cfg)
        df_to_table(self.tbl_bt_incic, inc)

    def _render_corr_table(self, corr: pd.DataFrame):
        """因子相关性彩色热力图：正相关红(共线风险)/负相关蓝(对冲)/接近0白。"""
        self.tbl_bt_corr.setColumnCount(0)
        self.tbl_bt_corr.setRowCount(0)
        if corr is None or corr.empty:
            return
        factors = list(corr.columns)
        self.tbl_bt_corr.setColumnCount(len(factors) + 1)
        self.tbl_bt_corr.setRowCount(len(factors))
        self.tbl_bt_corr.setHorizontalHeaderLabels(['因子'] + factors)
        for i, r in enumerate(factors):
            self.tbl_bt_corr.setItem(i, 0, QTableWidgetItem(str(r)))
            for j, c in enumerate(factors):
                v = corr.loc[r, c]
                s = '' if pd.isna(v) else f'{v:.2f}'
                item = QTableWidgetItem(s)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if not pd.isna(v):
                    if v >= 0:
                        g = int(255 - 120 * min(1.0, v))
                        item.setBackground(QBrush(QColor(255, g, g)))
                    else:
                        vv = min(1.0, -v)
                        b2 = int(255 - 120 * vv)
                        item.setBackground(QBrush(QColor(b2, b2, 255)))
                self.tbl_bt_corr.setItem(i, j + 1, item)
        self.tbl_bt_corr.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _export_backtest_csv(self, d, list_type, caliber, mg, summary, ic, perf, roll, n_days, curve):
        """T10：回测结果落盘到 reports/（受 config.backtest_cost.report_csv 控制）。

        导出文件（均含 日期_清单_口径 戳，符合架构设计 §2）：
          bt_detail_<stamp>.csv        单日明细（含 净收益%/无法成交）
          bt_summary_<stamp>.csv       汇总（口径/成本/无法成交比例/基准净收益%）
          bt_ic_<stamp>.csv            单策略 IC
          bt_rolling_<stamp>.csv       滚动 IC（CI95+n）
          bt_net_curve_<stamp>.csv     跨日等权净值（组合/基准）
          bt_performance_<stamp>.csv   绩效指标（净/基准收益、Sharpe、最大回撤、胜率、成本、无法成交比例、n）
        """
        bcfg = self.cfg.get('backtest_cost', {})
        if not bcfg.get('report_csv', False):
            return
        rdir = os.path.join(BASE_DIR, 'reports')
        os.makedirs(rdir, exist_ok=True)
        stamp = f'{d}_{list_type}_{caliber}'
        try:
            if mg is not None and not mg.empty:
                mg.to_csv(os.path.join(rdir, f'bt_detail_{stamp}.csv'), index=False, encoding='utf-8-sig')
            if summary is not None:
                pd.DataFrame([summary]).to_csv(os.path.join(rdir, f'bt_summary_{stamp}.csv'), index=False, encoding='utf-8-sig')
            if ic:
                pd.DataFrame(ic).T.reset_index().rename(columns={'index': '策略'}).to_csv(
                    os.path.join(rdir, f'bt_ic_{stamp}.csv'), index=False, encoding='utf-8-sig')
            if roll is not None and not roll.empty:
                roll.to_csv(os.path.join(rdir, f'bt_rolling_{stamp}.csv'), index=False, encoding='utf-8-sig')
            if (curve or {}).get('sufficient'):
                pc = (curve or {}).get('portfolio', {}).get('curve')
                bc = (curve or {}).get('benchmark', {}).get('curve')
                if pc is not None and bc is not None:
                    pd.DataFrame({'净值_组合': pd.Series(pc.values, dtype='float64'),
                                 '净值_基准': pd.Series(bc.values, dtype='float64')}).to_csv(
                        os.path.join(rdir, f'bt_net_curve_{stamp}.csv'), index=False, encoding='utf-8-sig')
            # 绩效指标表（单日 perf + 跨日曲线，便于独立核验）
            port = (curve or {}).get('portfolio', {})
            suff = (curve or {}).get('sufficient')
            perf_rows = [
                ('口径', perf.get('caliber', caliber)),
                ('样本n', perf.get('n', 0)),
                ('可成交数', perf.get('n_tradable', 0)),
                ('净组合收益%', round(perf.get('net_return', 0.0) * 100, 2)),
                ('毛组合收益%', round(perf.get('gross_return', 0.0) * 100, 2)),
                ('基准净收益%', round(perf.get('bench_return', 0.0) * 100, 2)),
                ('双边成本%', round(perf.get('cost_pct', 0.0) * 100, 3)),
                ('无法成交比例', round(perf.get('cant_trade_ratio', 0.0), 4)),
                ('本日胜率', round(perf.get('win_rate', 0.0), 4)),
                ('累计净收益%(跨日)', round(port.get('net_return', 0.0) * 100, 2) if suff else '样本不足'),
                ('年化Sharpe(跨日)', round(port.get('sharpe', float('nan')), 2) if suff else '样本不足'),
                ('最大回撤(跨日)', round(port.get('max_dd', 0.0) * 100, 2) if suff else '样本不足'),
                ('跨日胜率', round(port.get('win_rate', 0.0), 4) if suff else '样本不足'),
                ('可回测交易日', (curve or {}).get('n_days', 0)),
            ]
            pd.DataFrame(perf_rows, columns=['指标', '数值']).to_csv(
                os.path.join(rdir, f'bt_performance_{stamp}.csv'), index=False, encoding='utf-8-sig')
            self.log(f'回测报告已导出到 reports/（detail/summary/ic/rolling/net_curve/performance，{stamp}）')
        except Exception:
            self.log('导出回测CSV出错:\n' + traceback.format_exc())

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

    def _sync_cfg_widgets(self):
        """把 self.cfg 的回测成本/动态权重同步到界面控件（用于「恢复默认」后刷新）。"""
        w = getattr(self, 'cfg_widgets', None)
        if not w:
            return
        bcfg = self.cfg.get('backtest_cost', {})
        dw = self.cfg.get('dynamic_weights', {})
        w['bc_commission'].setValue(float(bcfg.get('commission', 0.00025)))
        w['bc_stamp_tax'].setValue(float(bcfg.get('stamp_tax', 0.001)))
        w['bc_slippage_main'].setValue(float(bcfg.get('slippage_main', 0.001)))
        w['bc_slippage_20cm'].setValue(float(bcfg.get('slippage_20cm', 0.002)))
        w['bc_caliber'].setCurrentText(str(bcfg.get('caliber', 'close')))
        w['bc_report_csv'].setChecked(bool(bcfg.get('report_csv', False)))
        w['dw_ema_window'].setValue(float(dw.get('ema_window', 10)))
        w['dw_shrinkage_lambda'].setValue(float(dw.get('shrinkage_lambda', 0.3)))
        w['dw_significance_alpha'].setValue(float(dw.get('significance_alpha', 0.05)))
        w['dw_auto_kill_negative'].setChecked(bool(dw.get('auto_kill_negative', True)))
        w['dw_enabled'].setChecked(bool(dw.get('enabled', True)))

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
            # T9（P0-1）：回测成本 / 动态权重 落盘
            bcfg = self.cfg.setdefault('backtest_cost', {})
            bcfg['commission'] = float(self.cfg_widgets['bc_commission'].value())
            bcfg['stamp_tax'] = float(self.cfg_widgets['bc_stamp_tax'].value())
            bcfg['slippage_main'] = float(self.cfg_widgets['bc_slippage_main'].value())
            bcfg['slippage_20cm'] = float(self.cfg_widgets['bc_slippage_20cm'].value())
            bcfg['caliber'] = self.cfg_widgets['bc_caliber'].currentText()
            bcfg['report_csv'] = self.cfg_widgets['bc_report_csv'].isChecked()
            dw = self.cfg.setdefault('dynamic_weights', {})
            dw['ema_window'] = int(self.cfg_widgets['dw_ema_window'].value())
            dw['shrinkage_lambda'] = float(self.cfg_widgets['dw_shrinkage_lambda'].value())
            dw['significance_alpha'] = float(self.cfg_widgets['dw_significance_alpha'].value())
            dw['auto_kill_negative'] = self.cfg_widgets['dw_auto_kill_negative'].isChecked()
            dw['enabled'] = self.cfg_widgets['dw_enabled'].isChecked()
            strategies.save_config(BASE_DIR, self.cfg)
            self.log('参数已保存到 config.json')
            QMessageBox.information(self, '保存成功', '参数已保存，下次预测立即生效')
        except Exception as e:
            QMessageBox.critical(self, '保存失败', str(e))

    def reset_cfg(self):
        self.cfg = strategies.load_config('/nonexistent')  # 拿默认值
        strategies.save_config(BASE_DIR, self.cfg)
        self._load_weights_table()
        self._sync_cfg_widgets()
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