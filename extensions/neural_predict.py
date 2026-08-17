# -*- coding: utf-8 -*-
"""神经网络预测（Neural Predict）
==============================
独立的神经网络选股能力。与现有「五因子涨停模型 / 抄底 / 反转 / 防御」完全解耦：

  - 数据自举：用 AKShare 历史日 K 线（stock_zh_a_hist）为股票池构建样本，
    特征全部来自「价格/量能/波动」等通用量价指标，不依赖封流比等涨停细节字段。
  - 标签：次日涨跌幅（回归）；再由 sigmoid 映射为「次日上涨概率」与 0~100 评分。
  - 模型：默认 sklearn MLPRegressor（多层感知机，即前馈神经网络）；
    可选 backend='torch' 使用 PyTorch 深网。
  - 存储：模型与历史单独存 nn_model.db，不污染主 data.db；可通过回测同口径评估。

为什么要「独立」：现有五因子是人工加权线性组合，容易在单一市场风格下团灭；
神经网络从量价序列里学非线性映射，是一个正交的、数据驱动的预测视角。两者并列，
由回测（IC / 动态权重）决定每天该信谁。

训练数据来源说明：
  - 本地 data/ 仅 2 天快照，不足以训练，因此训练必须外援 AKShare 历史行情。
  - 首次训练会拉取股票池历史（一次，较慢，结果缓存到 nn_model.db），之后每日
    增量更新，推理只需读本地缓存，秒级。
"""
import os
import sqlite3
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from extensions.base import Extension, Param


# ----------------------------------------------------------------------------
# 特征工程（与数据源无关的纯量价特征）
# ----------------------------------------------------------------------------
def compute_features(bars):
    """bars: DataFrame，列含 close/open/high/low/vol/amount/pct，按日期升序。
    返回特征 dict（最新一根 bar 处的截面特征）；不足则返回 None。"""
    if bars is None or len(bars) < 21:
        return None
    close = bars['close'].astype(float).values
    high = bars['high'].astype(float).values
    low = bars['low'].astype(float).values
    op = bars['open'].astype(float).values
    vol = bars['vol'].astype(float).values
    amt = bars['amount'].astype(float).values
    n = len(close)
    # 日收益率序列
    ret = np.diff(close) / close[:-1]
    if len(ret) < 5:
        return None

    def safe_ratio(i, j):
        return (close[i] / close[j] - 1.0) if close[j] != 0 else 0.0

    r1 = safe_ratio(n - 1, n - 2)
    r5 = safe_ratio(n - 1, max(n - 6, 0))
    r10 = safe_ratio(n - 1, max(n - 11, 0))
    r20 = safe_ratio(n - 1, max(n - 21, 0))
    r60 = safe_ratio(n - 1, max(n - 61, 0)) if n > 61 else r20

    vol_20 = float(np.std(ret[-20:])) if len(ret) >= 20 else float(np.std(ret))
    mean_vol = float(np.mean(vol[-20:])) if len(vol) >= 20 else float(np.mean(vol))
    vol_ratio = float(vol[-1] / mean_vol) if mean_vol > 0 else 1.0
    gap = float(op[-1] / close[-2] - 1.0) if close[-2] != 0 else 0.0
    rng = (high[-1] - low[-1])
    range_pos = float((close[-1] - low[-1]) / rng) if rng > 0 else 0.5
    # 成交额对数 z-score（近 60 日）
    lamt = np.log(amt + 1.0)
    w = lamt[-60:] if len(lamt) >= 60 else lamt
    amt_z = float((lamt[-1] - w.mean()) / (w.std() + 1e-9))
    # 近 20 日上涨天数占比
    up_ratio = float(np.mean(ret[-20:] > 0)) if len(ret) >= 20 else 0.5
    # 近期动量（5 日均线 / 20 日均线 - 1）
    ma5 = close[-5:].mean()
    ma20 = close[-20:].mean()
    ma_ratio = float(ma5 / ma20 - 1.0) if ma20 != 0 else 0.0

    return {
        'ret_1': r1, 'ret_5': r5, 'ret_10': r10, 'ret_20': r20, 'ret_60': r60,
        'vol_20': vol_20, 'vol_ratio': vol_ratio, 'gap': gap, 'range_pos': range_pos,
        'amt_z': amt_z, 'up_ratio': up_ratio, 'ma_ratio': ma_ratio,
    }


FEATURE_NAMES = ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_60', 'vol_20',
                 'vol_ratio', 'gap', 'range_pos', 'amt_z', 'up_ratio', 'ma_ratio']


# ----------------------------------------------------------------------------
# 神经网络引擎（独立存储 nn_model.db）
# ----------------------------------------------------------------------------
class NeuralEngine:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS nn_history(
            date TEXT, code TEXT, open REAL, close REAL, high REAL, low REAL,
            vol REAL, amount REAL, pct REAL,
            PRIMARY KEY(date, code));
        CREATE TABLE IF NOT EXISTS nn_model(
            id INTEGER PRIMARY KEY CHECK (id=1),
            backend TEXT, scaler_blob BLOB, model_blob BLOB,
            meta TEXT, trained_at TEXT, n_samples INTEGER, feature_names TEXT);
        """)
        self.conn.commit()

    # ---- 历史数据自举/更新（AKShare）----
    def bootstrap(self, codes, lookback_days=250, progress=None):
        import akshare as ak
        import time
        start = (datetime.now() - pd.Timedelta(days=lookback_days + 30)).strftime('%Y%m%d')
        end = datetime.now().strftime('%Y%m%d')
        total = len(codes)
        done = 0
        for i, code in enumerate(codes):
            ok = False
            for attempt in range(3):      # 单股重试，应对东财偶发限速/中断
                try:
                    df = ak.stock_zh_a_hist(symbol=str(code), period='daily',
                                            start_date=start, end_date=end, adjust='')
                    if df is not None and not df.empty:
                        rows = []
                        for _, r in df.iterrows():
                            d = str(r['日期']).replace('-', '')
                            o = float(r['开盘']); c = float(r['收盘'])
                            h = float(r['最高']); l = float(r['最低'])
                            v = float(r['成交量']); a = float(r['成交额'])
                            p = float(r['涨跌幅'])
                            rows.append((d, code, o, c, h, l, v, a, p))
                        self.conn.executemany(
                            'INSERT OR REPLACE INTO nn_history VALUES(?,?,?,?,?,?,?,?,?)', rows)
                        self.conn.commit()
                        ok = True
                        break
                except Exception:
                    time.sleep(1.0 * (attempt + 1))
            done += 1
            if progress:
                progress(done, total, code if ok else f'{code}(失败)')

    def update_latest(self, codes, progress=None):
        """增量追加每个股票最近 5 个交易日（用于每日更新）。"""
        import akshare as ak
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - pd.Timedelta(days=10)).strftime('%Y%m%d')
        done = 0
        for code in codes:
            try:
                df = ak.stock_zh_a_hist(symbol=str(code), period='daily',
                                        start_date=start, end_date=end, adjust='')
                if df is not None and not df.empty:
                    rows = [(str(r['日期']).replace('-', ''), code, float(r['开盘']),
                             float(r['收盘']), float(r['最高']), float(r['最低']),
                             float(r['成交量']), float(r['成交额']), float(r['涨跌幅']))
                            for _, r in df.iterrows()]
                    self.conn.executemany(
                        'INSERT OR REPLACE INTO nn_history VALUES(?,?,?,?,?,?,?,?,?)', rows)
                    self.conn.commit()
            except Exception:
                pass
            done += 1
            if progress:
                progress(done, len(codes), code)

    def history_for(self, code):
        rows = self.conn.execute(
            'SELECT date,open,close,high,low,vol,amount,pct FROM nn_history WHERE code=? ORDER BY date',
            (code,)).fetchall()
        if not rows:
            return None
        return pd.DataFrame(rows, columns=['date', 'open', 'close', 'high', 'low', 'vol', 'amount', 'pct'])

    def universe(self):
        return [r[0] for r in self.conn.execute(
            'SELECT DISTINCT code FROM nn_history').fetchall()]

    # ---- 数据集构建 ----
    def build_dataset(self, min_hist=60):
        X, y, codes = [], [], []
        for code in self.universe():
            bars = self.history_for(code)
            if bars is None or len(bars) < min_hist + 1:
                continue
            # 逐日滑动构造样本：t 处特征 → t+1 处 pct 标签
            for t in range(min_hist - 1, len(bars) - 1):
                win = bars.iloc[:t + 1]
                feat = compute_features(win)
                if feat is None:
                    continue
                label = float(bars.iloc[t + 1]['pct'])
                X.append([feat[k] for k in FEATURE_NAMES])
                y.append(label)
                codes.append(code)
        if not X:
            return None, None, None
        return np.array(X, dtype=float), np.array(y, dtype=float), codes

    # ---- 训练 ----
    def train(self, X, y, cfg, progress=None):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        backend = cfg.get('backend', 'sklearn')
        if backend == 'torch':
            model_blob, meta = self._train_torch(Xs, y, cfg, progress)
        else:
            model_blob, meta = self._train_sklearn(Xs, y, cfg, progress)
        blob = pickle.dumps(scaler)
        self.conn.execute('DELETE FROM nn_model')
        self.conn.execute(
            'INSERT INTO nn_model VALUES(1,?,?,?,?,?,?,?)',
            (backend, blob, model_blob, meta, datetime.now().strftime('%Y-%m-%d %H:%M'),
             len(X), ','.join(FEATURE_NAMES)))
        self.conn.commit()

    def _train_sklearn(self, Xs, y, cfg, progress):
        from sklearn.neural_network import MLPRegressor
        hidden = tuple(int(x) for x in str(cfg.get('hidden', '64,32')).split(',') if x)
        model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=int(cfg.get('epochs', 300)),
                             alpha=float(cfg.get('alpha', 1e-4)), random_state=42,
                             early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)
        model.fit(Xs, y)
        return pickle.dumps(model), f'sklearn MLP hidden={hidden}'

    def _train_torch(self, Xs, y, cfg, progress):
        import io
        import torch
        import torch.nn as nn
        hidden = tuple(int(x) for x in str(cfg.get('hidden', '64,32')).split(',') if x)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class MLP(nn.Module):
            def __init__(self, nin, hs):
                super().__init__()
                layers = []
                d = nin
                for h in hs:
                    layers += [nn.Linear(d, h), nn.ReLU()]
                    d = h
                layers.append(nn.Linear(d, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x).squeeze(-1)

        xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        model = MLP(Xs.shape[1], hidden).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get('lr', 1e-3)))
        lossf = nn.MSELoss()
        bs = int(cfg.get('batch', 256))
        ep = int(cfg.get('epochs', 60))
        for e in range(ep):
            model.train()
            perm = torch.randperm(xt.shape[0])
            for i in range(0, xt.shape[0], bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(model(xt[idx].to(dev)), yt[idx].to(dev))
                loss.backward()
                opt.step()
            if progress and (e + 1) % 10 == 0:
                progress(e + 1, ep, 'torch')
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue(), f'torch MLP hidden={hidden} dev={dev.type}'

    def has_model(self):
        return self.conn.execute('SELECT 1 FROM nn_model WHERE id=1').fetchone() is not None

    def load_model(self):
        row = self.conn.execute(
            'SELECT backend, scaler_blob, model_blob, feature_names FROM nn_model WHERE id=1').fetchone()
        if not row:
            return None
        backend, scaler_blob, model_blob, fnames = row
        scaler = pickle.loads(scaler_blob)
        return backend, scaler, model_blob, fnames.split(',')

    def predict_features(self, X_df):
        """X_df: DataFrame，列为 FEATURE_NAMES。返回预测次日涨幅 ndarray。"""
        loaded = self.load_model()
        if loaded is None:
            raise RuntimeError('模型未训练')
        backend, scaler, model_blob, _ = loaded
        Xs = scaler.transform(X_df[FEATURE_NAMES].values.astype(float))
        if backend == 'torch':
            import io, torch
            import torch.nn as nn

            class MLP(nn.Module):
                def __init__(self, nin, hs):
                    super().__init__()
                    layers = []
                    d = nin
                    for h in hs:
                        layers += [nn.Linear(d, h), nn.ReLU()]
                        d = h
                    layers.append(nn.Linear(d, 1))
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x).squeeze(-1)

            hidden = tuple(int(x.strip()) for x in str(self._hidden_from_meta()).split(',')
                           if x.strip().isdigit())
            model = MLP(Xs.shape[1], hidden)
            model.load_state_dict(torch.load(io.BytesIO(model_blob), map_location='cpu',
                                              weights_only=True))
            model.eval()
            with torch.no_grad():
                out = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
        else:
            import pickle
            model = pickle.loads(model_blob)
            out = model.predict(Xs)
        return out

    def _hidden_from_meta(self):
        row = self.conn.execute('SELECT meta FROM nn_model WHERE id=1').fetchone()
        # meta 形如 'sklearn MLP hidden=(64, 32)' 或 'torch MLP hidden=(64, 32) dev=cuda'
        try:
            s = row[0].split('hidden=')[1].strip()   # '(64, 32) dev=cuda'
            # 只取第一个括号内的内容（落到第一个 ')' 为止），避开末尾的 ' dev=cuda' 等后缀
            start = s.find('(')
            end = s.find(')', start)
            if start != -1 and end != -1:
                inner = s[start + 1:end]
                return inner.replace(' ', '')    # '64,32'
        except Exception:
            pass
        return '64,32'


# ----------------------------------------------------------------------------
# 扩展：接入框架（自动出现在 GUI + 回测）
# ----------------------------------------------------------------------------
class NeuralPredictExtension(Extension):
    key = 'neural_predict'
    name = '神经网络预测'
    description = '独立神经网络：从量价序列学习次日涨跌映射，与五因子体系正交'
    datasource = 'akshare'          # 训练/推理特征均来自 AKShare 历史与快照
    list_type = '神经网络预测'
    prob_col = '上涨概率%'
    score_col = 'NN评分'
    # 目标：次日上涨且有一定幅度。涨≥2% 合格，涨≥5% 满分；下跌扣分
    scoring = [[-99, -80], [-4, -50], [-2, -25], [0, 20], [2, 60], [4, 80], [6, 95], [9.5, 100]]
    hit_threshold = 2.0
    detail_keys = ['NN预测涨幅%', '上涨概率%', 'NN评分', '量能比', '下影线比', '近期动量']
    supports_training = True       # 框架据此显示「训练/更新模型」按钮

    params = [
        Param('universe_source', '股票池来源', 'choice', 'hs300', choices=['hs300', 'zz500', 'imported'],
              help='hs300=沪深300成分(通用性强); zz500=中证500; imported=用本地data中的代码'),
        Param('lookback_days', '训练回看天数', 'int', 250, 60, 1200, step=30),
        Param('max_universe', '股票池上限(控制首训时长)', 'int', 150, 20, 500, step=10),
        Param('min_hist', '单股最少历史交易日', 'int', 60, 30, 250, step=10),
        Param('backend', '神经网络后端', 'choice', 'sklearn', choices=['sklearn', 'torch'],
              help='sklearn=多层感知机(快/稳); torch=PyTorch深网(需GPU更佳)'),
        Param('hidden', '隐藏层结构', 'str', '64,32', help='逗号分隔，如 64,32 或 128,64,32'),
        Param('epochs', '训练轮数', 'int', 300, 20, 2000, step=20),
        Param('auto_train', '无模型时自动训练', 'bool', True, help='运行前若模型不存在则先训练(首训较慢)'),
        Param('fetch_missing', '推理时补全未知股票', 'bool', False,
              help='对快照中不在股票池的股票临时拉历史(慢，默认关)'),
        Param('max_predict_calls', '推理最大补拉数', 'int', 50, 0, 300, step=10),
        Param('top_n', '输出数量', 'int', 30, 5, 200, step=5),
    ]

    def _universe_codes(self, cfg, ctx):
        src = cfg.get('universe_source', 'hs300')
        if src == 'imported':
            try:
                import glob, os
                codes = set()
                for f in glob.glob(os.path.join(ctx.data_dir, '*.xlsx')):
                    try:
                        d = __import__('core.loader', fromlist=['read_any']).read_any(f)
                        codes.update(d['代码6'].astype(str).str.extract(r'(\d{6})')[0].dropna())
                    except Exception:
                        pass
                return [c for c in codes if c][:int(cfg['max_universe'])]
            except Exception:
                return []
        # hs300 / zz500
        try:
            import akshare as ak
            idx = '000300' if src == 'hs300' else '000905'
            df = ak.index_stock_cons_csindex(symbol=idx)
            codes = df['成分券代码'].astype(str).str.extract(r'(\d{6})')[0].dropna().tolist() \
                if '成分券代码' in df.columns else df.iloc[:, 0].astype(str).str.extract(r'(\d{6})')[0].dropna().tolist()
            return codes[:int(cfg['max_universe'])]
        except Exception:
            return []

    def train(self, ctx, cfg, progress=None):
        """显式训练：拉历史 → 建数据集 → 训练 → 存 nn_model.db。返回 (ok, msg)。"""
        db_path = os.path.join(ctx.base_dir, 'nn_model.db')
        eng = NeuralEngine(db_path)
        codes = self._universe_codes(cfg, ctx)
        if not codes:
            return False, '无法获取股票池（检查联网 / universe_source）'
        msg = []
        eng.bootstrap(codes, lookback_days=int(cfg['lookback_days']),
                      progress=progress if progress else (lambda d, t, c: msg.append(f'拉历史 {d}/{t} {c}')))
        X, y, _ = eng.build_dataset(min_hist=int(cfg['min_hist']))
        if X is None or len(X) < 200:
            return False, f'样本不足（{0 if X is None else len(X)} 条），请增大 lookback_days 或股票池'
        eng.train(X, y, cfg)
        return True, f'训练完成：{len(X)} 样本 / 股票池 {len(codes)} / 后端 {cfg.get("backend","sklearn")}'

    def run(self, df, cfg, ctx):
        from extensions.base import normalize_market_cols
        d = normalize_market_cols(df)
        db_path = os.path.join(ctx.base_dir, 'nn_model.db')
        eng = NeuralEngine(db_path)
        if not eng.has_model():
            if cfg.get('auto_train', True):
                ok, msg = self.train(ctx, cfg)
                if not ok:
                    return {'主表': pd.DataFrame(), 'note': f'自动训练失败：{msg}', 'tables': {}}
            else:
                return {'主表': pd.DataFrame(),
                        'note': '模型尚未训练，请点「🧠 训练/更新模型」或开启 auto_train', 'tables': {}}

        # 为快照中每只股票构建特征（优先用缓存历史 + 当日快照作为最新 bar）
        snap_idx = {c: i for i, c in enumerate(d['代码6'].astype(str))}
        feat_rows, keep_codes, keep_idx = [], [], []
        universe = set(eng.universe())
        fetched = 0
        for code in d['代码6'].astype(str):
            bars = eng.history_for(code)
            today = d.iloc[snap_idx[code]] if code in snap_idx else None
            if bars is None or len(bars) < int(cfg['min_hist']):
                # 不在股票池：按需补拉
                if cfg.get('fetch_missing', False) and fetched < int(cfg['max_predict_calls']):
                    try:
                        import akshare as ak
                        h = ak.stock_zh_a_hist(symbol=code, period='daily',
                                               start_date=(datetime.now() - pd.Timedelta(
                                                   days=int(cfg['min_hist']) + 10)).strftime('%Y%m%d'),
                                               end_date=datetime.now().strftime('%Y%m%d'), adjust='')
                        if h is not None and not h.empty:
                            bars = pd.DataFrame({
                                'date': h['日期'].astype(str).str.replace('-', '', regex=False),
                                'open': h['开盘'], 'close': h['收盘'], 'high': h['最高'],
                                'low': h['最低'], 'vol': h['成交量'], 'amount': h['成交额'],
                                'pct': h['涨跌幅']})
                            fetched += 1
                    except Exception:
                        bars = None
                if bars is None or len(bars) < int(cfg['min_hist']):
                    continue
            # 用当日快照覆盖/追加最新 bar（让特征反映今日）
            if today is not None:
                new_row = pd.DataFrame([{
                    'date': str(getattr(ctx, 'target_date', None) or datetime.now().strftime('%Y%m%d')),
                    'open': float(today.get('开盘', np.nan)),
                    'close': float(today.get('现价', np.nan)),
                    'high': float(today.get('最高', np.nan)),
                    'low': float(today.get('最低', np.nan)),
                    'vol': float(today.get('成交量', 0) or 0),
                    'amount': float(today.get('成交额', 0) or 0),
                    'pct': float(today.get('涨幅', 0) or 0),
                }])
                bars = pd.concat([bars, new_row], ignore_index=True)
            feat = compute_features(bars)
            if feat is None:
                continue
            feat_rows.append([feat[k] for k in FEATURE_NAMES])
            keep_codes.append(code)
            keep_idx.append(snap_idx[code])

        if not feat_rows:
            return {'主表': pd.DataFrame(),
                    'note': '当前快照中无股票池覆盖的股票（先训练股票池，或开启 fetch_missing）', 'tables': {}}

        X_df = pd.DataFrame(feat_rows, columns=FEATURE_NAMES)
        pred_ret = eng.predict_features(X_df)  # 预测次日涨幅%

        out = d.iloc[keep_idx].copy()
        out['NN预测涨幅%'] = np.round(pred_ret, 2)
        # sigmoid 映射为上涨概率（以 2% 为中点）
        out['上涨概率%'] = np.round(100 / (1 + np.exp(-(pred_ret - 0.5) / 1.5)), 1)
        out['NN评分'] = np.clip(out['上涨概率%'], 5, 98)
        out['量能比'] = out['量比']
        rng = (out['最高'] - out['最低'])
        out['下影线比'] = np.where(rng > 0, (out['现价'] - out['最低']) / rng, 0.5)
        out['近期动量'] = out['20日涨幅']
        out = out.sort_values('NN预测涨幅%', ascending=False).head(int(cfg['top_n']))
        out.insert(0, '排名', range(1, len(out) + 1))

        cols = ['排名', '代码6', '名称', '涨幅', '20日涨幅', 'NN预测涨幅%', '上涨概率%',
                'NN评分', '量能比', '下影线比', '量能分' if '量能分' in out else '换手']
        cols = [c for c in cols if c in out.columns]
        main = out[cols].reset_index(drop=True)
        note = (f"神经网络预测 {len(main)} 只（股票池 {len(universe)} 只命中 {len(keep_codes)} 只）| "
                f"后端 {eng.load_model()[0] if eng.has_model() else '?'} | "
                f"平均预测次日涨幅 {out['NN预测涨幅%'].mean():.2f}%")
        return {'主表': main, 'note': note, 'tables': {}}
