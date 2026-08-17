# -*- coding: utf-8 -*-
"""神经网络模块离线回归测试（不联网）：
1) _hidden_from_meta 对 sklearn / torch 两种 meta 字符串的解析
2) NeuralEngine 核心：建库 -> build_dataset -> train(sklearn/torch) -> predict_features
3) NeuralPredictExtension.run 端到端（sklearn / torch 两种模型）
全部用合成历史数据，避免依赖 AKShare 网络。
"""
import os, sys, tempfile, types
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from extensions.neural_predict import (
    NeuralEngine, NeuralPredictExtension, FEATURE_NAMES,
)

PASS, FAIL = [], []
def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

# ---------------------------------------------------------------------------
# 0. meta 解析（torch 修复点：'dev=cuda' 后缀不应破坏元组解析）
# ---------------------------------------------------------------------------
eng = NeuralEngine.__new__(NeuralEngine)
def fake_meta(row_val):
    class _Cursor:
        def fetchone(self):
            return (row_val,)
    class _Conn:
        def execute(self, q):
            return _Cursor()
    eng.conn = _Conn()
    return eng._hidden_from_meta()
check('meta sklearn', fake_meta('sklearn MLP hidden=(64, 32)') == '64,32',
      f"-> {fake_meta('sklearn MLP hidden=(64, 32)')}")
check('meta torch(cuda后缀)', fake_meta('torch MLP hidden=(64, 32) dev=cuda') == '64,32',
      f"-> {fake_meta('torch MLP hidden=(64, 32) dev=cuda')}")
check('meta torch(cpu后缀)', fake_meta('torch MLP hidden=(128, 64, 32) dev=cpu') == '128,64,32',
      f"-> {fake_meta('torch MLP hidden=(128, 64, 32) dev=cpu')}")

# ---------------------------------------------------------------------------
# 1. 合成历史数据
# ---------------------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix='nn_test_')
db_path = os.path.join(tmp, 'nn_model.db')
engine = NeuralEngine(db_path)

rng = np.random.default_rng(42)
N_CODES, N_BARS = 30, 150
CODES = [f'{600000+i:06d}' for i in range(N_CODES)]
rows = []
for code in CODES:
    close = 10.0
    for t in range(N_BARS):
        ret = rng.normal(0.0005, 0.02)
        close *= (1 + ret)
        op = close * (1 + rng.normal(0, 0.005))
        hi = max(close, op) * (1 + abs(rng.normal(0, 0.01)))
        lo = min(close, op) * (1 - abs(rng.normal(0, 0.01)))
        vol = float(rng.integers(1e5, 5e5))
        amt = vol * close
        date = f'2025{1+(t//28):02d}{(t%28)+1:02d}'
        rows.append((date, code, float(op), float(close), float(hi), float(lo), vol, amt, float(ret*100)))
engine.conn.executemany(
    'INSERT OR REPLACE INTO nn_history VALUES(?,?,?,?,?,?,?,?,?)', rows)
engine.conn.commit()
check('universe 数量', len(engine.universe()) == N_CODES, f"-> {len(engine.universe())}")

# ---------------------------------------------------------------------------
# 2. build_dataset
# ---------------------------------------------------------------------------
X, y, codes = engine.build_dataset(min_hist=60)
check('build_dataset 样本充足', X is not None and len(X) > 200, f"-> {0 if X is None else len(X)} 样本")
check('特征数=12', X is not None and X.shape[1] == 12, f"-> {None if X is None else X.shape[1]}")

# ---------------------------------------------------------------------------
# 3. sklearn 训练 + 推理
# ---------------------------------------------------------------------------
engine.train(X, y, {'backend': 'sklearn', 'hidden': '64,32', 'epochs': 50})
check('sklearn 已存模型', engine.has_model())
check('sklearn backend 标记', engine.load_model()[0] == 'sklearn')
X_df = pd.DataFrame(X[:10], columns=FEATURE_NAMES)
pred_sk = engine.predict_features(X_df)
check('sklearn 推理形状', pred_sk.shape == (10,), f"-> {pred_sk.shape}")

# ---------------------------------------------------------------------------
# 4. torch 训练 + 推理（触发修复后的 meta 解析路径）
# ---------------------------------------------------------------------------
engine.train(X, y, {'backend': 'torch', 'hidden': '64,32', 'epochs': 5})
check('torch backend 标记', engine.load_model()[0] == 'torch')
pred_to = engine.predict_features(X_df)
check('torch 推理形状', pred_to.shape == (10,), f"-> {pred_to.shape}")
check('torch 预测为有限值', np.all(np.isfinite(pred_to)))

# ---------------------------------------------------------------------------
# 5. Extension.run 端到端（sklearn / torch 模型）
# ---------------------------------------------------------------------------
ext = NeuralPredictExtension()
cfg = ext.effective_params({'neural_predict': {
    'min_hist': 60, 'top_n': 20, 'auto_train': False, 'fetch_missing': False}}, {})
ctx = types.SimpleNamespace(db=None, cfg={}, base_dir=tmp, data_dir=tmp, target_date='20251231')
snap = pd.DataFrame({
    '代码6': CODES, '名称': [f'股{i}' for i in range(N_CODES)],
    '涨幅': rng.normal(0, 1, N_CODES), '现价': rng.uniform(8, 20, N_CODES),
    '昨收': rng.uniform(8, 20, N_CODES), '开盘': rng.uniform(8, 20, N_CODES),
    '最高': rng.uniform(10, 22, N_CODES), '最低': rng.uniform(7, 19, N_CODES),
    '换手': rng.uniform(1, 5, N_CODES), '量比': rng.uniform(0.8, 2.5, N_CODES),
    '20日涨幅': rng.normal(2, 3, N_CODES),
    '成交量': rng.uniform(1e5, 5e5, N_CODES), '成交额': rng.uniform(1e8, 5e8, N_CODES),
})
need_cols = ['排名', '代码6', '名称', 'NN预测涨幅%', '上涨概率%', 'NN评分']
for backend in ('sklearn', 'torch'):
    engine.train(X, y, {'backend': backend, 'hidden': '64,32',
                        'epochs': 50 if backend == 'sklearn' else 5})
    c = dict(cfg); c['backend'] = backend
    res = ext.run(snap, c, ctx)
    main = res.get('主表')
    check(f'run({backend}) 主表非空', main is not None and not main.empty,
          f"-> {0 if main is None else len(main)} 行")
    check(f'run({backend}) 含关键列',
          main is not None and all(c2 in main.columns for c2 in need_cols))

print('\n==== NN 回归汇总 ====')
print(f'PASS: {len(PASS)}  FAIL: {len(FAIL)}')
if FAIL:
    print('失败项:', FAIL); sys.exit(1)
print('全部通过 ✅')
