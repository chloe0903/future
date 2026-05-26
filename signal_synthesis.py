import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = r'C:\Users\chloe\Desktop\quant'

# ── 读取数据 ──────────────────────────────────────────
print('Loading...')
factors      = pickle.load(open(os.path.join(DATA_DIR, 'factors.pkl'),       'rb'))
ic_results   = pickle.load(open(os.path.join(DATA_DIR, 'ic_results.pkl'),    'rb'))
valid_factors = pickle.load(open(os.path.join(DATA_DIR, 'valid_factors.pkl'),'rb'))

print(f'有效因子: {valid_factors}')

# ── 只用rank版本的三个独立因子 ────────────────────────
# mom_5m和rank_mom_5m本质相同，只保留rank版本
use_factors = ['rank_mom_5m', 'rank_mom_30m', 'rank_vwap_dev']

# ── 计算每个因子的IR权重 ──────────────────────────────
ir_weights = {}
for name in use_factors:
    ic   = ic_results[name]
    mean = ic.mean()
    std  = ic.std()
    ir   = abs(mean) / std
    ir_weights[name] = ir

# 归一化权重，加起来=1
total = sum(ir_weights.values())
weights = {k: v/total for k, v in ir_weights.items()}

print('\n====== 因子权重 ======')
for name, w in weights.items():
    print(f'  {name:20s}: {w:.1%}')

# ── 合成综合Alpha信号 ─────────────────────────────────
print('\n合成Alpha信号...')

# 加权合成：每个时间点的综合信号 = 各因子值 × 对应权重
alpha_raw = sum(factors[name] * w for name, w in weights.items())

print(f'Alpha形状: {alpha_raw.shape}')
print(f'Alpha样例:\n{alpha_raw.tail(3).round(3)}')

# ── 平滑处理：3期移动平均 ─────────────────────────────
# 降低噪音，避免信号频繁翻转
alpha_smooth = alpha_raw.rolling(3).mean()

print(f'\n平滑后Alpha样例:\n{alpha_smooth.tail(3).round(3)}')

# ── 对比平滑前后 ──────────────────────────────────────
print('\n====== 平滑效果对比 ======')
print(f'原始信号标准差:   {alpha_raw.std().mean():.4f}')
print(f'平滑后信号标准差: {alpha_smooth.std().mean():.4f}')
print(f'标准差降低:       {(1 - alpha_smooth.std().mean()/alpha_raw.std().mean()):.1%}')

# ── 画图：某一天的信号分布 ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：原始信号 vs 平滑信号（以RB为例）
ax = axes[0]
sample = alpha_raw['RB'].iloc[:500]
sample_smooth = alpha_smooth['RB'].iloc[:500]
ax.plot(sample.values, alpha=0.5, color='steelblue', linewidth=0.8, label='Raw')
ax.plot(sample_smooth.values, color='red', linewidth=1.5, label='Smoothed (3MA)')
ax.set_title('RB Alpha Signal — Raw vs Smoothed', fontsize=13)
ax.set_xlabel('Time steps')
ax.set_ylabel('Alpha')
ax.legend()
ax.grid(alpha=0.3)

# 右图：某时间点的截面信号分布
ax = axes[1]
snapshot = alpha_smooth.iloc[-1].sort_values()
colors = ['#D62728' if v < 0 else '#2CA02C' for v in snapshot.values]
ax.bar(snapshot.index, snapshot.values, color=colors)
ax.axhline(0, color='black', linewidth=1.0)
ax.set_title('Latest Alpha Signal by Symbol', fontsize=13)
ax.set_ylabel('Alpha Score')
ax.grid(axis='y', alpha=0.3)
for i, (sym, val) in enumerate(snapshot.items()):
    ax.text(i, val + 0.02 * np.sign(val), f'{val:.2f}',
            ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'alpha_signal.png'), dpi=150, bbox_inches='tight')
print('\n图已保存: alpha_signal.png')
plt.show()

# ── 保存信号 ──────────────────────────────────────────
pickle.dump(alpha_smooth, open(os.path.join(DATA_DIR, 'alpha_signal.pkl'), 'wb'))
print('alpha_signal.pkl 保存完成')