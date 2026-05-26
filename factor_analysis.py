import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = r'C:\Users\chloe\Desktop\quant'

# ── 读取数据 ──────────────────────────────────────────
print('Loading...')
factors = pickle.load(open(os.path.join(DATA_DIR, 'factors.pkl'), 'rb'))
close   = pickle.load(open(os.path.join(DATA_DIR, 'panel_close.pkl'), 'rb'))
print(f'因子数量: {len(factors)}')

# ── 未来5分钟收益率 ───────────────────────────────────
forward_ret = close.pct_change(5).shift(-5)

time_gaps = pd.Series(close.index).diff().dt.total_seconds().fillna(0) / 60
time_gaps.index = close.index
is_session_start = time_gaps > 10
boundary_times = close.index[is_session_start.values]
for t in boundary_times:
    idx = close.index.get_loc(t)
    start = max(0, idx - 5)
    forward_ret.iloc[start:idx] = np.nan

print(f'有效forward_ret比例: {forward_ret.notna().mean().mean():.1%}')

# ══════════════════════════════════════════════════════
# 向量化IC计算（快速版）
# ══════════════════════════════════════════════════════
def calc_daily_ic_fast(factor, forward_ret):
    dates    = pd.Series(factor.index.date, index=factor.index)
    f_ranked = factor.rank(axis=1)
    r_ranked = forward_ret.rank(axis=1)

    ic_list = []
    for date, grp_idx in dates.groupby(dates).groups.items():
        f = f_ranked.loc[grp_idx].values.flatten()
        r = r_ranked.loc[grp_idx].values.flatten()
        valid = ~(np.isnan(f) | np.isnan(r))
        if valid.sum() < 10:
            continue
        corr = np.corrcoef(f[valid], r[valid])[0, 1]
        ic_list.append((pd.Timestamp(date), corr))

    return pd.Series(dict(ic_list))

# ── 计算所有因子IC ────────────────────────────────────
print('\nCalculating IC...')
ic_results = {}
for name, factor in factors.items():
    ic   = calc_daily_ic_fast(factor, forward_ret)
    mean = ic.mean()
    std  = ic.std()
    ir   = mean / std if std > 0 else 0
    ic_results[name] = ic
    print(f'{name:20s}: IC={mean:.4f}, std={std:.4f}, IR={ir:.4f}')

# ── 筛选有效因子 ──────────────────────────────────────
print('\n====== 有效因子筛选 ======')
print('条件: |IC均值| > 0.02 且 |IR| > 0.5\n')
valid_factors = []
for name, ic in ic_results.items():
    mean = ic.mean()
    ir   = abs(mean) / ic.std()
    flag = '✓' if abs(mean) > 0.02 and ir > 0.5 else '✗'
    print(f'{flag} {name:20s}: IC={mean:.4f}, IR={ir:.4f}')
    if abs(mean) > 0.02 and ir > 0.5:
        valid_factors.append(name)

print(f'\n通过筛选的因子: {valid_factors}')

# ══════════════════════════════════════════════════════
# IC衰减
# ══════════════════════════════════════════════════════
print('\n计算IC衰减...')
decay_horizons = [1, 5, 10, 20, 30, 60]
decay_results  = {}

for name in valid_factors:
    print(f'  {name}...')
    factor = factors[name]
    decay  = []
    for h in decay_horizons:
        print(f'    horizon={h}...', end=' ')
        fwd = close.pct_change(h).shift(-h)
        fwd.iloc[-h:] = np.nan
        ic = calc_daily_ic_fast(factor, fwd)
        decay.append(ic.mean())
        print(f'IC={ic.mean():.4f}')
    decay_results[name] = decay

# ══════════════════════════════════════════════════════
# 画图
# ══════════════════════════════════════════════════════

# IC走势图
n    = len(factors)
cols = 2
rows = (n + 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
axes = axes.flatten()

for i, (name, ic) in enumerate(ic_results.items()):
    ax   = axes[i]
    mean = ic.mean()
    ir   = abs(mean) / ic.std()
    flag = '✓' if abs(mean) > 0.02 and ir > 0.5 else '✗'
    ic.plot(ax=ax, alpha=0.5, color='steelblue', linewidth=0.8)
    ic.rolling(20).mean().plot(ax=ax, color='red', linewidth=1.5, label='20d MA')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f'{flag} {name}  IC={mean:.4f}  IR={ir:.3f}')
    ax.legend(fontsize=8)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'ic_series.png'), dpi=150)
print('\nIC走势图已保存: ic_series.png')
plt.show()

# IC衰减图
if decay_results:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name, decay in decay_results.items():
        ax2.plot(decay_horizons, decay, marker='o', label=name)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('预测horizon（分钟）')
    ax2.set_ylabel('IC均值')
    ax2.set_title('IC衰减图')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'ic_decay.png'), dpi=150)
    print('IC衰减图已保存: ic_decay.png')
    plt.show()

# ── 保存结果 ──────────────────────────────────────────
pickle.dump(ic_results,    open(os.path.join(DATA_DIR, 'ic_results.pkl'),    'wb'))
pickle.dump(valid_factors, open(os.path.join(DATA_DIR, 'valid_factors.pkl'), 'wb'))
print('\n结果已保存')