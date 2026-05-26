import pickle
import pandas as pd
import numpy as np
import os

DATA_DIR = r'C:\Users\chloe\Desktop\quant'

close  = pickle.load(open(os.path.join(DATA_DIR, 'panel_close.pkl'), 'rb'))
volume = pickle.load(open(os.path.join(DATA_DIR, 'panel_volume.pkl'), 'rb'))

close  = close['2022-01-01':'2025-05-31']
volume = volume['2022-01-01':'2025-05-31']

# ── 测试不同因子的IC ──────────────────────────────────
fwd_30m = close.pct_change(30).shift(-30)  # 未来30分钟收益率

def cs_normalize(df):
    mean = df.mean(axis=1)
    std  = df.std(axis=1)
    return df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

def calc_ic(factor, forward_ret, sample_every=30):
    dates = pd.Series(factor.index.date, index=factor.index).unique()
    ic_list = []
    for date in dates:
        mask = factor.index.date == date
        f = factor[mask].values.flatten()
        r = forward_ret[mask].values.flatten()
        valid = ~(np.isnan(f) | np.isnan(r))
        if valid.sum() < 10:
            continue
        corr = np.corrcoef(
            pd.Series(f[valid]).rank(),
            pd.Series(r[valid]).rank()
        )[0, 1]
        ic_list.append((pd.Timestamp(date), corr))
    return pd.Series(dict(ic_list))

print('测试各因子对未来30分钟收益率的预测力:\n')

# 1. 短期反转（5分钟）
mom_5m = cs_normalize(close.pct_change(5))
ic = calc_ic(mom_5m, fwd_30m)
print(f'mom_5m    反转: IC={ic.mean():.4f}, IR={abs(ic.mean())/ic.std():.3f}')

# 2. 中期反转（30分钟）
mom_30m = cs_normalize(close.pct_change(30))
ic = calc_ic(mom_30m, fwd_30m)
print(f'mom_30m   反转: IC={ic.mean():.4f}, IR={abs(ic.mean())/ic.std():.3f}')

# 3. 日内趋势（今日开盘到现在）
daily_open = close.groupby(close.index.date).transform('first')
intraday = cs_normalize((close - daily_open) / daily_open)
ic = calc_ic(intraday, fwd_30m)
print(f'intraday  趋势: IC={ic.mean():.4f}, IR={abs(ic.mean())/ic.std():.3f}')

# 4. 隔夜跳空
daily_first = close.groupby(close.index.date).first()
daily_last  = close.groupby(close.index.date).last()
gap_daily   = daily_first / daily_last.shift(1) - 1
gap = close.copy()
for date in gap_daily.index:
    mask = close.index.date == date
    gap.loc[mask] = gap_daily.loc[date].values
gap_cs = cs_normalize(gap)
ic = calc_ic(gap_cs, fwd_30m)
print(f'overnight 跳空: IC={ic.mean():.4f}, IR={abs(ic.mean())/ic.std():.3f}')

# 5. 中长期趋势（过去1天收益率）
mom_1d = cs_normalize(close.pct_change(240))
ic = calc_ic(mom_1d, fwd_30m)
print(f'mom_1d    趋势: IC={ic.mean():.4f}, IR={abs(ic.mean())/ic.std():.3f}')

# 6. 过去5天趋势
mom_5d = cs_normalize(close.pct_change(240*5))
ic = calc_ic(mom_5d, fwd_30m)
print(f'mom_5d    趋势: IC={ic.mean():.4f}, IR={abs(ic.mean())/ic.sd():.3f}')

# 7. VWAP偏离
vwap = (close * volume).rolling(60).sum() / volume.rolling(60).sum()
vwap_dev = cs_normalize((close - vwap) / vwap)
ic = calc_ic(vwap_dev, fwd_30m)
print(f'vwap_dev  反转: IC={ic.mean():.4f}, IR={abs(ic.mean())/ic.std():.3f}')

# 分年度看
print('\n分年度IC（mom_5m，预测未来30分钟）:')
for year in [2022, 2023, 2024, 2025]:
    mask = mom_5m.index.year == year
    ic_yr = calc_ic(mom_5m[mask], fwd_30m[mask])
    print(f'  {year}: IC={ic_yr.mean():.4f}, IR={abs(ic_yr.mean())/ic_yr.std():.3f}')