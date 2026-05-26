import pickle
import pandas as pd
import numpy as np
import os

DATA_DIR = r'C:\Users\chloe\Desktop\quant'

print('Loading data...')
close  = pickle.load(open(os.path.join(DATA_DIR, 'panel_close.pkl'),  'rb'))
volume = pickle.load(open(os.path.join(DATA_DIR, 'panel_volume.pkl'), 'rb'))
open_  = pickle.load(open(os.path.join(DATA_DIR, 'panel_open.pkl'),   'rb'))
print(f'close: {close.shape}')

def cs_normalize(df):
    mean = df.mean(axis=1)
    std  = df.std(axis=1)
    return df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

def cs_rank(df):
    return df.rank(axis=1, pct=True) - 0.5

def broadcast_daily_to_minute(daily_df, minute_index):
    result = pd.DataFrame(index=minute_index, columns=daily_df.columns, dtype=float)
    dates  = pd.Series(minute_index.date, index=minute_index)
    for date, row in daily_df.iterrows():
        mask = dates == date
        result.loc[mask] = row.values
    return result

# ── 动量因子 ──────────────────────────────────────────
print('计算动量因子...')
ret_1m  = close.pct_change(1)
mom_5m  = close.pct_change(5)
mom_30m = close.pct_change(30)

# ── 波动率和价量因子 ──────────────────────────────────
print('计算波动率和价量因子...')
vol_60m = ret_1m.rolling(60).std()

print('  价量相关性（较慢）...')
corr_pv_60m = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
for col in close.columns:
    corr_pv_60m[col] = close[col].rolling(60).corr(volume[col])

# ── 新增因子 ──────────────────────────────────────────
print('计算新增因子...')

# 隔夜跳空
print('  隔夜跳空...')
daily_first = close.groupby(close.index.date).first()
daily_last  = close.groupby(close.index.date).last()
overnight_gap_daily = daily_first / daily_last.shift(1) - 1
overnight_gap = broadcast_daily_to_minute(overnight_gap_daily, close.index)

# VWAP偏离
print('  VWAP偏离...')
vwap_60m = (close * volume).rolling(60).sum() / volume.rolling(60).sum()
vwap_dev  = (close - vwap_60m) / vwap_60m

# 日内趋势
print('  日内趋势...')
daily_open    = open_.groupby(open_.index.date).first()
intraday_open = broadcast_daily_to_minute(daily_open, close.index)
intraday_trend = (close - intraday_open) / intraday_open

# ── 标准化和排名 ──────────────────────────────────────
print('横截面标准化...')
factors = {
    'mom_5m':          cs_normalize(mom_5m),
    'mom_30m':         cs_normalize(mom_30m),
    'vol_60m':         cs_normalize(vol_60m),
    'corr_pv_60m':     cs_normalize(corr_pv_60m),
    'overnight_gap':   cs_normalize(overnight_gap),
    'vwap_dev':        cs_normalize(vwap_dev),
    'intraday_trend':  cs_normalize(intraday_trend),
    'rank_mom_5m':     cs_rank(mom_5m),
    'rank_mom_30m':    cs_rank(mom_30m),
    'rank_vol_60m':    cs_rank(vol_60m),
    'rank_corr_pv':    cs_rank(corr_pv_60m),
    'rank_overnight':  cs_rank(overnight_gap),
    'rank_vwap_dev':   cs_rank(vwap_dev),
    'rank_intraday':   cs_rank(intraday_trend),
}

print('\n====== 因子汇总 ======')
for name, df in factors.items():
    nan_pct = df.isnull().sum().sum() / df.size * 100
    print(f'{name:20s}: NaN={nan_pct:.1f}%')

pickle.dump(factors, open(os.path.join(DATA_DIR, 'factors.pkl'), 'wb'))
print('\nfactors.pkl 保存完成，共', len(factors), '个因子')