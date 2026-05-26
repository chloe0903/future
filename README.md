# Minute-Frequency Commodity Futures Alpha Strategy

A quantitative trading strategy for Chinese commodity futures markets using minute-level data.
Covers 8 futures contracts: RB (Rebar), I (Iron Ore), JM (Coking Coal), P (Palm Oil),
Y (Soybean Oil), M (Soybean Meal), TA (PTA), MA (Methanol).

---

## Project Structure

| File | Description |
|------|-------------|
| `factor_calculation.py` | Computes 14 alpha factors with cross-sectional normalization |
| `factor_analysis.py` | IC analysis, factor screening, IC decay plots |
| `signal_synthesis.py` | Combines valid factors into a composite alpha signal |
| `backtest.py` | Local backtesting engine with performance attribution |

---

## Pipeline
panel_close.pkl / panel_volume.pkl / panel_open.pkl
↓
factor_calculation.py   →  factors.pkl (14 factors)
↓
factor_analysis.py      →  ic_results.pkl, valid_factors.pkl, ic_series.png, ic_decay.png
↓
signal_synthesis.py     →  alpha_signal.pkl, alpha_signal.png
↓
backtest.py             →  backtest_result.png, performance metrics

---

## factor_calculation.py

Loads minute-level panel data and computes 14 factors across 8 symbols.

**Time-series factors (7):**

| Factor | Formula | Type |
|--------|---------|------|
| mom_5m | (close_t - close_{t-5}) / close_{t-5} | Short-term reversal |
| mom_30m | (close_t - close_{t-30}) / close_{t-30} | Medium-term reversal |
| vol_60m | Rolling 60-min return std | Volatility |
| corr_pv_60m | Rolling 60-min price-volume Spearman correlation | Price-volume |
| overnight_gap | Today's first close / yesterday's last close - 1 | Overnight gap |
| vwap_dev | (close - VWAP_60m) / VWAP_60m | Mean reversion |
| intraday_trend | (close - today's open) / today's open | Intraday trend |

**Cross-sectional rank factors (7):** `rank_` prefix versions of each factor above,
normalized to [-0.5, +0.5].

All factors are cross-sectionally z-score normalized at each timestamp.

**Output:** `factors.pkl` — dict of {factor_name: DataFrame(time × symbol)}

---

## factor_analysis.py

Computes Rank IC (Spearman correlation) between each factor and 5-minute forward returns.
Aggregates by trading day to avoid look-ahead bias across session boundaries.

**Screening criteria:** |IC mean| > 0.02 AND |IR| > 0.5

**Results — 6 factors passed screening:**

| Factor | IC Mean | IR |
|--------|---------|----|
| rank_mom_5m | -0.0749 | 1.727 |
| rank_vwap_dev | -0.0457 | 0.914 |
| rank_mom_30m | -0.0366 | 0.773 |

All valid factors are **reversal signals** (negative IC): symbols that rose more in the
recent window tend to fall in the next 5 minutes (microstructure mean reversion).

IC decay analysis shows signal half-life of ~5-10 minutes.

**Output:** `ic_results.pkl`, `valid_factors.pkl`, `ic_series.png`, `ic_decay.png`

---

## signal_synthesis.py

Combines the 3 independent rank factors using ICIR weighting:

| Factor | Weight |
|--------|--------|
| rank_mom_5m | 50.6% |
| rank_vwap_dev | 26.8% |
| rank_mom_30m | 22.7% |

Applies 3-period moving average smoothing to reduce noise (9.1% std reduction).

**Trading logic:** Long the 2 symbols with the **lowest** alpha score,
short the 2 symbols with the **highest** alpha score.

**Output:** `alpha_signal.pkl`, `alpha_signal.png`

---

## backtest.py

Local backtesting engine simulating the full strategy from 2022-01-01 to 2025-05-31.

**Settings:**

| Parameter | Value |
|-----------|-------|
| Initial capital | 1,000,000 CNY |
| Commission | 0.02% per side |
| Slippage | 0.01% per trade |
| Rebalance frequency | Every 30 minutes |
| Long / Short | Top 2 / Bottom 2 by alpha |
| Stop loss | 2% intraday drawdown → full liquidation |
| Trend filter | Pause trading when 5+ symbols move >0.3% in same direction |

**Key finding:** Factor signals are valid every year (IC < -0.06, IR > 1.6 annually),
but the 30-minute rebalance frequency is too slow relative to the signal half-life
(5-10 min), causing returns to be eroded by transaction costs.

---

## Requirements
pandas
numpy
matplotlib
pickle (built-in)

Data sourced from JoinQuant (JQData) platform.
Raw panel pkl files are excluded from this repo (.gitignore).

---

## Key Results

| Year | Return | Sharpe | Max Drawdown |
|------|--------|--------|--------------|
| 2022 | -36.9% | -1.25 | -45.0% |
| 2023 | -25.0% | -1.13 | -32.3% |
| 2024 | -25.8% | -1.27 | -28.3% |
| 2025 | +21.6% | +1.00 | -6.4% |

2022-2023 underperformance is attributed to the Russia-Ukraine war triggering
a sustained one-directional rally in commodity futures, which reversal factors
structurally cannot handle. Factor IC remains consistently negative throughout,
confirming the signal itself is valid.
