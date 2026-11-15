# US Stock Screener — Detailed Architecture & Technical Specification (`feat/hourly-entry-timing`)

> **Purpose of this document:** Provide a complete, transparent technical specification of the **US Stock Screener** (`us-trading-app`) for third-party quantitative and engineering review, highlighting exact formulas, hyperparameters, validation methodology, and known areas for improvement.

---

## 1. System Overview

The **US Stock Screener** is a two-stage swing-trading decision-support system for US large-cap equities (**S&P 500 + Nasdaq 100**, ~515 liquid stocks):

1. **Stage 1 — Daily (`1D`) ML Alpha & Risk Model (Frozen):**
   - Two independent **LightGBM** binary classifiers (`model_long`, `model_short`) trained on 10 years of daily EOD OHLCV data using **Triple-Barrier** labels (`15`-trading-day horizon).
   - Ranks the universe by predicted win probability (`% Win`), filters for liquidity and minimum target move, and explains every pick with **SHAP `TreeExplainer`** feature attributions.
2. **Stage 2 — Intraday (`1H` / `4H`) Entry-Timing Layer (Decoupled):**
   - Evaluates live `1h` bars (and `4h`-resampled trend context) from Yahoo Finance on the Top 10 Longs and Top 10 Shorts without altering the frozen `1D` ML probabilities.
   - Emits `ENTER_NOW`, `NEUTRAL_LIMIT`, or `WAIT_PULLBACK` / `WAIT_BOUNCE` signals, polling hourly with audio/browser notifications and visual alerts.

### Tech Stack
- **Backend:** Python 3.12, FastAPI, Uvicorn (`127.0.0.1:8001`), LightGBM, SHAP, scikit-learn, pandas, NumPy, yfinance, SQLite (`backend/data_cache/`).
- **Frontend:** React 18, Vite (`127.0.0.1:5174`), Tailwind CSS, custom SVG charts (`PriceChart.jsx`, `EquityChart.jsx`).

---

## 2. Universe, Data Ingestion & Caching

- **Universe (`app/data/universe.py`):** S&P 500 + Nasdaq 100 constituents (~515 symbols), tagged with `sector`, `index_membership`, and `is_shortable` metadata.
- **Storage & Sync (`app/data/adapter.py`):**
  - Historical daily OHLCV bars (`10` years) are cached in a local SQLite database (`ohlcv` and `benchmark` tables).
  - Incremental sync (`sync_symbols`) fetches only the delta (`last_date - 3 days` to `today + 1`) for cached symbols and full `10y` history for new symbols.
- **Liquidity & Volatility Pre-Filters (`app/config.py`):**
  - `min_avg_dollar_vol_m = 25.0` (minimum 20-day average daily dollar volume of \$25M).
  - `min_target_gain_pct = 5.0` (requires `2.5 × ATR(14) / Close >= 5.0%` so setups have meaningful swing amplitude).
  - `top_k = 10` (Top 10 Longs and Top 10 Shorts).
  - `min_win_prob = 0.0` (always populates Top 10 even in weak regimes; see Section 9).
- **Provisional 3:50 PM ET Mode:** Supports overlaying live intraday quotes onto today's provisional daily bar (`apply_provisional_snapshot`) to run EOD-style scans 10 minutes before market close.

---

## 3. Feature Engineering (38 Leak-Free Daily Features)

Computed in `app/features/indicators.py` (`compute_features`) strictly from past/current daily bars:

| Category | Features (`38` total) |
|---|---|
| **Returns / Momentum (5)** | `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d` |
| **Trend & Moving Averages (7)** | `dist_ema10`, `dist_ema20`, `dist_ema50`, `dist_ema200`, `ema20_slope`, `ema50_slope`, `ema_stack` (`EMA20 > EMA50 > EMA200`) |
| **Oscillators & Momentum (9)** | `rsi_14`, `rsi_5`, `rsi_slope_5`, `macd_hist_norm` (ATR-normalized), `macd_hist_slope`, `stoch_k`, `stoch_d`, `adx_14`, `di_diff` (`DI+ - DI-`) |
| **Volatility & Squeeze (4)** | `atr_pct` (`ATR14 / Close`), `bb_width`, `bb_pct_b`, `squeeze_ratio` |
| **Volume & Money Flow (5)** | `vol_ratio_20`, `vol_z_20`, `obv_slope_10`, `cmf_20` (Chaikin Money Flow), `up_down_vol_10` |
| **Candlestick / Range (4)** | `clv` (Close Location Value), `body_to_range`, `dist_52w_high`, `dist_52w_low` |
| **Relative Strength vs. S&P 500 (4)** | `rs_5d`, `rs_20d`, `rs_60d`, `rs_slope_10` |

---

## 4. Target Labeling (Triple-Barrier Method)

Implemented in `app/ml/labeling.py` (`compute_triple_barrier_labels`):

For each historical bar $i$ with entry price $E = \text{Close}_i$ and volatility $A = \text{ATR}_{14,i}$:
- **Take-Profit Barrier:** $2.5 \times A$ (`target_atr_mult = 2.5`)
- **Stop-Loss Barrier:** $2.0 \times A$ (`stop_atr_mult = 2.0`)
- **Vertical Time Barrier:** $H = 15$ trading days (`horizon_days = 15`)

Scanning forward bars $j \in [i+1, i+15]$:
- **Long (`y_long`):**
  - If $\text{Low}_j \le E - 2.0A$ first $\rightarrow$ `y_long = 0`, realized return $= -2.0A / E$.
  - Else if $\text{High}_j \ge E + 2.5A$ first $\rightarrow$ `y_long = 1`, realized return $= +2.5A / E$.
  - If neither barrier is hit within 15 days $\rightarrow$ `y_long = 0`, realized return $= (\text{Close}_{i+15} - E) / E$.
- **Short (`y_short`):** Symmetric ($E - 2.5A$ TP, $E + 2.0A$ SL).
- **Conservative Tie-Breaking:** Stop-loss is checked **before** take-profit on each bar to avoid optimistic same-bar bias.

---

## 5. ML Model, Scoring & Explainability

Implemented in `app/ml/pipeline.py`:

- **Model:** Two `lightgbm.LGBMClassifier` models (`model_long`, `model_short`) with:
  - `n_estimators=220`, `learning_rate=0.04`, `num_leaves=31`, `max_depth=6`
  - `min_child_samples=60`, `colsample_bytree=0.75`, `subsample=0.8`, `subsample_freq=1`
  - `reg_alpha=0.5`, `reg_lambda=1.0`, `random_state=42`
- **Scoring & Ranking:**
  - Raw predicted probability: `prob = model.predict_proba(X)[:, 1]` $\rightarrow$ displayed as **`win_prob_pct` (`% Win`)**.
  - Risk-Reward Ratio: $RR = 2.5 / 2.0 = 1.25$.
  - Composite Score: $\text{Score} = \text{prob} \times RR \times 100$.
  - Grades: `A+` ($\ge 115$), `A` ($\ge 95$), `B` ($\ge 78$), `C` ($< 78$).
  - Theoretical breakeven win rate at $1.25R$ (ignoring time-outs) is $\frac{1}{1 + 1.25} = 44.4\%$.
- **Explainability:** `shap.TreeExplainer` extracts the top 4 positive/negative feature drivers per recommended stock.

---

## 6. Walk-Forward Validation & Observed Backtest Metrics

Implemented in `run_walk_forward_backtest` (`app/ml/pipeline.py`):

- **Split Design:** 3-split expanding-window walk-forward starting at 45% of the date range.
- **Leakage Prevention:** An embargo gap of `horizon_days = 15` trading days is enforced between `train_cutoff` and `test_start` so no training label overlaps with test dates.
- **Portfolio Simulation:** Rebalanced every `5` trading days; selects top `10` Longs + top `3` Shortable Shorts passing liquidity/gain filters; scales average step return by `0.35` leverage/exposure factor.
- **Observed OOS Metrics (from frozen `LIVE_EOD` model trained `2026-11-13`):**
  - **OOS Days:** `1,364` | **Total Trades:** `728`
  - **OOS Win Rate (positive return):** `52.9%`
  - **Precision@5 (Long TB Hit Rate):** `41.9%`
  - **Precision@5 (Short TB Hit Rate):** `25.4%`
  - **Profit Factor:** `1.33`
  - **Max Drawdown:** `-30.7%`

---

## 7. Hourly (`1H` / `4H`) Entry-Timing Layer

Implemented in `app/ml/intraday_entry.py` (`analyze_hourly_entry`) and surfaced via `/api/entry-timing/scan` & `/api/entry-timing/{symbol}`:

- **Data:** Downloads `60d` of `1h` bars from Yahoo Finance (`yfinance`) and resamples `1h -> 4h` for broader intraday trend context.
- **Indicators:** `EMA(20)` on `1h` and `4h`, `RSI(14)` on `1h` and `4h`, `ATR(14)` on `1h`, and `% distance from 1h EMA(20)`.
- **Decision Rules:**
  - **LONG:**
    - `ENTER_NOW`: `1h RSI <= 62.0` AND `dist_ema20_1h <= +1.0%` AND `Close >= 0.99 × EMA20_4h`
    - `WAIT_PULLBACK`: `1h RSI > 68.0` OR `dist_ema20_1h > +1.8%`
    - `NEUTRAL_LIMIT`: otherwise; `suggested_entry = min(Close, EMA20_1h + 0.25 × ATR_1h)`
  - **SHORT:**
    - `ENTER_NOW`: `1h RSI >= 38.0` AND `dist_ema20_1h >= -1.0%` AND `Close <= 1.01 × EMA20_4h`
    - `WAIT_BOUNCE`: `1h RSI < 32.0` OR `dist_ema20_1h < -1.8%`
    - `NEUTRAL_LIMIT`: otherwise; `suggested_entry = max(Close, EMA20_1h - 0.25 × ATR_1h)`
- **Frontend Alerting (`App.jsx`):** Auto-polls `/api/entry-timing/scan` every 60 minutes (while the browser tab is open), plays a WebAudio chime + Browser Notification on newly entered `ENTER_NOW` stocks, and renders dramatic flashing emerald (Long) / rose (Short) banners and card borders.

---

## 8. Paper-Trade Ledger & Tracking

Implemented in `app/ledger.py` (`recommendations_log.csv`):
- Automatically logs Top Longs and Top Shorts on each scan date and tracks daily OHLC progression to resolve `TP_HIT`, `SL_HIT`, `TIME_EXIT` (after 15 trading days), or `OPEN` status with MFE/MAE and realized return.

---

## 9. Known Limitations & Areas for Third-Party Review

We specifically invite review and recommendations on the following areas:

1. **Probability Calibration:**
   - LightGBM `predict_proba` outputs are uncalibrated tree scores. Adding **Platt scaling** or **Isotonic Regression** on out-of-fold validation predictions could make `% Win` directly interpretable as true calibrated hit probability.
2. **Short-Side Precision Gap (`25.4%` vs `41.9%` Long):**
   - US large-cap equities have a structural upward drift, and Short setups frequently show `% Win` in the 20s–30s (below the `44.4%` $1.25R$ breakeven).
   - Options to explore: regime-aware gating (only enable Shorts when SPY < 50/200 EMA or VIX elevated), asymmetric Short barriers, or raising `min_win_prob` above `0.0` so low-conviction setups are suppressed instead of filling Top 10.
3. **Transaction Costs, Slippage & Borrow Fees:**
   - The walk-forward backtest currently evaluates gross Triple-Barrier returns without deducting commissions, bid-ask slippage, or short borrow costs.
4. **Benchmark Table Fallback:**
   - If the `benchmark` SQLite table is empty, relative-strength features (`rs_5d`, `rs_20d`, `rs_60d`, `rs_slope_10`) fall back to neutral defaults. Ensuring `^GSPC` is always synced and persisted would stabilize RS features.
5. **Intraday Entry-Timing Validation:**
   - The `1H`/`4H` `ENTER_NOW` thresholds (`RSI <= 62`, `dist <= 1.0%`) are rule-based heuristics and have not yet been backtested on historical `1h` fills to quantify slippage/drawdown improvement vs. market-on-close/next-open entry.
6. **Background Scheduling Independence:**
   - Hourly `ENTER_NOW` polling currently runs via `setInterval` in the React frontend (requires the browser tab to remain open, though the Antigravity workspace window itself does not need to stay focused). Moving the hourly scheduler to a FastAPI background worker (`APScheduler`) with webhook/email/Telegram alerts would make alerting 100% headless.
7. **Experiment Already Tested & Rejected:**
   - **4H Classical Chart Patterns (8 geometric patterns):** Tested on `feat/chart-patterns` over 84 OOS rebalance dates (`2025-02-24` to `2026-10-19`, 483 symbols). Pure `1D` ML (`Win Rate 53.9%`, `PF 1.58`, `Equity 176.84`) outperformed `1D ML + 4H Pattern Confluence` (`Win Rate 52.2%`, `PF 1.48`, `Equity 159.47`) and `4H Breakout Only` (`Win Rate 43.4%`, `PF 0.94`, `Equity 98.68`), so geometric pattern re-ranking was dropped.
