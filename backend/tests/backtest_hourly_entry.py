"""Backtest the 1h entry-timing hypothesis against blind EOD entry using the frozen daily US model."""

import numpy as np
import pandas as pd

from app.config import DEFAULT_CONFIG
from app.data.adapter import YFinanceUSAdapter
from app.features.indicators import FEATURE_COLS, compute_features
from app.ml.intraday_entry import compute_hourly_indicators, fetch_hourly_bars
from app.ml.pipeline import load_artifacts


def main() -> None:
    adapter = YFinanceUSAdapter()
    universe = adapter.load_cached_universe()
    history = adapter.load_history([m.symbol for m in universe], years=DEFAULT_CONFIG.history_years)
    bench = adapter.load_benchmark(years=DEFAULT_CONFIG.history_years)
    artifacts = load_artifacts()
    if artifacts is None:
        raise RuntimeError("Frozen daily model artifacts not found.")

    # Use last ~120 trading days where 1h bars (up to 365d) overlap cleanly
    ref_dates = sorted(history["AAPL"].index) if "AAPL" in history else sorted(next(iter(history.values())).index)
    sample_dates = ref_dates[-125:-16:5]  # every 5 trading days (~22 dates)
    print(f"Testing {len(sample_dates)} weekly dates from {sample_dates[0].strftime('%Y-%m-%d')} to {sample_dates[-1].strftime('%Y-%m-%d')}")

    # Precompute daily features & probabilities for those dates
    panel_rows = []
    for sym, df in history.items():
        if len(df) < 120:
            continue
        feats = compute_features(df, bench)
        sub = feats[feats.index.isin(sample_dates)].copy()
        if sub.empty:
            continue
        sub["symbol"] = sym
        sub["date"] = sub.index
        panel_rows.append(sub)

    snap = pd.concat(panel_rows, ignore_index=True).dropna(subset=["atr_14"])
    snap = snap[snap["avg_dollar_vol_m"] >= DEFAULT_CONFIG.min_avg_dollar_vol_m].copy()
    snap["target_gain_pct"] = (DEFAULT_CONFIG.target_atr_mult * snap["atr_14"] / snap["close"]) * 100.0
    snap = snap[snap["target_gain_pct"] >= DEFAULT_CONFIG.min_target_gain_pct].copy()
    snap["prob_long"] = artifacts.model_long.predict_proba(snap[FEATURE_COLS])[:, 1]

    # Pick Top 5 Longs per rebalance date
    picks = []
    for dt in sample_dates:
        day = snap[snap["date"] == dt]
        for _, r in day.nlargest(5, "prob_long").iterrows():
            picks.append(r)

    unique_syms = sorted({r["symbol"] for r in picks})
    print(f"Downloading 1h bars (1y) for {len(unique_syms)} picked symbols...")
    hourly_cache: dict[str, pd.DataFrame] = {}
    for sym in unique_syms:
        h = fetch_hourly_bars(sym, period="1y")
        if not h.empty and len(h) >= 30:
            hourly_cache[sym] = compute_hourly_indicators(h)

    blind_rets, timed_rets = [], []
    blind_hits, timed_hits = 0, 0
    timed_entries = 0

    for r in picks:
        sym = r["symbol"]
        dt = pd.Timestamp(r["date"]).normalize()
        df_d = history[sym]
        fwd_d = df_d[df_d.index > dt].iloc[: DEFAULT_CONFIG.horizon_days]
        if fwd_d.empty:
            continue

        entry_b = float(r["close"])
        atr = float(r["atr_14"])
        tp_b = entry_b + DEFAULT_CONFIG.target_atr_mult * atr
        sl_b = entry_b - DEFAULT_CONFIG.stop_atr_mult * atr

        # 1. Blind EOD entry outcome
        ret_b = (float(fwd_d.iloc[-1]["close"]) - entry_b) / entry_b
        hit_b = 0
        for _, bar in fwd_d.iterrows():
            if float(bar["low"]) <= sl_b:
                ret_b = (sl_b - entry_b) / entry_b
                break
            if float(bar["high"]) >= tp_b:
                ret_b = (tp_b - entry_b) / entry_b
                hit_b = 1
                break
        blind_rets.append(ret_b)
        blind_hits += hit_b

        # 2. Hourly entry timing: wait up to 2 trading days (14 1h bars) for 1h RSI <= 62 & near EMA20
        h_df = hourly_cache.get(sym)
        if h_df is None:
            continue
        h_idx_naive = h_df.index.tz_localize(None) if h_df.index.tz is not None else h_df.index
        h_fwd = h_df[h_idx_naive > dt].iloc[:14]
        entry_t = None
        entry_ts = None
        for ts, hbar in h_fwd.iterrows():
            if float(hbar["rsi_14_1h"]) <= 62.0 and float(hbar["dist_ema20_1h_pct"]) <= 0.8:
                entry_t = float(hbar["close"])
                entry_ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
                break
        if entry_t is None:
            continue  # skipped overextended chase

        timed_entries += 1
        tp_t = entry_t + DEFAULT_CONFIG.target_atr_mult * atr
        sl_t = entry_t - DEFAULT_CONFIG.stop_atr_mult * atr
        fwd_after = df_d[df_d.index >= entry_ts.normalize()].iloc[: DEFAULT_CONFIG.horizon_days]
        if fwd_after.empty:
            continue
        ret_t = (float(fwd_after.iloc[-1]["close"]) - entry_t) / entry_t
        hit_t = 0
        for _, bar in fwd_after.iterrows():
            if float(bar["low"]) <= sl_t:
                ret_t = (sl_t - entry_t) / entry_t
                break
            if float(bar["high"]) >= tp_t:
                ret_t = (tp_t - entry_t) / entry_t
                hit_t = 1
                break
        timed_rets.append(ret_t)
        timed_hits += hit_t

    def _stats(name: str, rets: list[float], hits: int) -> None:
        if not rets:
            print(f"{name}: 0 trades")
            return
        wins = [x for x in rets if x > 0]
        losses = [-x for x in rets if x <= 0]
        pf = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else 9.99
        print(
            f"{name:28s} | n={len(rets):3d} | Target%={hits/len(rets)*100:5.1f}% | "
            f"Win%={len(wins)/len(rets)*100:5.1f}% | AvgRet={np.mean(rets)*100:+5.2f}% | PF={pf:4.2f}"
        )

    print("\n=== HOURLY ENTRY TIMING HYPOTHESIS BACKTEST ===")
    _stats("Blind EOD Entry (Baseline)", blind_rets, blind_hits)
    _stats("1h Pullback Entry Timing", timed_rets, timed_hits)


if __name__ == "__main__":
    main()
