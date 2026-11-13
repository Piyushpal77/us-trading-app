from __future__ import annotations

import numpy as np
import pandas as pd


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    atr: pd.Series,
    target_atr_mult: float = 2.5,
    stop_atr_mult: float = 1.5,
    horizon_days: int = 15,
) -> pd.DataFrame:
    """Compute forward Triple-Barrier outcomes for both Long and Short setups."""
    n = len(df)
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    atrs = atr.to_numpy(dtype=float)

    y_long = np.full(n, np.nan)
    y_short = np.full(n, np.nan)
    long_ret = np.full(n, np.nan)
    short_ret = np.full(n, np.nan)

    for i in range(n - horizon_days):
        entry = closes[i]
        a = atrs[i]
        if not np.isfinite(entry) or not np.isfinite(a) or a <= 0 or entry <= 0:
            continue

        l_tp = entry + target_atr_mult * a
        l_sl = entry - stop_atr_mult * a
        l_outcome = 0.0
        l_r = (closes[i + horizon_days] - entry) / entry

        s_tp = entry - target_atr_mult * a
        s_sl = entry + stop_atr_mult * a
        s_outcome = 0.0
        s_r = (entry - closes[i + horizon_days]) / entry

        l_done = False
        s_done = False
        for j in range(i + 1, i + horizon_days + 1):
            hi = highs[j]
            lo = lows[j]
            if not l_done:
                hit_sl = lo <= l_sl
                hit_tp = hi >= l_tp
                if hit_sl:
                    l_outcome = 0.0
                    l_r = (l_sl - entry) / entry
                    l_done = True
                elif hit_tp:
                    l_outcome = 1.0
                    l_r = (l_tp - entry) / entry
                    l_done = True

            if not s_done:
                hit_sl = hi >= s_sl
                hit_tp = lo <= s_tp
                if hit_sl:
                    s_outcome = 0.0
                    s_r = (entry - s_sl) / entry
                    s_done = True
                elif hit_tp:
                    s_outcome = 1.0
                    s_r = (entry - s_tp) / entry
                    s_done = True

            if l_done and s_done:
                break

        y_long[i] = l_outcome
        y_short[i] = s_outcome
        long_ret[i] = l_r
        short_ret[i] = s_r

    return pd.DataFrame(
        {
            "y_long": y_long,
            "y_short": y_short,
            "long_ret": long_ret,
            "short_ret": short_ret,
        },
        index=df.index,
    )
