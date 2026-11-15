"""Vectorised labels for the V3 research harness.

- Forward h-day close-to-close return and the market-relative, volatility-adjusted target
  (computed cross-sectionally in add_relative_targets).
- Gap-realistic triple-barrier outcomes (R-multiples) for any (horizon, stop, target) combo.
Entry is the close of day t (matches the 15:50 provisional / EOD scan mode).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def triple_barrier_r(
    df: pd.DataFrame,
    atr: pd.Series,
    horizon: int,
    stop_mult: float,
    target_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (long_r, short_r, long_tp_hit, short_tp_hit) in R-multiples (risk = stop_mult*ATR).

    Stop is checked before target within a bar (conservative). Fills on gap days use the open.
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    a = atr.to_numpy(float)
    n = len(c)
    long_r = np.full(n, np.nan)
    short_r = np.full(n, np.nan)
    l_tp = np.full(n, np.nan)
    s_tp = np.full(n, np.nan)
    m = n - horizon
    if m <= 0:
        return long_r, short_r, l_tp, s_tp

    # windows of the next `horizon` bars for entry index i: bars i+1 .. i+horizon
    W_o = sliding_window_view(o[1:], horizon)[:m]
    W_h = sliding_window_view(h[1:], horizon)[:m]
    W_l = sliding_window_view(lo[1:], horizon)[:m]
    entry = c[:m]
    atr_e = a[:m]
    risk = stop_mult * atr_e
    valid = np.isfinite(entry) & np.isfinite(atr_e) & (atr_e > 0) & (entry > 0)
    exit_c = c[horizon : horizon + m]

    def _first(mask: np.ndarray) -> np.ndarray:
        any_ = mask.any(axis=1)
        idx = np.where(any_, mask.argmax(axis=1), horizon)
        return idx

    rows = np.arange(m)

    # ---- Long
    sl = entry - stop_mult * atr_e
    tp = entry + target_mult * atr_e
    i_sl = _first(W_l <= sl[:, None])
    i_tp = _first(W_h >= tp[:, None])
    sl_first = (i_sl <= i_tp) & (i_sl < horizon)
    tp_first = (i_tp < i_sl) & (i_tp < horizon)
    px = exit_c.copy()
    op_sl = W_o[rows, np.minimum(i_sl, horizon - 1)]
    op_tp = W_o[rows, np.minimum(i_tp, horizon - 1)]
    px = np.where(sl_first, np.minimum(op_sl, sl), px)
    px = np.where(tp_first, np.maximum(op_tp, tp), px)
    lr = (px - entry) / risk
    long_r[:m] = np.where(valid, lr, np.nan)
    l_tp[:m] = np.where(valid, tp_first.astype(float), np.nan)

    # ---- Short
    sl_s = entry + stop_mult * atr_e
    tp_s = entry - target_mult * atr_e
    i_sl = _first(W_h >= sl_s[:, None])
    i_tp = _first(W_l <= tp_s[:, None])
    sl_first = (i_sl <= i_tp) & (i_sl < horizon)
    tp_first = (i_tp < i_sl) & (i_tp < horizon)
    px = exit_c.copy()
    op_sl = W_o[rows, np.minimum(i_sl, horizon - 1)]
    op_tp = W_o[rows, np.minimum(i_tp, horizon - 1)]
    px = np.where(sl_first, np.maximum(op_sl, sl_s), px)
    px = np.where(tp_first, np.minimum(op_tp, tp_s), px)
    sr = (entry - px) / risk
    short_r[:m] = np.where(valid, sr, np.nan)
    s_tp[:m] = np.where(valid, tp_first.astype(float), np.nan)
    return long_r, short_r, l_tp, s_tp


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def add_relative_targets(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """target_rel_h = (fwd_ret_h - same-date universe median) / (atr_pct * sqrt(h)); plus per-date percentile."""
    for h in horizons:
        fr = panel[f"fwd_ret_{h}"]
        med = panel.groupby("date")[f"fwd_ret_{h}"].transform("median")
        tr = (fr - med) / (panel["atr_pct"] * np.sqrt(h))
        panel[f"target_rel_{h}"] = tr.clip(-6.0, 6.0)
        panel[f"target_pct_{h}"] = panel.groupby("date")[f"target_rel_{h}"].rank(pct=True)
    return panel
