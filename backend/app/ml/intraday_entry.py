from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


US_EASTERN = ZoneInfo("America/New_York")


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def fetch_hourly_bars(symbol: str, period: str = "60d") -> pd.DataFrame:
    """Download 1h OHLCV bars from Yahoo Finance and normalize columns."""
    raw = yf.download(symbol, period=period, interval="1h", progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna()
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(US_EASTERN)
    else:
        df.index = df.index.tz_convert(US_EASTERN)
    # Drop partial in-progress 1h bar so RSI/EMA are computed on completed bars only
    now_et = datetime.now(US_EASTERN)
    last_ts = df.index[-1]
    if last_ts.date() == now_et.date() and last_ts.hour == now_et.hour and len(df) > 1:
        df = df.iloc[:-1]
    return df


def compute_hourly_indicators(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Compute 1h EMA(20), RSI(14), ATR(14), and 09:30-ET-aligned 4h-resampled RSI/EMA context."""
    df = df_1h.copy()
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    df["ema_20_1h"] = c.ewm(span=20, adjust=False).mean()
    df["rsi_14_1h"] = _compute_rsi(c, 14)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr_14_1h"] = tr.rolling(14).mean()
    df["dist_ema20_1h_pct"] = (c / df["ema_20_1h"] - 1.0) * 100.0
    df["dist_ema20_1h_atr"] = (c - df["ema_20_1h"]) / df["atr_14_1h"].replace(0.0, np.nan)

    # Align 4h resample to 09:30 ET session open so bars don't straddle sessions
    df_4h = (
        df[["open", "high", "low", "close", "volume"]]
        .resample("4h", origin="1970-01-01 09:30:00")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    if len(df_4h) >= 20:
        df_4h["ema_20_4h"] = df_4h["close"].ewm(span=20, adjust=False).mean()
        df_4h["rsi_14_4h"] = _compute_rsi(df_4h["close"], 14)
        df["ema_20_4h"] = df_4h["ema_20_4h"].reindex(df.index, method="ffill")
        df["rsi_14_4h"] = df_4h["rsi_14_4h"].reindex(df.index, method="ffill")
    else:
        df["ema_20_4h"] = df["ema_20_1h"]
        df["rsi_14_4h"] = df["rsi_14_1h"]
    return df


def analyze_hourly_entry(symbol: str, side: str = "LONG") -> dict[str, Any]:
    """Evaluate 1h/4h entry timing for a stock already selected by the frozen daily model."""
    df_1h = fetch_hourly_bars(symbol, period="60d")
    if df_1h.empty or len(df_1h) < 30:
        return {
            "symbol": symbol,
            "side": side,
            "status": "UNAVAILABLE",
            "signal": "NO_INTRADAY_DATA",
            "reason": "Insufficient 1h bars from Yahoo Finance.",
        }

    feats = compute_hourly_indicators(df_1h)
    last = feats.iloc[-1]
    px = float(last["close"])
    ema_1h = float(last["ema_20_1h"])
    ema_4h = float(last["ema_20_4h"])
    rsi_1h = float(last["rsi_14_1h"]) if np.isfinite(last["rsi_14_1h"]) else 50.0
    rsi_4h = float(last["rsi_14_4h"]) if np.isfinite(last["rsi_14_4h"]) else 50.0
    dist_pct = float(last["dist_ema20_1h_pct"]) if np.isfinite(last["dist_ema20_1h_pct"]) else 0.0
    atr_1h = float(last["atr_14_1h"]) if np.isfinite(last["atr_14_1h"]) else px * 0.005
    dist_atr = float(last["dist_ema20_1h_atr"]) if np.isfinite(last["dist_ema20_1h_atr"]) else 0.0

    side_u = side.upper().strip()
    if side_u == "LONG":
        if rsi_1h <= 58.0 and dist_atr <= 0.75 and px >= ema_4h * 0.99:
            signal = "ENTER_NOW"
            reason = f"1h RSI ({rsi_1h:.1f}) & price within {dist_atr:+.2f} ATR ({dist_pct:+.2f}%) of 1h EMA20."
        elif rsi_1h > 65.0 or dist_atr > 1.5:
            signal = "WAIT_PULLBACK"
            reason = f"Extended intraday (1h RSI {rsi_1h:.1f}, {dist_atr:+.2f} ATR above 1h EMA20); wait for dip."
        else:
            signal = "NEUTRAL_LIMIT"
            reason = f"Moderate intraday setup (1h RSI {rsi_1h:.1f}, {dist_atr:+.2f} ATR); consider limit near 1h EMA20."
        suggested_entry = round(min(px, ema_1h + 0.25 * atr_1h), 2)
    else:
        if rsi_1h >= 42.0 and dist_atr >= -0.75 and px <= ema_4h * 1.01:
            signal = "ENTER_NOW"
            reason = f"1h RSI ({rsi_1h:.1f}) & price within {dist_atr:+.2f} ATR ({dist_pct:+.2f}%) of 1h EMA20."
        elif rsi_1h < 35.0 or dist_atr < -1.5:
            signal = "WAIT_BOUNCE"
            reason = f"Oversold intraday (1h RSI {rsi_1h:.1f}, {dist_atr:+.2f} ATR below 1h EMA20); wait for bounce."
        else:
            signal = "NEUTRAL_LIMIT"
            reason = f"Moderate short setup (1h RSI {rsi_1h:.1f}, {dist_atr:+.2f} ATR); consider limit near 1h EMA20."
        suggested_entry = round(max(px, ema_1h - 0.25 * atr_1h), 2)

    return {
        "symbol": symbol,
        "side": side_u,
        "status": "OK",
        "signal": signal,
        "reason": reason,
        "last_1h_bar": str(feats.index[-1]),
        "current_price": round(px, 2),
        "suggested_entry": suggested_entry,
        "ema_20_1h": round(ema_1h, 2),
        "ema_20_4h": round(ema_4h, 2),
        "rsi_14_1h": round(rsi_1h, 1),
        "rsi_14_4h": round(rsi_4h, 1),
        "dist_ema20_1h_pct": round(dist_pct, 2),
    }
