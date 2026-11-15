from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


FEATURE_LABELS: dict[str, str] = {
    "ret_1d": "1D Return",
    "ret_5d": "5D Momentum",
    "ret_10d": "10D Momentum",
    "ret_20d": "20D Momentum",
    "ret_60d": "60D Trend",
    "dist_ema10": "Dist from EMA10",
    "dist_ema20": "Dist from EMA20",
    "dist_ema50": "Dist from EMA50",
    "dist_ema200": "Dist from EMA200",
    "ema20_slope": "EMA20 Slope",
    "ema50_slope": "EMA50 Slope",
    "ema_stack": "EMA Alignment (20>50>200)",
    "rsi_14": "RSI (14)",
    "rsi_5": "Fast RSI (5)",
    "rsi_slope_5": "RSI 5D Slope",
    "macd_hist_norm": "MACD Histogram (ATR-norm)",
    "macd_hist_slope": "MACD Hist Slope",
    "stoch_k": "Stochastic %K",
    "stoch_d": "Stochastic %D",
    "adx_14": "ADX (14) Trend Strength",
    "di_diff": "DI+ minus DI-",
    "atr_pct": "ATR % Volatility",
    "bb_width": "Bollinger Band Width",
    "bb_pct_b": "Bollinger %B Position",
    "squeeze_ratio": "Volatility Squeeze Ratio",
    "vol_ratio_20": "Volume vs 20D Avg",
    "vol_z_20": "Volume Z-Score (20D)",
    "obv_slope_10": "OBV 10D Slope",
    "cmf_20": "Chaikin Money Flow (20D)",
    "up_down_vol_10": "Up/Down Vol Ratio (10D)",
    "clv": "Close Location Value",
    "body_to_range": "Candle Body / Range",
    "dist_52w_high": "Dist from 52W High",
    "dist_52w_low": "Dist from 52W Low",
    "rs_5d": "5D Rel. Strength vs S&P 500",
    "rs_20d": "20D Rel. Strength vs S&P 500",
    "rs_60d": "60D Rel. Strength vs S&P 500",
    "rs_slope_10": "RS 10D Slope",
}

FEATURE_COLS: list[str] = list(FEATURE_LABELS.keys())


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_features(df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute leak-free daily technical, volume, volatility-squeeze, and relative-strength
    features. Safe for both historical EOD bars and 3:50 PM ET provisional bars.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    vol = df["volume"].astype(float).replace(0.0, np.nan)

    atr_14 = compute_atr(df, 14)
    out["atr_14"] = atr_14
    out["atr_pct"] = atr_14 / close

    for d in (1, 5, 10, 20, 60):
        out[f"ret_{d}d"] = close.pct_change(d)

    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    out["dist_ema10"] = (close - ema10) / ema10
    out["dist_ema20"] = (close - ema20) / ema20
    out["dist_ema50"] = (close - ema50) / ema50
    out["dist_ema200"] = (close - ema200) / ema200
    out["ema20_slope"] = ema20.pct_change(5)
    out["ema50_slope"] = ema50.pct_change(10)
    out["ema_stack"] = ((ema20 > ema50).astype(int) + (ema50 > ema200).astype(int)).astype(float)

    rsi14 = _rsi(close, 14)
    out["rsi_14"] = rsi14
    out["rsi_5"] = _rsi(close, 5)
    out["rsi_slope_5"] = rsi14.diff(5)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    out["macd_hist_norm"] = macd_hist / atr_14.replace(0.0, np.nan)
    out["macd_hist_slope"] = out["macd_hist_norm"].diff(3)

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100.0 * (close - low14) / (high14 - low14).replace(0.0, np.nan)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_14.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_14.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    out["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    out["di_diff"] = plus_di - minus_di

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    out["bb_width"] = (bb_upper - bb_lower) / sma20.replace(0.0, np.nan)
    out["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)
    out["squeeze_ratio"] = out["bb_width"] / out["bb_width"].rolling(120, min_periods=20).mean()

    vol_ma20 = vol.rolling(20).mean()
    vol_std20 = vol.rolling(20).std()
    out["vol_ratio_20"] = vol / vol_ma20.replace(0.0, np.nan)
    out["vol_z_20"] = (vol - vol_ma20) / vol_std20.replace(0.0, np.nan)

    obv = (np.sign(close.diff()).fillna(0.0) * vol.fillna(0.0)).cumsum()
    out["obv_slope_10"] = obv.diff(10) / (vol_ma20 * 10.0).replace(0.0, np.nan)

    hl_range = (high - low).replace(0.0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    out["clv"] = clv
    out["body_to_range"] = (close - open_) / hl_range
    out["cmf_20"] = (clv * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0.0, np.nan)

    up_vol = vol.where(close > close.shift(1), 0.0).rolling(10).sum()
    dn_vol = vol.where(close < close.shift(1), 0.0).rolling(10).sum()
    out["up_down_vol_10"] = (up_vol + 1.0) / (dn_vol + 1.0)

    high252 = high.rolling(252, min_periods=60).max()
    low252 = low.rolling(252, min_periods=60).min()
    out["dist_52w_high"] = (close - high252) / high252
    out["dist_52w_low"] = (close - low252) / low252

    if benchmark_df is not None and not benchmark_df.empty:
        b_close = benchmark_df["close"].reindex(df.index).ffill()
        for d in (5, 20, 60):
            out[f"rs_{d}d"] = close.pct_change(d) - b_close.pct_change(d)
        rs_line = close / b_close.replace(0.0, np.nan)
        out["rs_slope_10"] = rs_line.pct_change(10)
    else:
        warnings.warn(
            "benchmark_df is empty — relative strength features (rs_5d, rs_20d, rs_60d, rs_slope_10) are falling back to raw returns.",
            RuntimeWarning,
            stacklevel=2,
        )
        out["rs_5d"] = out["ret_5d"]
        out["rs_20d"] = out["ret_20d"]
        out["rs_60d"] = out["ret_60d"]
        out["rs_slope_10"] = out["ret_10d"]

    # Average daily dollar volume in USD Millions
    out["avg_dollar_vol_m"] = (close * vol).rolling(20).mean() / 1e6
    out["close"] = close
    return out
