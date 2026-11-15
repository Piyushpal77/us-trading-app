"""V3 feature set (Claude feedback): cross-sectional, regime, sector-relative, beta/residual,
gap, earnings and trend-quality features layered on top of the base technical features.

All features are leak-free: every value at date t only uses data available at the close of t
(scheduled earnings dates are public in advance, so "days to next earnings" is not look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.indicators import FEATURE_COLS as BASE_FEATURE_COLS
from app.features.indicators import compute_features

SECTOR_ALIASES = {"Technology": "Information Technology"}

# Per-stock (time-series) extra features
STOCK_EXTRA_COLS = [
    "gap_pct",
    "overnight_ret_5d",
    "intraday_ret_5d",
    "mom_12_1",
    "mom_6_1",
    "er_20",
    "trend_t_60",
    "atr_pct_vol_60",
    "beta_60",
    "resid_ret_20",
    "idio_vol_60",
    "days_to_earn",
    "days_since_earn",
    "last_earn_move_atr",
]

# Cross-sectional percentile ranks (per date)
CS_RANK_SOURCE = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "rs_5d", "rs_20d", "rs_60d",
    "dist_ema10", "dist_ema20", "dist_ema50", "dist_ema200",
    "dist_52w_high", "dist_52w_low",
    "rsi_14", "adx_14", "vol_ratio_20", "atr_pct",
    "mom_12_1", "mom_6_1", "resid_ret_20", "beta_60", "idio_vol_60", "er_20", "trend_t_60",
]
CS_RANK_COLS = [f"cs_{c}" for c in CS_RANK_SOURCE]

SECTOR_COLS = ["sect_rel_ret_20", "sect_rel_ret_60", "sector_rank_20"]

REGIME_COLS = [
    "spy_dist_ema50",
    "spy_dist_ema200",
    "spy_rv20",
    "vix",
    "vix_chg5",
    "vix_ratio",
    "breadth_50",
    "hi_lo_net",
    "xs_dispersion_20",
]

V3_FEATURE_COLS: list[str] = BASE_FEATURE_COLS + STOCK_EXTRA_COLS + CS_RANK_COLS + SECTOR_COLS + REGIME_COLS

V3_FEATURE_LABELS: dict[str, str] = {
    "gap_pct": "Overnight Gap %",
    "overnight_ret_5d": "5D Overnight Return",
    "intraday_ret_5d": "5D Intraday Return",
    "mom_12_1": "12-1 Momentum",
    "mom_6_1": "6-1 Momentum",
    "er_20": "Kaufman Efficiency (20D)",
    "trend_t_60": "60D Trend t-stat",
    "atr_pct_vol_60": "ATR% Variability (60D)",
    "beta_60": "60D Beta to S&P 500",
    "resid_ret_20": "20D Residual Return",
    "idio_vol_60": "Idiosyncratic Vol (60D)",
    "days_to_earn": "Days to Next Earnings",
    "days_since_earn": "Days Since Earnings",
    "last_earn_move_atr": "Last Earnings Move (ATR)",
    "sect_rel_ret_20": "20D Return vs Sector",
    "sect_rel_ret_60": "60D Return vs Sector",
    "sector_rank_20": "Sector 20D Rank",
    "spy_dist_ema50": "S&P vs EMA50",
    "spy_dist_ema200": "S&P vs EMA200",
    "spy_rv20": "S&P 20D Realized Vol",
    "vix": "VIX Level",
    "vix_chg5": "VIX 5D Change",
    "vix_ratio": "VIX / VIX3M",
    "breadth_50": "% Universe > EMA50",
    "hi_lo_net": "52W Highs - Lows %",
    "xs_dispersion_20": "Cross-Sectional Dispersion",
}
for _c in CS_RANK_SOURCE:
    V3_FEATURE_LABELS[f"cs_{_c}"] = f"XS Rank: {_c}"


def _trading_day_distance(dates: pd.DatetimeIndex, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each trading date, trading-day count to next event (>= today) and since last event (< today)."""
    n = len(dates)
    to_next = np.full(n, np.nan)
    since_last = np.full(n, np.nan)
    if events.size == 0 or n == 0:
        return to_next, since_last
    future_bdays = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=120)
    full_dates = dates.union(future_bdays)
    n_full = len(full_dates)
    pos = np.searchsorted(full_dates.values, events)  # index of first trading date >= event
    pos = np.unique(pos[pos < n_full])
    if pos.size == 0:
        return to_next, since_last
    idx = np.arange(n)
    nxt = np.searchsorted(pos, idx, side="left")
    has_next = nxt < len(pos)
    to_next[has_next] = pos[nxt[has_next]] - idx[has_next]
    prv = nxt - 1
    has_prev = prv >= 0
    since_last[has_prev] = idx[has_prev] - pos[prv[has_prev]]
    to_next = np.where(to_next > 70, np.nan, to_next)
    return to_next, since_last


def compute_stock_features_v3(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    earnings_dates: np.ndarray | None = None,
) -> pd.DataFrame:
    out = compute_features(df, benchmark_df)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    prev_close = close.shift(1)

    out["gap_pct"] = open_ / prev_close - 1.0
    out["overnight_ret_5d"] = np.log(open_ / prev_close).rolling(5).sum()
    out["intraday_ret_5d"] = np.log(close / open_).rolling(5).sum()
    out["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    out["mom_6_1"] = close.shift(21) / close.shift(126) - 1.0

    chg = close.diff().abs()
    out["er_20"] = (close - close.shift(20)).abs() / chg.rolling(20).sum().replace(0.0, np.nan)

    logc = np.log(close)
    tidx = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    r = logc.rolling(60).corr(tidx).clip(-0.9999, 0.9999)
    out["trend_t_60"] = r * np.sqrt(58.0 / (1.0 - r**2))

    atrp = out["atr_pct"]
    out["atr_pct_vol_60"] = atrp.rolling(60).std() / atrp.rolling(60).mean()

    ret = close.pct_change()
    b_close = benchmark_df["close"].reindex(df.index).ffill()
    mret = b_close.pct_change()
    cov = ret.rolling(60).cov(mret)
    var = mret.rolling(60).var()
    beta = cov / var.replace(0.0, np.nan)
    out["beta_60"] = beta
    resid = ret - beta.shift(1) * mret
    out["resid_ret_20"] = resid.rolling(20).sum()
    out["idio_vol_60"] = resid.rolling(60).std()

    if earnings_dates is not None and len(earnings_dates) > 0:
        ev = np.sort(np.asarray(earnings_dates, dtype="datetime64[ns]"))
        to_next, since_last = _trading_day_distance(df.index, ev)
        out["days_to_earn"] = to_next
        out["days_since_earn"] = since_last
        # reaction move: max |return| on the event date and the next trading day (BMO/AMC unknown)
        abs_r = (ret.abs() / atrp.shift(1)).to_numpy()
        pos = np.unique(np.searchsorted(df.index.values, ev))
        move = np.full(len(df), np.nan)
        for p in pos:
            if p < len(df):
                m = np.nanmax(abs_r[p : min(p + 2, len(df))])
                start = min(p + 2, len(df))
                move[start:] = m  # becomes known after the reaction window
        out["last_earn_move_atr"] = move
    else:
        out["days_to_earn"] = np.nan
        out["days_since_earn"] = np.nan
        out["last_earn_move_atr"] = np.nan
    return out


def add_cross_sectional_features(panel: pd.DataFrame, regime_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add per-date cross-sectional ranks, sector-relative and market-regime features."""
    g = panel.groupby("date", sort=False)
    for c in CS_RANK_SOURCE:
        panel[f"cs_{c}"] = g[c].rank(pct=True)

    sect_med20 = panel.groupby(["date", "sector"])["ret_20d"].transform("median")
    sect_med60 = panel.groupby(["date", "sector"])["ret_60d"].transform("median")
    panel["sect_rel_ret_20"] = panel["ret_20d"] - sect_med20
    panel["sect_rel_ret_60"] = panel["ret_60d"] - sect_med60
    sect_tbl = panel.groupby(["date", "sector"])["ret_20d"].median().rename("s20").reset_index()
    sect_tbl["sector_rank_20"] = sect_tbl.groupby("date")["s20"].rank(pct=True)
    panel = panel.merge(sect_tbl[["date", "sector", "sector_rank_20"]], on=["date", "sector"], how="left")

    g = panel.groupby("date", sort=False)
    daily = pd.DataFrame(
        {
            "breadth_50": g["dist_ema50"].apply(lambda s: float((s > 0).mean())),
            "hi_lo_net": g.apply(
                lambda d: float((d["dist_52w_high"] > -0.01).mean() - (d["dist_52w_low"] < 0.01).mean()),
                include_groups=False,
            ),
            "xs_dispersion_20": g["ret_20d"].std(),
        }
    )
    daily.index = pd.to_datetime(daily.index)

    if regime_df is not None and not regime_df.empty:
        reg = regime_df.copy().sort_index()
        spy = reg["spy_close"].astype(float)
        ema50 = spy.ewm(span=50, adjust=False).mean()
        ema200 = spy.ewm(span=200, adjust=False).mean()
        reg_feats = pd.DataFrame(index=reg.index)
        reg_feats["spy_dist_ema50"] = spy / ema50 - 1.0
        reg_feats["spy_dist_ema200"] = spy / ema200 - 1.0
        reg_feats["spy_rv20"] = spy.pct_change().rolling(20).std() * np.sqrt(252.0)
        reg_feats["vix"] = reg["vix"].astype(float)
        reg_feats["vix_chg5"] = reg["vix"].astype(float).pct_change(5)
        reg_feats["vix_ratio"] = reg["vix"].astype(float) / reg["vix3m"].astype(float)
        daily = daily.join(reg_feats.reindex(daily.index).ffill(), how="left")
    else:
        for c in ("spy_dist_ema50", "spy_dist_ema200", "spy_rv20", "vix", "vix_chg5", "vix_ratio"):
            daily[c] = np.nan

    daily = daily.reset_index().rename(columns={"index": "date"})
    panel = panel.merge(daily, on="date", how="left")
    return panel
