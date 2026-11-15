"""Download market-regime series (^VIX, ^VIX3M, SPY) and historical earnings dates.

Outputs (in data_cache/):
  - regime_daily.csv  : date, vix, vix3m, spy_close
  - earnings_dates.csv: symbol, earnings_date

Usage:  python scripts/fetch_regime_earnings.py [--skip-earnings]
yfinance earnings dates are known to be incomplete; treat them as best-effort.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from app.config import DATA_DIR
from app.data.adapter import YFinanceUSAdapter


def fetch_regime(years: int = 11) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    out = {}
    for sym, col in (("^VIX", "vix"), ("^VIX3M", "vix3m"), ("SPY", "spy_close")):
        raw = yf.download(sym, start=start, interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"Failed to download {sym}")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        out[col] = close.astype(float)
    df = pd.DataFrame(out).sort_index()
    df.index.name = "date"
    return df


def _earn_one(sym: str) -> list[tuple[str, pd.Timestamp]]:
    for attempt in range(3):
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=60)
            if ed is None or ed.empty:
                return []
            idx = pd.to_datetime(ed.index)
            if idx.tz is not None:
                idx = idx.tz_convert("America/New_York").tz_localize(None)
            return [(sym, d.normalize()) for d in idx]
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def fetch_earnings(symbols: list[str]) -> pd.DataFrame:
    rows: list[tuple[str, pd.Timestamp]] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_earn_one, s): s for s in symbols}
        for i, f in enumerate(as_completed(futs)):
            rows.extend(f.result())
            if (i + 1) % 50 == 0:
                print(f"  earnings: {i + 1}/{len(symbols)} symbols", flush=True)
    df = pd.DataFrame(rows, columns=["symbol", "earnings_date"]).drop_duplicates()
    return df.sort_values(["symbol", "earnings_date"])


if __name__ == "__main__":
    reg = fetch_regime()
    reg.to_csv(DATA_DIR / "regime_daily.csv")
    print("regime rows:", len(reg), reg.index.min().date(), reg.index.max().date())
    if "--skip-earnings" not in sys.argv:
        syms = [m.symbol for m in YFinanceUSAdapter().load_cached_universe(limit=550)]
        ed = fetch_earnings(syms)
        ed.to_csv(DATA_DIR / "earnings_dates.csv", index=False)
        print("earnings rows:", len(ed), "symbols:", ed["symbol"].nunique() if not ed.empty else 0)
