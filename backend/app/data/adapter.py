from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import DATA_DIR
from app.data.universe import FALLBACK_US_NAMES, FALLBACK_US_STOCKS, UniverseMember, fetch_us_universe


US_EASTERN = ZoneInfo("America/New_York")
BENCHMARK_SYMBOL = "^GSPC"  # S&P 500 Index
FALLBACK_BENCHMARK = "SPY"


class MarketDataAdapter(ABC):
    """Pluggable market data interface for US EOD and 15:50 ET provisional snapshots."""

    @abstractmethod
    def get_universe(self, limit: int = 550) -> list[UniverseMember]:
        ...

    @abstractmethod
    def load_history(self, symbols: Iterable[str], years: int = 10) -> dict[str, pd.DataFrame]:
        ...

    @abstractmethod
    def load_benchmark(self, years: int = 10) -> pd.DataFrame:
        ...

    def apply_provisional_snapshot(
        self,
        history: dict[str, pd.DataFrame],
        quotes: dict[str, dict[str, float]],
        trade_date: pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Overlay a 3:50 PM ET live quote snapshot onto historical EOD bars
        so the ML model treats today's provisional bar as EOD before the 4:00 PM ET close.
        """
        bar_date = (trade_date or pd.Timestamp(datetime.now(US_EASTERN).date())).normalize()
        updated: dict[str, pd.DataFrame] = {}
        for sym, df in history.items():
            if sym not in quotes or df.empty:
                updated[sym] = df
                continue
            q = quotes[sym]
            prov_row = pd.DataFrame(
                [
                    {
                        "open": float(q["open"]),
                        "high": float(q["high"]),
                        "low": float(q["low"]),
                        "close": float(q["ltp"]),
                        "volume": float(q["volume"]),
                    }
                ],
                index=pd.DatetimeIndex([bar_date], name="date"),
            )
            trimmed = df[df.index < bar_date]
            updated[sym] = pd.concat([trimmed, prov_row]).sort_index()
        return updated


class YFinanceUSAdapter(MarketDataAdapter):
    """
    Free public adapter using yfinance backed by a local SQLite database
    (`data_cache/market_data.db`). Only downloads incremental delta bars for symbols
    that already have historical data stored in SQLite.
    """

    def __init__(self, cache_dir: Path = DATA_DIR) -> None:
        self.cache_dir = cache_dir
        self.db_path = cache_dir / "market_data.db"
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS universe (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    sector TEXT NOT NULL,
                    index_membership TEXT NOT NULL,
                    is_shortable INTEGER NOT NULL
                )
                """
            )
            cols = {r[1] for r in conn.execute("PRAGMA table_info(universe)").fetchall()}
            if "name" not in cols:
                conn.execute("ALTER TABLE universe ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark (
                    date TEXT PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_date ON ohlcv(symbol, date)")

    def has_cached_universe(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM universe").fetchone()
            return bool(row and row[0] > 0)

    def get_universe(self, limit: int = 550) -> list[UniverseMember]:
        members = fetch_us_universe(limit=limit)
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO universe (symbol, name, sector, index_membership, is_shortable) VALUES (?, ?, ?, ?, ?)",
                [(m.symbol, m.name, m.sector, m.index_membership, int(m.is_shortable)) for m in members],
            )
        return members

    def load_cached_universe(self, limit: int = 550) -> list[UniverseMember]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, name, sector, index_membership, is_shortable FROM universe LIMIT ?", (limit,)
            ).fetchall()
        if rows:
            # If existing cached rows have empty names from before this update, refresh once
            if any(not str(r[1]).strip() for r in rows[:10]):
                return self.get_universe(limit=limit)
            return [
                UniverseMember(
                    symbol=str(r[0]),
                    name=str(r[1]) or FALLBACK_US_NAMES.get(str(r[0]), str(r[0])),
                    sector=str(r[2]),
                    index_membership=str(r[3]),
                    is_shortable=bool(r[4]),
                )
                for r in rows
            ]
        return self.get_universe(limit=limit)

    def _upsert_ohlcv(self, conn: sqlite3.Connection, sym: str, df: pd.DataFrame) -> None:
        rows = [
            (
                sym,
                idx.strftime("%Y-%m-%d"),
                float(r["open"]),
                float(r["high"]),
                float(r["low"]),
                float(r["close"]),
                float(r["volume"]),
            )
            for idx, r in df.iterrows()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv (symbol, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    def sync_symbols(self, symbols: list[str], years: int = 10, batch_size: int = 50) -> dict[str, int]:
        """Incremental SQLite sync: delta-only for existing symbols, full `years` for new ones."""
        today = datetime.now(US_EASTERN).date()
        end = today + timedelta(days=1)
        full_start = end - timedelta(days=int(years * 365.25) + 30)

        with self._connect() as conn:
            max_dates = dict(
                conn.execute("SELECT symbol, MAX(date) FROM ohlcv GROUP BY symbol").fetchall()
            )

        inc_symbols: list[str] = []
        full_symbols: list[str] = []
        oldest_inc_start = today - timedelta(days=5)

        for sym in symbols:
            last_dt_str = max_dates.get(sym)
            if last_dt_str:
                last_dt = datetime.strptime(last_dt_str, "%Y-%m-%d").date()
                cand_start = last_dt - timedelta(days=3)
                if cand_start < oldest_inc_start:
                    oldest_inc_start = cand_start
                inc_symbols.append(sym)
            else:
                full_symbols.append(sym)

        synced_inc = self._download_and_store_batches(inc_symbols, oldest_inc_start.isoformat(), end.isoformat(), batch_size)
        synced_full = self._download_and_store_batches(full_symbols, full_start.isoformat(), end.isoformat(), batch_size)
        self._sync_benchmark(full_start.isoformat(), end.isoformat())
        return {
            "incremental_updated": synced_inc,
            "full_downloaded": synced_full,
            "total_synced": synced_inc + synced_full,
        }

    def _download_and_store_batches(
        self, symbols: list[str], start_iso: str, end_iso: str, batch_size: int
    ) -> int:
        if not symbols:
            return 0
        synced = 0
        with self._connect() as conn:
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                raw = yf.download(
                    tickers=batch,
                    start=start_iso,
                    end=end_iso,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                if raw.empty:
                    continue

                for sym in batch:
                    try:
                        sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
                        clean = self._normalize_ohlcv(sub)
                        if not clean.empty:
                            self._upsert_ohlcv(conn, sym, clean)
                            synced += 1
                    except Exception:
                        continue
                conn.commit()
        return synced

    def _sync_benchmark(self, full_start: str, end: str) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(date) FROM benchmark").fetchone()
            start_iso = full_start
            if row and row[0]:
                last_dt = datetime.strptime(row[0], "%Y-%m-%d").date()
                start_iso = (last_dt - timedelta(days=3)).isoformat()

        for idx_sym in (BENCHMARK_SYMBOL, FALLBACK_BENCHMARK):
            try:
                raw = yf.download(
                    idx_sym,
                    start=start_iso,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                clean = self._normalize_ohlcv(raw)
                if not clean.empty:
                    rows = [
                        (
                            idx.strftime("%Y-%m-%d"),
                            float(r["open"]),
                            float(r["high"]),
                            float(r["low"]),
                            float(r["close"]),
                            float(r["volume"]),
                        )
                        for idx, r in clean.iterrows()
                    ]
                    with self._connect() as conn:
                        conn.executemany(
                            "INSERT OR REPLACE INTO benchmark (date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)",
                            rows,
                        )
                    return
            except Exception:
                continue

    def load_history(self, symbols: Iterable[str], years: int = 10) -> dict[str, pd.DataFrame]:
        cutoff_str = (pd.Timestamp.now().normalize() - pd.Timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
        sym_list = list(symbols)
        if not sym_list:
            return {}

        with self._connect() as conn:
            df_all = pd.read_sql_query(
                "SELECT symbol, date, open, high, low, close, volume FROM ohlcv WHERE date >= ? ORDER BY symbol, date",
                conn,
                params=(cutoff_str,),
            )

        if df_all.empty:
            return {}

        sym_set = set(sym_list)
        df_all = df_all[df_all["symbol"].isin(sym_set)]
        df_all["date"] = pd.to_datetime(df_all["date"])

        out: dict[str, pd.DataFrame] = {}
        for sym, grp in df_all.groupby("symbol", sort=False):
            if len(grp) >= 120:
                sub = grp[["date", "open", "high", "low", "close", "volume"]].set_index("date")
                out[str(sym)] = sub
        return out

    def load_benchmark(self, years: int = 10) -> pd.DataFrame:
        cutoff_str = (pd.Timestamp.now().normalize() - pd.Timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM benchmark WHERE date >= ? ORDER BY date",
                conn,
                params=(cutoff_str,),
            )
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        work = df.copy()
        if isinstance(work.columns, pd.MultiIndex):
            work.columns = [str(c[0]).lower() for c in work.columns]
        else:
            work.columns = [str(c).lower() for c in work.columns]
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in work.columns]
        if len(cols) < 5:
            return pd.DataFrame()
        clean = work[cols].dropna().copy()
        clean.index = pd.to_datetime(clean.index).tz_localize(None).normalize()
        clean.index.name = "date"
        clean = clean[~clean.index.duplicated(keep="last")].sort_index()
        clean = clean[clean["close"] > 0]
        return clean


class USBrokerAdapterStub(MarketDataAdapter):
    """
    Placeholder for Alpaca / Interactive Brokers / Polygon / Schwab live feed integration.
    Implement `fetch_provisional_quotes_1550()` to feed 3:50 PM ET LTP/OHLCV
    into `apply_provisional_snapshot()`.
    """

    def __init__(self, base_adapter: YFinanceUSAdapter) -> None:
        self.base = base_adapter

    def get_universe(self, limit: int = 550) -> list[UniverseMember]:
        return self.base.load_cached_universe(limit=limit)

    def load_history(self, symbols: Iterable[str], years: int = 10) -> dict[str, pd.DataFrame]:
        return self.base.load_history(symbols, years=years)

    def load_benchmark(self, years: int = 10) -> pd.DataFrame:
        return self.base.load_benchmark(years=years)

    def fetch_provisional_quotes_1550(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        raise NotImplementedError(
            "Connect Alpaca/IBKR/Polygon here to stream 3:50 PM ET quotes."
        )


def generate_synthetic_demo_universe(
    n_stocks: int = 40, years: int = 3
) -> tuple[list[UniverseMember], dict[str, pd.DataFrame], pd.DataFrame]:
    """Generate realistic synthetic US equities OHLCV data for instant offline testing."""
    rng = np.random.default_rng(77)
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=int(years * 252))
    bench_ret = rng.normal(0.00045, 0.0095, size=len(dates))
    bench_close = 5200.0 * np.exp(np.cumsum(bench_ret))
    bench_df = pd.DataFrame(
        {
            "open": bench_close * (1 + rng.normal(0, 0.002, len(dates))),
            "high": bench_close * (1 + np.abs(rng.normal(0.004, 0.003, len(dates)))),
            "low": bench_close * (1 - np.abs(rng.normal(0.004, 0.003, len(dates)))),
            "close": bench_close,
            "volume": rng.integers(50_000_000, 120_000_000, size=len(dates)).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )

    sample = FALLBACK_US_STOCKS[:n_stocks]
    members: list[UniverseMember] = []
    history: dict[str, pd.DataFrame] = {}

    for idx, (sym, sec, tag) in enumerate(sample):
        beta = rng.uniform(0.75, 1.55)
        idio = rng.normal(0.0002, 0.016, size=len(dates))
        regime = np.sin(np.linspace(0, 10 * np.pi + idx, len(dates))) * 0.0012
        ret = beta * bench_ret + idio + regime
        close = rng.uniform(60, 650) * np.exp(np.cumsum(ret))
        spread = np.abs(rng.normal(0.009, 0.005, size=len(dates)))
        vol_base = rng.uniform(2e6, 2.5e7)
        vol = vol_base * np.exp(np.abs(ret) * 28 + rng.normal(0, 0.35, len(dates)))
        df = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.004, len(dates))),
                "high": close * (1 + spread),
                "low": close * (1 - spread),
                "close": close,
                "volume": vol,
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        history[sym] = df
        members.append(
            UniverseMember(
                symbol=sym,
                name=FALLBACK_US_NAMES.get(sym, sym),
                sector=sec,
                index_membership=tag,
                is_shortable=True,
            )
        )

    return members, history, bench_df
