from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from app.config import DATA_DIR, DEFAULT_CONFIG, ScreenerConfig
from app.data.adapter import (
    YFinanceUSAdapter,
    generate_synthetic_demo_universe,
)
from app.data.universe import UniverseMember
from app.features.indicators import compute_features
from app.ledger import (
    LEDGER_CSV_PATH,
    get_trade_chart_data,
    load_ledger,
    record_and_update_ledger,
    render_ledger_html,
    sort_ledger_by_win_prob,
)
from app.ml.intraday_entry import analyze_hourly_entry
from app.ml.pipeline import (
    TrainedArtifacts,
    load_artifacts,
    scan_latest,
    score_single_stock,
    train_and_save_models,
)


app = FastAPI(title="US Trading App — S&P 500 & Nasdaq 100 ML Screener", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adapter = YFinanceUSAdapter()
HOLDINGS_PATH = DATA_DIR / "holdings.json"


class AppState:
    config: ScreenerConfig = DEFAULT_CONFIG
    data_mode: str = "DEMO"  # DEMO | LIVE_EOD | PROVISIONAL_1550
    universe: list[UniverseMember] = []
    history: dict[str, pd.DataFrame] = {}
    benchmark_df: pd.DataFrame = pd.DataFrame()
    artifacts: TrainedArtifacts | None = None
    last_scan: dict[str, Any] | None = None
    index_outlook: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = [
        {"symbol": "AAPL", "entry_price": 224.0, "qty": 20},
        {"symbol": "NVDA", "entry_price": 135.0, "qty": 30},
        {"symbol": "MSFT", "entry_price": 418.0, "qty": 10},
    ]


state = AppState()


def _load_holdings() -> None:
    if HOLDINGS_PATH.exists():
        try:
            state.holdings = json.loads(HOLDINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass


def _save_holdings() -> None:
    HOLDINGS_PATH.write_text(json.dumps(state.holdings, indent=2), encoding="utf-8")


def _bootstrap_state() -> None:
    _load_holdings()
    cached_univ = (
        adapter.load_cached_universe(limit=state.config.universe_limit)
        if adapter.has_cached_universe()
        else []
    )
    if cached_univ:
        hist = adapter.load_history([m.symbol for m in cached_univ], years=state.config.history_years)
        bench = adapter.load_benchmark(years=state.config.history_years)
        if hist:
            state.universe = cached_univ
            state.history = hist
            state.benchmark_df = bench
            state.data_mode = "LIVE_EOD"
            state.artifacts = load_artifacts()
            if state.artifacts is not None:
                state.last_scan = scan_latest(
                    state.artifacts,
                    state.universe,
                    state.history,
                    state.benchmark_df,
                    state.config,
                    held_symbols=[h["symbol"] for h in state.holdings],
                    scan_mode=state.data_mode,
                )
                record_and_update_ledger(state.last_scan, state.history, state.config.horizon_days)
                return

    univ, hist, bench = generate_synthetic_demo_universe(n_stocks=40, years=3)
    state.universe = univ
    state.history = hist
    state.benchmark_df = bench
    state.data_mode = "DEMO"
    state.artifacts = load_artifacts()
    if state.artifacts is None or state.artifacts.data_mode != "DEMO":
        state.artifacts = train_and_save_models(
            state.universe, state.history, state.benchmark_df, state.config, data_mode="DEMO"
        )
    state.last_scan = scan_latest(
        state.artifacts,
        state.universe,
        state.history,
        state.benchmark_df,
        state.config,
        held_symbols=[h["symbol"] for h in state.holdings],
        scan_mode="DEMO",
    )


@app.on_event("startup")
def on_startup() -> None:
    _bootstrap_state()


class SyncRequest(BaseModel):
    universe_limit: int | None = None
    years: int | None = None


class ScanRequest(BaseModel):
    simulate_1550_provisional: bool = False
    provisional_quotes: dict[str, dict[str, float]] | None = None


class HoldingItem(BaseModel):
    symbol: str
    entry_price: float = 0.0
    qty: int = 1


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "data_mode": state.data_mode,
        "universe_loaded": len(state.history),
        "model_trained_at": state.artifacts.trained_at if state.artifacts else None,
        "model_data_mode": state.artifacts.data_mode if state.artifacts else None,
    }


@app.get("/api/config")
def get_config() -> ScreenerConfig:
    return state.config


@app.post("/api/config")
def update_config(cfg: ScreenerConfig) -> ScreenerConfig:
    state.config = cfg
    return state.config


def _compute_single_index_outlook(sym: str, name: str, df: pd.DataFrame, horizon_days: int) -> dict[str, Any] | None:
    if df is None or len(df) < 50:
        return None
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    cur = float(c.iloc[-1])
    prev = float(c.iloc[-2]) if len(c) >= 2 else cur
    chg_1d = (cur / prev - 1.0) * 100.0 if prev else 0.0
    ret_20d = (cur / float(c.iloc[-21]) - 1.0) if len(c) >= 21 else 0.0
    ema_20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    ema_50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14).mean().iloc[-1])
    if not np.isfinite(atr_14) or atr_14 <= 0:
        atr_14 = cur * 0.01

    # Damped trend + EMA regime projection over horizon_days (15 trading days)
    trend_component = (ret_20d / 20.0) * horizon_days * 0.5
    regime_component = 0.008 if cur > ema_20 > ema_50 else (-0.008 if cur < ema_20 < ema_50 else 0.0)
    sigma_h = (atr_14 * np.sqrt(horizon_days)) / cur
    est_ret = float(np.clip(trend_component + regime_component, -1.5 * sigma_h, 1.5 * sigma_h))
    est_pct = round(est_ret * 100.0, 2)
    proj_val = round(cur * (1.0 + est_ret), 2)
    range_low = round(cur - atr_14 * np.sqrt(horizon_days), 2)
    range_high = round(cur + atr_14 * np.sqrt(horizon_days), 2)
    direction = "BULLISH" if est_pct >= 0.4 else ("BEARISH" if est_pct <= -0.4 else "NEUTRAL")
    as_of = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else ""

    return {
        "symbol": sym,
        "name": name,
        "value": round(cur, 2),
        "change_1d_pct": round(chg_1d, 2),
        "projected_value": proj_val,
        "projected_change_pct": est_pct,
        "range_low": range_low,
        "range_high": range_high,
        "direction": direction,
        "horizon_days": horizon_days,
        "as_of": as_of,
    }


def _refresh_index_outlook(horizon_days: int) -> list[dict[str, Any]]:
    import yfinance as yf

    specs = [("^GSPC", "S&P 500"), ("^NDX", "Nasdaq 100")]
    out: list[dict[str, Any]] = []
    for sym, name in specs:
        try:
            raw = yf.download(sym, period="1y", interval="1d", progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [str(col[0]).lower() for col in raw.columns]
                else:
                    raw.columns = [str(col).lower() for col in raw.columns]
                item = _compute_single_index_outlook(sym, name, raw, horizon_days)
                if item:
                    out.append(item)
        except Exception:
            continue
    if out:
        state.index_outlook = out
    elif not state.index_outlook and state.benchmark_df is not None and not state.benchmark_df.empty:
        fb = _compute_single_index_outlook("SPY", "S&P 500 (SPY)", state.benchmark_df, horizon_days)
        if fb:
            state.index_outlook = [fb]
    return state.index_outlook


@app.post("/api/pipeline/sync")
def sync_data(req: SyncRequest | None = None) -> dict[str, Any]:
    limit = (req.universe_limit if req and req.universe_limit else state.config.universe_limit)
    years = (req.years if req and req.years else state.config.history_years)

    univ = adapter.get_universe(limit=limit)
    sync_stats = adapter.sync_symbols([m.symbol for m in univ], years=years)
    hist = adapter.load_history([m.symbol for m in univ], years=years)
    if not hist:
        raise HTTPException(
            status_code=502,
            detail="Could not download US OHLCV from Yahoo Finance. Check network access.",
        )
    bench = adapter.load_benchmark(years=years)

    state.universe = univ
    state.history = hist
    state.benchmark_df = bench
    state.data_mode = "LIVE_EOD"
    if state.last_scan is not None:
        record_and_update_ledger(state.last_scan, state.history, state.config.horizon_days)
    _refresh_index_outlook(state.config.horizon_days)
    return {
        "status": "synced",
        "universe_requested": len(univ),
        "symbols_loaded": len(hist),
        "incremental_updated": sync_stats["incremental_updated"],
        "full_downloaded": sync_stats["full_downloaded"],
        "data_mode": state.data_mode,
    }


@app.post("/api/pipeline/train")
def train_models() -> dict[str, Any]:
    if not state.history:
        raise HTTPException(status_code=400, detail="No market data loaded to train on.")
    artifacts = train_and_save_models(
        state.universe,
        state.history,
        state.benchmark_df,
        state.config,
        data_mode=state.data_mode,
    )
    state.artifacts = artifacts
    state.last_scan = scan_latest(
        artifacts,
        state.universe,
        state.history,
        state.benchmark_df,
        state.config,
        held_symbols=[h["symbol"] for h in state.holdings],
        scan_mode=state.data_mode,
    )
    if state.data_mode == "LIVE_EOD":
        record_and_update_ledger(state.last_scan, state.history, state.config.horizon_days)
    return {
        "status": "trained",
        "trained_at": artifacts.trained_at,
        "data_mode": artifacts.data_mode,
        "backtest_metrics": artifacts.backtest.get("metrics", {}),
    }


@app.post("/api/scan")
def run_scan(req: ScanRequest | None = None) -> dict[str, Any]:
    if state.artifacts is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    hist = state.history
    mode = state.data_mode
    if req and req.provisional_quotes:
        hist = adapter.apply_provisional_snapshot(hist, req.provisional_quotes)
        mode = "PROVISIONAL_1550"
    elif req and req.simulate_1550_provisional:
        rng = np.random.default_rng(int(pd.Timestamp.now().timestamp()) % 10000)
        sim_quotes: dict[str, dict[str, float]] = {}
        for sym, df in hist.items():
            last = df.iloc[-1]
            c = float(last["close"]) * float(1.0 + rng.normal(0, 0.008))
            sim_quotes[sym] = {
                "open": float(last["open"]),
                "high": max(float(last["high"]), c),
                "low": min(float(last["low"]), c),
                "ltp": c,
                "volume": float(last["volume"]) * float(rng.uniform(0.85, 1.35)),
            }
        hist = adapter.apply_provisional_snapshot(hist, sim_quotes)
        mode = "PROVISIONAL_1550"

    res = scan_latest(
        state.artifacts,
        state.universe,
        hist,
        state.benchmark_df,
        state.config,
        held_symbols=[h["symbol"] for h in state.holdings],
        scan_mode=mode,
    )
    if mode == "LIVE_EOD":
        record_and_update_ledger(res, hist, state.config.horizon_days)
    res["market_indices"] = state.index_outlook or _refresh_index_outlook(state.config.horizon_days)
    state.last_scan = res
    return res


@app.get("/api/scan/latest")
def get_latest_scan() -> dict[str, Any]:
    if state.last_scan is None:
        return run_scan()
    if not state.last_scan.get("market_indices"):
        state.last_scan["market_indices"] = state.index_outlook or _refresh_index_outlook(state.config.horizon_days)
    return state.last_scan


@app.get("/api/universe")
def get_universe_list() -> list[dict[str, str]]:
    return [{"symbol": m.symbol, "name": m.name} for m in state.universe if m.symbol in state.history]


@app.get("/api/analyze")
def analyze_stock(q: str) -> dict[str, Any]:
    if state.artifacts is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")
    res = score_single_stock(
        state.artifacts,
        state.universe,
        state.history,
        state.benchmark_df,
        state.config,
        query=q,
    )
    if res is None:
        raise HTTPException(status_code=404, detail=f"Stock '{q}' not found in loaded universe.")
    return res


@app.get("/api/entry-timing/scan")
def scan_hourly_entry_timing() -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if state.last_scan is None:
        get_latest_scan()
    targets = [
        {"symbol": item["symbol"], "name": item.get("name", item["symbol"]), "side": item["side"]}
        for item in (state.last_scan.get("top_longs") or []) + (state.last_scan.get("top_shorts") or [])
    ]
    items: dict[str, Any] = {}
    enter_now: list[dict[str, Any]] = []

    def _check(t: dict[str, str]) -> tuple[str, dict[str, Any]]:
        res = analyze_hourly_entry(t["symbol"], side=t["side"])
        res["name"] = t["name"]
        return f"{t['side']}:{t['symbol']}", res

    with ThreadPoolExecutor(max_workers=6) as pool:
        for key, res in pool.map(_check, targets):
            items[key] = res
            if res.get("status") == "OK" and res.get("signal") == "ENTER_NOW":
                enter_now.append(res)

    return {
        "checked_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "count": len(items),
        "enter_now": enter_now,
        "items": items,
    }


@app.get("/api/entry-timing/{symbol}")
def get_hourly_entry_timing(symbol: str, side: str = "LONG") -> dict[str, Any]:
    return analyze_hourly_entry(symbol.upper().strip(), side=side)


@app.get("/api/stock/{symbol}")
def get_stock_detail(symbol: str, days: int = 120) -> dict[str, Any]:
    sym = symbol.upper().strip()
    if sym not in state.history:
        raise HTTPException(status_code=404, detail=f"Symbol {sym} not in loaded universe.")
    df = state.history[sym].tail(days)
    feats = compute_features(state.history[sym], state.benchmark_df).tail(days)

    candles = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": round(float(r["open"]), 2),
            "high": round(float(r["high"]), 2),
            "low": round(float(r["low"]), 2),
            "close": round(float(r["close"]), 2),
            "volume": int(r["volume"]),
            "rsi_14": round(float(feats.loc[idx, "rsi_14"]), 1)
            if pd.notna(feats.loc[idx, "rsi_14"])
            else None,
        }
        for idx, r in df.iterrows()
    ]
    return {"symbol": sym, "candles": candles}


@app.get("/api/backtest")
def get_backtest() -> dict[str, Any]:
    if state.artifacts is None:
        return {"equity_curve": [], "metrics": {}, "feature_importance": []}
    return {
        "equity_curve": state.artifacts.backtest.get("equity_curve", []),
        "metrics": state.artifacts.backtest.get("metrics", {}),
        "screener_eval": state.artifacts.backtest.get("screener_eval", {}),
        "feature_importance": state.artifacts.feature_importance,
        "trained_at": state.artifacts.trained_at,
        "data_mode": state.artifacts.data_mode,
    }


@app.get("/api/holdings")
def get_holdings() -> list[dict[str, Any]]:
    return state.holdings


@app.post("/api/holdings")
def add_holding(item: HoldingItem) -> list[dict[str, Any]]:
    sym = item.symbol.upper().strip()
    state.holdings = [h for h in state.holdings if h["symbol"] != sym]
    state.holdings.append({"symbol": sym, "entry_price": item.entry_price, "qty": item.qty})
    _save_holdings()
    return state.holdings


@app.delete("/api/holdings/{symbol}")
def delete_holding(symbol: str) -> list[dict[str, Any]]:
    sym = symbol.upper().strip()
    state.holdings = [h for h in state.holdings if h["symbol"] != sym]
    _save_holdings()
    return state.holdings


@app.get("/api/ledger")
def get_ledger() -> dict[str, Any]:
    df = load_ledger()
    df = sort_ledger_by_win_prob(df[df["scan_mode"] == "LIVE_EOD"])
    rows = df.to_dict(orient="records")
    return {
        "count": len(rows),
        "longs": [r for r in rows if r["side"] == "LONG"],
        "shorts": [r for r in rows if r["side"] == "SHORT"],
        "rows": rows,
    }


@app.get("/api/ledger/chart/{symbol}")
def get_ledger_trade_chart(symbol: str, rec_date: str, side: str = "LONG") -> dict[str, Any]:
    data = get_trade_chart_data(symbol, rec_date, side, state.history)
    if data is None:
        raise HTTPException(status_code=404, detail="Tracked trade not found.")
    return data


@app.get("/api/ledger/view", response_class=HTMLResponse)
def view_ledger_spreadsheet() -> HTMLResponse:
    df = load_ledger()
    return HTMLResponse(content=render_ledger_html(df))


@app.get("/api/ledger/csv")
def download_ledger_csv() -> FileResponse:
    if not LEDGER_CSV_PATH.exists():
        load_ledger()
    return FileResponse(
        path=str(LEDGER_CSV_PATH),
        media_type="text/csv",
        filename="recommendations_log.csv",
    )
