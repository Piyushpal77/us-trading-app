"""Run the Claude-feedback V3 A/B experiments end-to-end.

  python scripts/run_v3_experiments.py build     # build + cache panel  (data_cache/v3/panel.parquet)
  python scripts/run_v3_experiments.py train     # walk-forward variants -> oos_preds.parquet + meta.json
  python scripts/run_v3_experiments.py eval      # metrics + selection backtest -> results.json
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

from app.config import DATA_DIR
from app.data.adapter import YFinanceUSAdapter

OUT = DATA_DIR / "v3"
OUT.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build() -> None:
    from app.ml.research_v3 import build_v3_panel

    db = YFinanceUSAdapter()
    u = db.load_cached_universe(limit=550)
    h = db.load_history([m.symbol for m in u])
    b = db.load_benchmark()
    if b.empty:
        raise RuntimeError("Benchmark table is empty - refusing to build features.")
    reg = pd.read_csv(DATA_DIR / "regime_daily.csv", parse_dates=["date"]).set_index("date")
    earn = pd.read_csv(DATA_DIR / "earnings_dates.csv", parse_dates=["earnings_date"])
    t = time.time()
    panel = build_v3_panel(u, h, b, reg, earn)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    _log(f"panel {panel.shape} {panel['date'].min().date()}..{panel['date'].max().date()} in {time.time() - t:.0f}s")


def train() -> None:
    from app.ml.research_v3 import run_variants

    panel = pd.read_parquet(OUT / "panel.parquet")
    _log(f"loaded panel {panel.shape}")
    oos, meta = run_variants(panel, step=5, log=_log)
    oos.to_parquet(OUT / "oos_preds.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _log(f"saved oos {oos.shape}")


def evaluate() -> None:
    from app.ml.eval_v3 import run_full_evaluation

    oos = pd.read_parquet(OUT / "oos_preds.parquet")
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    reg = pd.read_csv(DATA_DIR / "regime_daily.csv", parse_dates=["date"]).set_index("date")
    panel_ret = pd.read_parquet(OUT / "panel.parquet", columns=["date", "symbol", "ret_1d", "long_tp", "short_tp"])
    if "long_tp" not in oos.columns:
        oos = oos.merge(panel_ret[["date", "symbol", "long_tp", "short_tp"]], on=["date", "symbol"], how="left")
    results = run_full_evaluation(oos, meta, reg, panel_ret[["date", "symbol", "ret_1d"]], log=_log)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _log("saved results.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("build", "all"):
        build()
    if cmd in ("train", "all"):
        train()
    if cmd in ("eval", "all"):
        evaluate()
