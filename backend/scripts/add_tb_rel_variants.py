"""Add market-relative Triple-Barrier ranker variants (tb_rel_base, tb_rel_full, tb_ens)
with daily-IC early stopping to oos_preds.parquet and re-run evaluation.
"""
from __future__ import annotations

import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.features.features_v3 import CS_RANK_COLS, V3_FEATURE_COLS
from app.features.indicators import FEATURE_COLS as BASE_FEATURE_COLS
from app.ml.research_v3 import (
    MAIN_H,
    MAIN_STOP,
    MAIN_TGT,
    fit_lgb_variant,
    fit_ridge,
    make_half_year_folds,
    r_col,
    rank_in_date,
)

OUT = DATA_DIR / "v3"
LR = r_col(MAIN_H, MAIN_STOP, MAIN_TGT)

# Pure cross-sectional features (exclude date-constant regime levels that cannot rank within a date)
XS_V3_FEATURES = [c for c in V3_FEATURE_COLS if c not in (
    "spy_dist_ema50", "spy_dist_ema200", "spy_rv20", "vix", "vix_chg5", "vix_ratio", "breadth_50", "hi_lo_net", "xs_dispersion_20"
)]


def main() -> None:
    panel = pd.read_parquet(OUT / "panel.parquet")
    base = panel[panel["avg_dollar_vol_m"] >= 25.0].copy()
    base["tb_pct_15"] = base.groupby("date")[LR].rank(pct=True)

    folds = make_half_year_folds(base["date"].values)
    all_dates = np.sort(base["date"].unique())
    train_dates = set(all_dates[::5])

    oos = pd.read_parquet(OUT / "oos_preds.parquet")
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))

    new_frames = []
    for fd, fmeta in zip(folds, meta["folds"]):
        t0 = time.time()
        te = base[(base["date"] >= fd.test_start) & (base["date"] <= fd.test_end)].copy()
        tr = base[(base["date"] <= fd.train_end) & (base["date"].isin(train_dates))].dropna(subset=["tb_pct_15", "target_pct_15"])

        # 1) Relative close-to-close target with XS-only V3 features + daily_ic early stopping
        r_xs = fit_lgb_variant(tr, te, XS_V3_FEATURES, "target_pct_15", MAIN_H)
        # 2) Relative Triple-Barrier R target with XS-only V3 features + daily_ic early stopping
        r_tb = fit_lgb_variant(tr, te, XS_V3_FEATURES, "tb_pct_15", MAIN_H)
        fmeta["tb_rel_xs"] = {"iters": r_tb.best_iter, "q90": r_tb.valid_q90}

        # 3) 5-seed + Ridge ensemble on tb_pct_15
        seeds = [r_tb.pred]
        for sd in range(1, 5):
            seeds.append(fit_lgb_variant(tr, te, XS_V3_FEATURES, "tb_pct_15", MAIN_H, seed=sd, fixed_iter=r_tb.best_iter).pred)
        d = te["date"]
        lgb_rank = np.mean([rank_in_date(d, p) for p in seeds], axis=0)
        ridge_rank = rank_in_date(d, fit_ridge(tr, te, "tb_pct_15"))
        mom = te["mom_12_1"].to_numpy(float)
        mom_rank = rank_in_date(d, np.where(np.isfinite(mom), mom, np.nanmedian(mom)))
        tb_ens = 0.6 * lgb_rank + 0.25 * ridge_rank + 0.15 * mom_rank
        fmeta["tb_ens"] = {"iters": r_tb.best_iter, "q90": 0.90}

        sub = pd.DataFrame({
            "date": te["date"].values,
            "symbol": te["symbol"].values,
            "rel_xs_icstop": r_xs.pred,
            "tb_rel_xs": r_tb.pred,
            "tb_ens": tb_ens,
        })
        new_frames.append(sub)
        print(f"fold {fd.fid}: iters(rel_xs)={r_xs.best_iter} iters(tb_rel)={r_tb.best_iter} ({time.time()-t0:.1f}s)", flush=True)

    add_df = pd.concat(new_frames, ignore_index=True)
    for col in ("rel_xs_icstop", "tb_rel_xs", "tb_ens"):
        if col in oos.columns:
            oos = oos.drop(columns=[col])
    oos = oos.merge(add_df, on=["date", "symbol"], how="left")
    oos.to_parquet(OUT / "oos_preds.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("updated oos_preds.parquet", oos.shape, flush=True)


if __name__ == "__main__":
    main()
