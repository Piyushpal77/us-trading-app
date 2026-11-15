"""V3 research harness (Claude feedback): builds the V3 panel, runs A/B model variants on identical
half-yearly purged walk-forward folds, and stores out-of-sample predictions for evaluation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.config import ScreenerConfig
from app.data.universe import UniverseMember
from app.features.features_v3 import (
    CS_RANK_COLS,
    REGIME_COLS,
    SECTOR_ALIASES,
    V3_FEATURE_COLS,
    add_cross_sectional_features,
    compute_stock_features_v3,
)
from app.features.indicators import FEATURE_COLS as BASE_FEATURE_COLS
from app.ml.labeling_v3 import add_relative_targets, forward_return, triple_barrier_r

MAIN_H, MAIN_STOP, MAIN_TGT = 15, 2.0, 2.5
HORIZONS = [5, 10, 15, 20]
BARRIER_GRID = [(s, t) for s in (1.5, 2.0, 2.5) for t in (1.5, 2.0, 2.5)]
PURGE_DAYS = 21  # >= max horizon (20) + 1


def r_col(h: int, s: float, t: float) -> str:
    return f"lr_{h}_s{s}_t{t}"


def sr_col(h: int, s: float, t: float) -> str:
    return f"sr_{h}_s{s}_t{t}"


# --------------------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------------------
def build_v3_panel(
    universe: list[UniverseMember],
    history: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    regime_df: pd.DataFrame | None,
    earnings_df: pd.DataFrame | None,
    include_labels: bool = True,
    min_bars: int = 300,
) -> pd.DataFrame:
    meta = {m.symbol: m for m in universe}
    earn_map: dict[str, np.ndarray] = {}
    if earnings_df is not None and not earnings_df.empty:
        for sym, g in earnings_df.groupby("symbol"):
            earn_map[str(sym)] = pd.to_datetime(g["earnings_date"]).values

    frames: list[pd.DataFrame] = []
    for sym, df in history.items():
        if len(df) < min_bars:
            continue
        f = compute_stock_features_v3(df, benchmark_df, earn_map.get(sym))
        m = meta.get(sym)
        f["symbol"] = sym
        f["date"] = f.index
        sec = m.sector if m else "Equities"
        f["sector"] = SECTOR_ALIASES.get(sec, sec)
        f["is_shortable"] = bool(m.is_shortable) if m else True
        if include_labels:
            for h in HORIZONS:
                f[f"fwd_ret_{h}"] = forward_return(df["close"].astype(float), h)
            atr = f["atr_14"]
            for h in HORIZONS:
                grid = BARRIER_GRID if h == MAIN_H else [(MAIN_STOP, MAIN_TGT)]
                for s, t in grid:
                    lr, sr, ltp, stp = triple_barrier_r(df, atr, h, s, t)
                    f[r_col(h, s, t)] = lr
                    f[sr_col(h, s, t)] = sr
                    if (h, s, t) == (MAIN_H, MAIN_STOP, MAIN_TGT):
                        f["long_tp"] = ltp
                        f["short_tp"] = stp
        f = f.iloc[252:]
        frames.append(f.reset_index(drop=True))

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = add_cross_sectional_features(panel, regime_df)
    if include_labels:
        panel = add_relative_targets(panel, HORIZONS)
        lr = panel[r_col(MAIN_H, MAIN_STOP, MAIN_TGT)]
        sr = panel[sr_col(MAIN_H, MAIN_STOP, MAIN_TGT)]
        risk_pct = MAIN_STOP * panel["atr_pct"]
        # columns used by the legacy 3-class baseline (raw returns + classes 0=SL,1=TO,2=TP)
        panel["long_ret"] = lr * risk_pct
        panel["short_ret"] = sr * risk_pct
        panel["c_long"] = np.where(panel["long_tp"] == 1, 2.0, np.where(lr <= -0.999, 0.0, 1.0))
        panel["c_short"] = np.where(panel["short_tp"] == 1, 2.0, np.where(sr <= -0.999, 0.0, 1.0))
        panel.loc[lr.isna(), ["c_long"]] = np.nan
        panel.loc[sr.isna(), ["c_short"]] = np.nan

    fcols = [c for c in panel.columns if panel[c].dtype == np.float64]
    panel[fcols] = panel[fcols].astype(np.float32)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------------------
# Folds and weights
# --------------------------------------------------------------------------------------
@dataclass
class Fold:
    fid: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_end: pd.Timestamp  # purged


def make_half_year_folds(dates: np.ndarray, first_test: str = "2020-01-01") -> list[Fold]:
    u = np.sort(pd.to_datetime(np.unique(dates)))
    last = u[-1]
    starts = pd.date_range(first_test, last, freq="6MS")
    folds: list[Fold] = []
    for i, st in enumerate(starts):
        en = starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(starts) else last
        idx = np.searchsorted(u, st)
        if idx >= len(u):
            break
        tr_end_idx = idx - PURGE_DAYS - 1
        if tr_end_idx < 250:
            continue
        folds.append(Fold(i, pd.Timestamp(u[idx]), pd.Timestamp(en), pd.Timestamp(u[tr_end_idx])))
    return folds


def sample_weights(dates: pd.Series, horizon: int, ref_date: pd.Timestamp, half_life_years: float = 3.0) -> np.ndarray:
    """Label-uniqueness (1 / mean concurrency over the label window) x exponential time decay."""
    ud = np.sort(dates.unique())
    all_days = pd.bdate_range(ud[0], pd.Timestamp(ud[-1]) + pd.tseries.offsets.BDay(horizon + 1))
    pos = np.searchsorted(all_days.values, ud)
    conc = np.zeros(len(all_days) + horizon + 2)
    for p in pos:
        conc[p + 1 : p + horizon + 1] += 1
    uniq = np.array([np.mean(1.0 / np.maximum(conc[p + 1 : p + horizon + 1], 1)) for p in pos])
    uniq_map = dict(zip(ud, uniq))
    u = dates.map(uniq_map).to_numpy(float)
    age_years = (ref_date - dates).dt.days.to_numpy(float) / 365.25
    decay = 0.5 ** (age_years / half_life_years)
    w = u * decay
    return w / w.mean()


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def lgb_params(seed: int = 0, objective: str = "regression") -> dict[str, Any]:
    p = dict(
        objective=objective,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        min_data_in_leaf=500,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=10.0,
        extra_trees=True,
        seed=seed,
        num_threads=12,
        verbose=-1,
    )
    if objective == "lambdarank":
        p.update(metric="ndcg", eval_at=[10, 50], lambdarank_truncation_level=60, label_gain=list(range(0, 32)))
    else:
        p.update(metric="l2")
    return p


def _split_fit_valid(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ud = np.sort(tr["date"].unique())
    v_start = pd.Timestamp(ud[-1]) - pd.Timedelta(days=365)
    v_idx = np.searchsorted(ud, v_start)
    fit_end = ud[max(0, v_idx - PURGE_DAYS)]
    return tr[tr["date"] <= fit_end], tr[tr["date"] >= ud[v_idx]]


def _daily_ic(df: pd.DataFrame, score: np.ndarray, target: str) -> float:
    tmp = pd.DataFrame({"d": df["date"].values, "s": score, "y": df[target].values}).dropna()
    ics = tmp.groupby("d").apply(lambda g: g["s"].rank().corr(g["y"].rank()), include_groups=False)
    return float(ics.mean())


def make_daily_ic_feval(dates: np.ndarray, y: np.ndarray) -> Callable:
    """LightGBM feval: mean over dates of the within-date correlation between prediction and target."""
    codes = pd.factorize(dates)[0]
    cnt = np.bincount(codes).astype(float)
    y = np.nan_to_num(y.astype(float), nan=0.5)
    y_dm = y - (np.bincount(codes, y) / cnt)[codes]
    syy = np.bincount(codes, y_dm * y_dm)

    def feval(preds: np.ndarray, _ds: lgb.Dataset):
        p = preds - (np.bincount(codes, preds) / cnt)[codes]
        sxy = np.bincount(codes, p * y_dm)
        sxx = np.bincount(codes, p * p)
        ic = sxy / np.sqrt(sxx * syy + 1e-18)
        return "daily_ic", float(np.mean(ic)), True

    return feval


@dataclass
class FitResult:
    pred: np.ndarray
    best_iter: int = 0
    valid_q90: float = np.nan
    features: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def fit_lgb_variant(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    features: list[str],
    target: str,
    horizon: int,
    seed: int = 0,
    objective: str = "regression",
    prune: bool = False,
    fixed_iter: int | None = None,
) -> FitResult:
    def _ds(d: pd.DataFrame, feats: list[str], w: np.ndarray | None) -> lgb.Dataset:
        y = d[target].to_numpy(float)
        if objective == "lambdarank":
            y = np.clip(np.floor(d[target].to_numpy(float) * 20), 0, 19).astype(int)  # 20 relevance bins
            grp = d.groupby("date", sort=False).size().to_numpy()
            return lgb.Dataset(d[feats], label=y, group=grp, weight=None, free_raw_data=True)
        return lgb.Dataset(d[feats], label=y - 0.5, weight=w, free_raw_data=True)

    ref = pd.Timestamp(tr["date"].max())
    feats = list(features)
    best_iter = fixed_iter or 0
    valid_q90 = np.nan
    if fixed_iter is None:
        fit, val = _split_fit_valid(tr)
        w_fit = sample_weights(fit["date"], horizon, ref)
        dtr = _ds(fit, feats, w_fit)
        dva = _ds(val, feats, None)
        params = lgb_params(seed, objective)
        params["metric"] = "None"  # early-stop on mean daily IC (the metric we actually care about)
        feval = make_daily_ic_feval(val["date"].to_numpy(), val[target].to_numpy(float))
        bst = lgb.train(
            params, dtr, num_boost_round=2000, valid_sets=[dva], feval=feval,
            callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)],
        )
        best_iter = max(30, int(bst.best_iteration or 300))
        vpred = bst.predict(val[feats], num_iteration=best_iter)
        valid_q90 = float(np.quantile(vpred, 0.90))
        if prune:
            base_ic = _daily_ic(val, vpred, target)
            rng = np.random.default_rng(seed)
            keep = []
            Xv = val[feats].copy()
            for c in feats:
                orig = Xv[c].to_numpy().copy()
                Xv[c] = rng.permutation(orig)
                ic_p = _daily_ic(val, bst.predict(Xv, num_iteration=best_iter), target)
                Xv[c] = orig
                if base_ic - ic_p > 0:
                    keep.append(c)
            feats = keep if len(keep) >= 8 else feats
    w_all = sample_weights(tr["date"], horizon, ref)
    if objective == "lambdarank":
        w_all = None
    bst_full = lgb.train(lgb_params(seed, objective), _ds(tr, feats, w_all), num_boost_round=best_iter)
    pred = bst_full.predict(te[feats])
    return FitResult(pred=pred, best_iter=best_iter, valid_q90=valid_q90, features=feats, extra={"model": bst_full})


def fit_ridge(tr: pd.DataFrame, te: pd.DataFrame, target: str) -> np.ndarray:
    feats = CS_RANK_COLS + REGIME_COLS
    mu = tr[REGIME_COLS].mean()
    sd = tr[REGIME_COLS].std().replace(0, 1)

    def prep(d: pd.DataFrame) -> np.ndarray:
        x = d[feats].copy()
        x[CS_RANK_COLS] = x[CS_RANK_COLS].fillna(0.5) - 0.5
        x[REGIME_COLS] = ((x[REGIME_COLS] - mu) / sd).fillna(0.0)
        return x.to_numpy(float)

    m = Ridge(alpha=10.0)
    m.fit(prep(tr), tr[target].to_numpy(float) - 0.5)
    return m.predict(prep(te))


def rank_in_date(dates: pd.Series, s: np.ndarray) -> np.ndarray:
    return pd.Series(s, index=dates.index).groupby(dates.values).rank(pct=True).to_numpy()


# --------------------------------------------------------------------------------------
# Walk-forward runner
# --------------------------------------------------------------------------------------
def run_variants(panel: pd.DataFrame, step: int = 5, log: Callable[[str], None] = print) -> tuple[pd.DataFrame, dict]:
    from app.ml.pipeline import _fit_calibrated_3class, _predict_calibrated_ev  # legacy baseline

    cfg = ScreenerConfig()
    base = panel[panel["avg_dollar_vol_m"] >= 25.0].copy()
    folds = make_half_year_folds(base["date"].values)
    all_dates = np.sort(base["date"].unique())
    train_dates = set(all_dates[::step])
    out_frames: list[pd.DataFrame] = []
    meta: dict[str, Any] = {"folds": [], "feature_sets": {}}

    keep_cols = ["date", "symbol", "sector", "is_shortable", "close", "atr_pct", "avg_dollar_vol_m",
                 "mom_12_1", "days_to_earn", "breadth_50", "spy_dist_ema200", "vix", "long_tp", "short_tp"]
    label_cols = [c for c in base.columns if c.startswith(("lr_", "sr_", "target_rel_", "target_pct_", "fwd_ret_"))]

    for fd in folds:
        t0 = time.time()
        te = base[(base["date"] >= fd.test_start) & (base["date"] <= fd.test_end)].copy()
        tr_all = base[(base["date"] <= fd.train_end) & (base["date"].isin(train_dates))]
        res = te[keep_cols + label_cols].copy()
        res["fold_id"] = fd.fid
        fmeta: dict[str, Any] = {"fold": fd.fid, "test_start": str(fd.test_start.date()), "test_end": str(fd.test_end.date()),
                                  "train_end": str(fd.train_end.date()), "n_train": int(len(tr_all)), "n_test": int(len(te))}

        # 0) legacy production baseline: 3-class LGBM + isotonic EV on base features
        tr_l = tr_all.dropna(subset=["c_long", "c_short"])
        m_l, cal_l, e_l = _fit_calibrated_3class(tr_l, "c_long", "long_ret", cfg)
        res["baseline_v2_long"] = _predict_calibrated_ev(m_l, cal_l, e_l, te[BASE_FEATURE_COLS], cfg)[3]
        tr_s = tr_l[tr_l["is_shortable"]]
        m_s, cal_s, e_s = _fit_calibrated_3class(tr_s, "c_short", "short_ret", cfg)
        res["baseline_v2_short"] = _predict_calibrated_ev(m_s, cal_s, e_s, te[BASE_FEATURE_COLS], cfg)[3]

        # 1) 12-1 momentum
        res["mom_12_1_score"] = te["mom_12_1"].to_numpy()

        # 2..) relative-label LightGBM variants
        for h in HORIZONS:
            tgt = f"target_pct_{h}"
            trh = tr_all.dropna(subset=[tgt])
            if h == MAIN_H:
                r = fit_lgb_variant(trh, te, BASE_FEATURE_COLS, tgt, h)
                res["rel_base"] = r.pred
                fmeta["rel_base"] = {"iters": r.best_iter, "q90": r.valid_q90}

                r = fit_lgb_variant(trh, te, V3_FEATURE_COLS, tgt, h)
                res["rel_full"] = r.pred
                fmeta["rel_full"] = {"iters": r.best_iter, "q90": r.valid_q90}

                rp = fit_lgb_variant(trh, te, V3_FEATURE_COLS, tgt, h, prune=True)
                res["rel_pruned"] = rp.pred
                fmeta["rel_pruned"] = {"iters": rp.best_iter, "q90": rp.valid_q90, "n_features": len(rp.features)}
                meta["feature_sets"][fd.fid] = rp.features

                rk = fit_lgb_variant(trh, te, V3_FEATURE_COLS, tgt, h, objective="lambdarank")
                res["rank_full"] = rk.pred
                fmeta["rank_full"] = {"iters": rk.best_iter}

                # ensemble: 5 seeds (pruned features) + ridge + 12-1 momentum, rank-averaged per date
                seeds = [rp.pred]
                for sd in range(1, 5):
                    seeds.append(fit_lgb_variant(trh, te, rp.features, tgt, h, seed=sd, fixed_iter=rp.best_iter).pred)
                d = te["date"]
                lgb_rank = np.mean([rank_in_date(d, p) for p in seeds], axis=0)
                ridge_rank = rank_in_date(d, fit_ridge(trh, te, tgt))
                mom = te["mom_12_1"].to_numpy(float)
                mom_rank = rank_in_date(d, np.where(np.isfinite(mom), mom, np.nanmedian(mom)))
                res["ens"] = 0.6 * lgb_rank + 0.2 * ridge_rank + 0.2 * mom_rank
                res["lgb_seed_avg"] = lgb_rank
            else:
                r = fit_lgb_variant(trh, te, V3_FEATURE_COLS, tgt, h)
                res[f"rel_full_h{h}"] = r.pred
                fmeta[f"rel_full_h{h}"] = {"iters": r.best_iter}

        out_frames.append(res)
        fmeta["secs"] = round(time.time() - t0, 1)
        meta["folds"].append(fmeta)
        log(f"fold {fd.fid}: {fmeta['test_start']}..{fmeta['test_end']} train={fmeta['n_train']} test={fmeta['n_test']} "
            f"iters(rel_full)={fmeta['rel_full']['iters']} pruned_feats={fmeta['rel_pruned']['n_features']} {fmeta['secs']}s")

    return pd.concat(out_frames, ignore_index=True), meta
