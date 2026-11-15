from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.config import ScreenerConfig
from app.features.indicators import FEATURE_COLS


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap_ci(vals: list[float], n_boot: int = 500, seed: int = 42) -> list[float]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if len(arr) < 5:
        m = float(np.mean(arr)) if len(arr) > 0 else 0.0
        return [round(m, 3), round(m, 3)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def _decile_table(df: pd.DataFrame, prob_col: str, y_col: str, r_col: str) -> list[dict[str, Any]]:
    if df.empty or len(df) < 50:
        return []
    work = df[["date", prob_col, y_col, r_col]].dropna().copy()
    # Rank within each date first so deciles measure cross-sectional stock selection, not date effects
    work["pct_rank"] = work.groupby("date")[prob_col].rank(pct=True, method="first")
    work["decile"] = np.ceil(work["pct_rank"] * 10.0).clip(1, 10).astype(int)
    rows = []
    for d in range(1, 11):
        sub = work[work["decile"] == d]
        if sub.empty:
            continue
        rows.append(
            {
                "decile": d,
                "count": int(len(sub)),
                "avg_prob_pct": round(float(sub[prob_col].mean()) * 100.0, 1),
                "tb_hit_rate_pct": round(float(sub[y_col].mean()) * 100.0, 1),
                "avg_r": round(float(sub[r_col].mean()), 3),
                "win_rate_pct": round(float((sub[r_col] > 0).mean()) * 100.0, 1),
            }
        )
    return rows


def compute_mom_12_1(history: dict[str, pd.DataFrame] | None = None, panel: pd.DataFrame | None = None) -> dict[tuple[str, pd.Timestamp], float]:
    """Compute classic 12-1 momentum (t-21 / t-252 - 1) lookup for baseline comparison."""
    out: dict[tuple[str, pd.Timestamp], float] = {}
    if history:
        for sym, df in history.items():
            if len(df) < 255:
                continue
            c = df["close"].astype(float)
            mom = c.shift(21) / c.shift(252) - 1.0
            for dt, val in mom.dropna().items():
                out[(sym, pd.Timestamp(dt))] = float(val)
    elif panel is not None and not panel.empty:
        for sym, grp in panel.sort_values("date").groupby("symbol", sort=False):
            if len(grp) < 255:
                continue
            c = grp["close"].astype(float)
            mom = c.shift(21) / c.shift(252) - 1.0
            for dt, val in zip(grp["date"], mom):
                if np.isfinite(val):
                    out[(str(sym), pd.Timestamp(dt))] = float(val)
    return out


def train_logreg_fold(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Train L2 Logistic Regression baseline on standardized features."""
    x_tr = tr[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
    x_te = te[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)

    lr_l = LogisticRegression(C=0.2, max_iter=300, random_state=42)
    lr_l.fit(x_tr_s, tr["y_long"].astype(int))
    p_l = lr_l.predict_proba(x_te_s)[:, 1]

    lr_s = LogisticRegression(C=0.2, max_iter=300, random_state=42)
    lr_s.fit(x_tr_s, tr["y_short"].astype(int))
    p_s = lr_s.predict_proba(x_te_s)[:, 1]
    return p_l, p_s


def compute_screener_evaluation(
    oos: pd.DataFrame,
    history: dict[str, pd.DataFrame] | None,
    cfg: ScreenerConfig,
    n_bootstrap: int = 500,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Compute comprehensive screener evaluation metrics:
      - Lift over unconditional base rate (Long & Short)
      - Per-date Rank IC (Spearman) with mean, std, t-stat, and bootstrap 95% CI
      - Probability Decile Lift (D1..D10)
      - Per-fold stability across walk-forward splits
      - Baselines: Logistic Regression, 12-1 Momentum, and Random Pick
      - Volatility-bucket lift breakdown
    """
    if oos.empty:
        return {}

    work = oos.copy()
    work["target_gain_pct"] = (cfg.target_atr_mult * work["atr_14"] / work["close"]) * 100.0
    work = work[work["target_gain_pct"] >= cfg.min_target_gain_pct].copy()
    if work.empty:
        return {}

    risk_pct = (cfg.stop_atr_mult * work["atr_14"] / work["close"]).replace(0.0, np.nan)
    work["long_r"] = (work["long_ret"] / risk_pct).clip(-3.0, 4.0)
    work["short_r"] = (work["short_ret"] / risk_pct).clip(-3.0, 4.0)

    mom_map = compute_mom_12_1(history=history, panel=panel)
    work["mom_12_1"] = [mom_map.get((str(s), pd.Timestamp(d)), np.nan) for s, d in zip(work["symbol"], work["date"])]

    unique_dates = np.sort(work["date"].unique())
    # Use non-overlapping horizon_days steps so Rank IC t-stats and bootstrap CIs are independent
    rebal_dates = unique_dates[:: max(1, cfg.horizon_days)]

    long_ics: list[float] = []
    short_ics: list[float] = []
    long_lifts: list[float] = []
    short_lifts: list[float] = []

    lgbm_l_hits, lgbm_l_rs = [], []
    lgbm_s_hits, lgbm_s_rs = [], []
    base_l_hits, base_l_rs = [], []
    base_s_hits, base_s_rs = [], []

    lr_l_hits, lr_l_rs, lr_l_ics = [], [], []
    lr_s_hits, lr_s_rs, lr_s_ics = [], [], []

    mom_l_hits, mom_l_rs, mom_l_ics = [], [], []
    mom_s_hits, mom_s_rs, mom_s_ics = [], [], []

    fold_records: dict[int, dict[str, list[float]]] = {}

    for dt in rebal_dates:
        day = work[work["date"] == dt]
        if len(day) < 20:
            continue
        fold_id = int(day["fold_id"].iloc[0]) if "fold_id" in day.columns else 0
        f_rec = fold_records.setdefault(
            fold_id,
            {"l_top": [], "l_base": [], "s_top": [], "s_base": [], "l_ic": [], "s_ic": []},
        )

        # Unconditional base rates on eligible pool
        b_l_hit = float(day["y_long"].mean())
        b_l_r = float(day["long_r"].mean())
        base_l_hits.extend(day["y_long"].tolist())
        base_l_rs.extend(day["long_r"].tolist())

        # LightGBM Long Top-K
        top_l = day[day["prob_long"] >= cfg.min_win_prob].nlargest(cfg.top_k, "prob_long")
        if not top_l.empty:
            lgbm_l_hits.extend(top_l["y_long"].tolist())
            lgbm_l_rs.extend(top_l["long_r"].tolist())
            lift_l = (float(top_l["y_long"].mean()) - b_l_hit) * 100.0
            long_lifts.append(lift_l)
            f_rec["l_top"].extend(top_l["y_long"].tolist())
            f_rec["l_base"].extend(day["y_long"].tolist())

        ic_l = _spearman_corr(day["prob_long"].to_numpy(dtype=float), day["long_r"].to_numpy(dtype=float))
        if np.isfinite(ic_l):
            long_ics.append(ic_l)
            f_rec["l_ic"].append(ic_l)

        # Shortable pool
        s_pool = day[day["is_shortable"]]
        if len(s_pool) >= 10:
            b_s_hit = float(s_pool["y_short"].mean())
            base_s_hits.extend(s_pool["y_short"].tolist())
            base_s_rs.extend(s_pool["short_r"].tolist())
            top_s = s_pool[s_pool["prob_short"] >= cfg.min_win_prob].nlargest(cfg.top_k, "prob_short")
            if not top_s.empty:
                lgbm_s_hits.extend(top_s["y_short"].tolist())
                lgbm_s_rs.extend(top_s["short_r"].tolist())
                lift_s = (float(top_s["y_short"].mean()) - b_s_hit) * 100.0
                short_lifts.append(lift_s)
                f_rec["s_top"].extend(top_s["y_short"].tolist())
                f_rec["s_base"].extend(s_pool["y_short"].tolist())

            ic_s = _spearman_corr(s_pool["prob_short"].to_numpy(dtype=float), s_pool["short_r"].to_numpy(dtype=float))
            if np.isfinite(ic_s):
                short_ics.append(ic_s)
                f_rec["s_ic"].append(ic_s)

        # Logistic Regression baseline
        if "lr_prob_long" in day.columns:
            lr_tl = day.nlargest(cfg.top_k, "lr_prob_long")
            lr_l_hits.extend(lr_tl["y_long"].tolist())
            lr_l_rs.extend(lr_tl["long_r"].tolist())
            ic_lr_l = _spearman_corr(day["lr_prob_long"].to_numpy(dtype=float), day["long_r"].to_numpy(dtype=float))
            if np.isfinite(ic_lr_l):
                lr_l_ics.append(ic_lr_l)

            if len(s_pool) >= 10:
                lr_ts = s_pool.nlargest(cfg.top_k, "lr_prob_short")
                lr_s_hits.extend(lr_ts["y_short"].tolist())
                lr_s_rs.extend(lr_ts["short_r"].tolist())
                ic_lr_s = _spearman_corr(s_pool["lr_prob_short"].to_numpy(dtype=float), s_pool["short_r"].to_numpy(dtype=float))
                if np.isfinite(ic_lr_s):
                    lr_s_ics.append(ic_lr_s)

        # 12-1 Momentum baseline
        m_day = day.dropna(subset=["mom_12_1"])
        if len(m_day) >= cfg.top_k:
            m_tl = m_day.nlargest(cfg.top_k, "mom_12_1")
            mom_l_hits.extend(m_tl["y_long"].tolist())
            mom_l_rs.extend(m_tl["long_r"].tolist())
            ic_ml = _spearman_corr(m_day["mom_12_1"].to_numpy(dtype=float), m_day["long_r"].to_numpy(dtype=float))
            if np.isfinite(ic_ml):
                mom_l_ics.append(ic_ml)

            m_sp = m_day[m_day["is_shortable"]]
            if len(m_sp) >= cfg.top_k:
                m_ts = m_sp.nsmallest(cfg.top_k, "mom_12_1")
                mom_s_hits.extend(m_ts["y_short"].tolist())
                mom_s_rs.extend(m_ts["short_r"].tolist())
                ic_ms = _spearman_corr((-m_sp["mom_12_1"]).to_numpy(dtype=float), m_sp["short_r"].to_numpy(dtype=float))
                if np.isfinite(ic_ms):
                    mom_s_ics.append(ic_ms)

    def _ic_stats(ics: list[float]) -> dict[str, Any]:
        if not ics:
            return {"mean": 0.0, "std": 0.0, "t_stat": 0.0, "pos_pct": 0.0, "ci_95": [0.0, 0.0]}
        arr = np.array(ics, dtype=float)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        t = (m / (s / np.sqrt(len(arr)))) if s > 0 else 0.0
        return {
            "mean": round(m, 4),
            "std": round(s, 4),
            "t_stat": round(float(t), 2),
            "pos_pct": round(float((arr > 0).mean() * 100.0), 1),
            "ci_95": _bootstrap_ci(ics, n_boot=n_bootstrap),
        }

    base_l_pct = round(float(np.mean(base_l_hits) * 100.0), 1) if base_l_hits else 0.0
    top_l_pct = round(float(np.mean(lgbm_l_hits) * 100.0), 1) if lgbm_l_hits else 0.0
    base_s_pct = round(float(np.mean(base_s_hits) * 100.0), 1) if base_s_hits else 0.0
    top_s_pct = round(float(np.mean(lgbm_s_hits) * 100.0), 1) if lgbm_s_hits else 0.0

    # Per-fold stability
    per_fold = []
    l_beats = 0
    s_beats = 0
    for fid in sorted(fold_records.keys()):
        fr = fold_records[fid]
        lt = round(float(np.mean(fr["l_top"]) * 100.0), 1) if fr["l_top"] else 0.0
        lb = round(float(np.mean(fr["l_base"]) * 100.0), 1) if fr["l_base"] else 0.0
        st = round(float(np.mean(fr["s_top"]) * 100.0), 1) if fr["s_top"] else 0.0
        sb = round(float(np.mean(fr["s_base"]) * 100.0), 1) if fr["s_base"] else 0.0
        if lt > lb:
            l_beats += 1
        if st > sb:
            s_beats += 1
        per_fold.append(
            {
                "fold": fid + 1,
                "long_top_k_pct": lt,
                "long_base_pct": lb,
                "long_lift_pp": round(lt - lb, 1),
                "long_ic": round(float(np.mean(fr["l_ic"])), 4) if fr["l_ic"] else 0.0,
                "short_top_k_pct": st,
                "short_base_pct": sb,
                "short_lift_pp": round(st - sb, 1),
                "short_ic": round(float(np.mean(fr["s_ic"])), 4) if fr["s_ic"] else 0.0,
            }
        )

    # Within-date volatility bucket check (ATR% tertiles)
    vol_buckets = []
    try:
        work["vol_pct_rank"] = work.groupby("date")["atr_pct"].rank(pct=True, method="first")
        work["vol_bucket"] = np.where(
            work["vol_pct_rank"] <= 1.0 / 3.0,
            "Low Vol",
            np.where(work["vol_pct_rank"] <= 2.0 / 3.0, "Mid Vol", "High Vol"),
        )
        work["prob_pct_rank_date"] = work.groupby(["date", "vol_bucket"])["prob_long"].rank(pct=True, method="first")
        for vb in ["Low Vol", "Mid Vol", "High Vol"]:
            sub = work[work["vol_bucket"] == vb]
            if sub.empty:
                continue
            top_sub = sub[sub["prob_pct_rank_date"] >= 0.9]
            vol_buckets.append(
                {
                    "bucket": vb,
                    "base_long_pct": round(float(sub["y_long"].mean() * 100.0), 1),
                    "top10_long_pct": round(float(top_sub["y_long"].mean() * 100.0), 1) if not top_sub.empty else 0.0,
                    "lift_pp": round(float((top_sub["y_long"].mean() - sub["y_long"].mean()) * 100.0), 1) if not top_sub.empty else 0.0,
                }
            )
    except Exception:
        pass

    return {
        "rebalance_dates": int(len(rebal_dates)),
        "long_summary": {
            "top_k_hit_pct": top_l_pct,
            "base_rate_pct": base_l_pct,
            "lift_pp": round(top_l_pct - base_l_pct, 1),
            "lift_ratio": round(top_l_pct / max(0.1, base_l_pct), 2),
            "lift_ci_95": _bootstrap_ci(long_lifts, n_boot=n_bootstrap),
            "avg_r": round(float(np.mean(lgbm_l_rs)), 3) if lgbm_l_rs else 0.0,
            "base_avg_r": round(float(np.mean(base_l_rs)), 3) if base_l_rs else 0.0,
            "rank_ic": _ic_stats(long_ics),
            "folds_beating_base": f"{l_beats}/{len(per_fold)}",
        },
        "short_summary": {
            "top_k_hit_pct": top_s_pct,
            "base_rate_pct": base_s_pct,
            "lift_pp": round(top_s_pct - base_s_pct, 1),
            "lift_ratio": round(top_s_pct / max(0.1, base_s_pct), 2),
            "lift_ci_95": _bootstrap_ci(short_lifts, n_boot=n_bootstrap),
            "avg_r": round(float(np.mean(lgbm_s_rs)), 3) if lgbm_s_rs else 0.0,
            "base_avg_r": round(float(np.mean(base_s_rs)), 3) if base_s_rs else 0.0,
            "rank_ic": _ic_stats(short_ics),
            "folds_beating_base": f"{s_beats}/{len(per_fold)}",
        },
        "baselines": {
            "lgbm": {
                "long_hit_pct": top_l_pct,
                "long_avg_r": round(float(np.mean(lgbm_l_rs)), 3) if lgbm_l_rs else 0.0,
                "long_ic": _ic_stats(long_ics)["mean"],
                "short_hit_pct": top_s_pct,
                "short_avg_r": round(float(np.mean(lgbm_s_rs)), 3) if lgbm_s_rs else 0.0,
                "short_ic": _ic_stats(short_ics)["mean"],
            },
            "logistic_regression": {
                "long_hit_pct": round(float(np.mean(lr_l_hits) * 100.0), 1) if lr_l_hits else 0.0,
                "long_avg_r": round(float(np.mean(lr_l_rs)), 3) if lr_l_rs else 0.0,
                "long_ic": _ic_stats(lr_l_ics)["mean"],
                "short_hit_pct": round(float(np.mean(lr_s_hits) * 100.0), 1) if lr_s_hits else 0.0,
                "short_avg_r": round(float(np.mean(lr_s_rs)), 3) if lr_s_rs else 0.0,
                "short_ic": _ic_stats(lr_s_ics)["mean"],
            },
            "momentum_12_1": {
                "long_hit_pct": round(float(np.mean(mom_l_hits) * 100.0), 1) if mom_l_hits else 0.0,
                "long_avg_r": round(float(np.mean(mom_l_rs)), 3) if mom_l_rs else 0.0,
                "long_ic": _ic_stats(mom_l_ics)["mean"],
                "short_hit_pct": round(float(np.mean(mom_s_hits) * 100.0), 1) if mom_s_hits else 0.0,
                "short_avg_r": round(float(np.mean(mom_s_rs)), 3) if mom_s_rs else 0.0,
                "short_ic": _ic_stats(mom_s_ics)["mean"],
            },
            "random_pool": {
                "long_hit_pct": base_l_pct,
                "long_avg_r": round(float(np.mean(base_l_rs)), 3) if base_l_rs else 0.0,
                "long_ic": 0.0,
                "short_hit_pct": base_s_pct,
                "short_avg_r": round(float(np.mean(base_s_rs)), 3) if base_s_rs else 0.0,
                "short_ic": 0.0,
            },
        },
        "deciles_long": _decile_table(work, "prob_long", "y_long", "long_r"),
        "deciles_short": _decile_table(work[work["is_shortable"]], "prob_short", "y_short", "short_r"),
        "per_fold": per_fold,
        "vol_buckets": vol_buckets,
    }
