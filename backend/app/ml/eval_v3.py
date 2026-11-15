"""Evaluation for the V3 A/B experiments: pre-registered pass criteria, daily rank IC with Newey-West,
decile spreads with block bootstrap, paired comparison vs 12-1 momentum, regime splits, and a
selection-rule portfolio backtest (regime gate, earnings blackout, sector cap, correlation filter,
liquidity floor, EV floor) against SPY, momentum and a random pool.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from app.ml.research_v3 import BARRIER_GRID, MAIN_H, MAIN_STOP, MAIN_TGT, r_col, sr_col

LR = r_col(MAIN_H, MAIN_STOP, MAIN_TGT)
SR = sr_col(MAIN_H, MAIN_STOP, MAIN_TGT)
COST_ROUND_TRIP = 0.002  # 10 bps per side

# Pass criteria fixed BEFORE looking at results (from the feedback document)
PASS_CRITERIA = {
    "daily_ic_nw_t_min": 3.0,
    "spread_ci_low_min": 0.0,
    "paired_vs_mom_ci_low_min": 0.0,
    "fold_ic_positive_frac_min": 0.5,
}

VARIANTS = {
    "baseline_v2_long": "Current model (3-class EV, abs. label)",
    "mom_12_1_score": "12-1 momentum",
    "rel_base": "Relative label, base features",
    "rel_full": "Relative label + V3 features",
    "rel_pruned": "Relative + V3, permutation-pruned",
    "rank_full": "LambdaRank + V3 features",
    "lgb_seed_avg": "5-seed LGBM avg (pruned)",
    "ens": "Ensemble (5 seeds + ridge + momentum)",
    "rel_xs_icstop": "Relative fwd-ret + XS V3 (IC early-stop)",
    "tb_rel_xs": "Relative TB-R + XS V3 (IC early-stop)",
    "tb_ens": "Ensemble on Relative TB-R (5 seeds + ridge + mom)",
    "rel_full_h5": "Relative + V3, 5-day horizon",
    "rel_full_h10": "Relative + V3, 10-day horizon",
    "rel_full_h20": "Relative + V3, 20-day horizon",
}


# ---------------------------------------------------------------------------- statistics
def newey_west_t(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 5:
        return 0.0
    m = x.mean()
    e = x - m
    s = np.dot(e, e) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * np.dot(e[k:], e[:-k]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(m / se)


def block_bootstrap_ci(x: np.ndarray, block: int, n_boot: int = 2000, seed: int = 7) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _spearman_by_date(df: pd.DataFrame, score: str, target: str) -> pd.Series:
    d = df[["date", score, target]].dropna()
    d = d.assign(rs=d.groupby("date")[score].rank(), ry=d.groupby("date")[target].rank())
    g = d.groupby("date")
    cnt = g.size()
    ic = g.apply(lambda x: np.corrcoef(x["rs"], x["ry"])[0, 1] if len(x) > 20 else np.nan, include_groups=False)
    return ic[cnt > 20].dropna()


# ---------------------------------------------------------------------------- per-variant metrics
def variant_metrics(oos: pd.DataFrame, score: str, top_k: int = 10, h: int = MAIN_H) -> dict[str, Any]:
    lr = LR if h == MAIN_H else r_col(h, MAIN_STOP, MAIN_TGT)
    sr = SR if h == MAIN_H else sr_col(h, MAIN_STOP, MAIN_TGT)
    df = oos.dropna(subset=[score, lr])
    ic = _spearman_by_date(df, score, lr)
    ic_rel = _spearman_by_date(df, score, f"target_rel_{h}")
    fold_ic = df.assign(_d=df["date"]).groupby("fold_id").apply(
        lambda g: _spearman_by_date(g, score, lr).mean(), include_groups=False
    )

    # deciles per date
    df = df.assign(dec=df.groupby("date")[score].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)) + 1)
    dec_tbl = df.groupby("dec").agg(long_r=(lr, "mean"), short_r=(sr, "mean"), hit=("long_tp", "mean"), rel=(f"target_rel_{h}", "mean"))
    by_date_dec = df.groupby(["date", "dec"])[lr].mean().unstack()
    spread = (by_date_dec[10] - by_date_dec[1]).dropna()

    # top-k on non-overlapping rebalance dates
    dates = np.sort(df["date"].unique())[::h]
    reb = df[df["date"].isin(dates)]
    top = reb.sort_values(score, ascending=False).groupby("date").head(top_k)
    bot = reb[reb["is_shortable"]].sort_values(score, ascending=True).groupby("date").head(top_k)
    pool_r = reb.groupby("date")[lr].mean()
    pool_sr = reb[reb["is_shortable"]].groupby("date")[sr].mean()
    top_r = top.groupby("date")[lr].mean()
    bot_r = bot.groupby("date")[sr].mean()

    regimes = {}
    spy_up = df.groupby("date")["spy_dist_ema200"].first() > 0
    regimes["spy_above_200"] = round(float(ic[ic.index.isin(spy_up[spy_up].index)].mean()), 4)
    regimes["spy_below_200"] = round(float(ic[ic.index.isin(spy_up[~spy_up].index)].mean()), 4)
    vix = df.groupby("date")["vix"].first()
    terc = pd.qcut(vix, 3, labels=["low", "mid", "high"])
    for t in ["low", "mid", "high"]:
        regimes[f"vix_{t}"] = round(float(ic[ic.index.isin(terc[terc == t].index)].mean()), 4)

    return {
        "label": VARIANTS.get(score, score),
        "daily_ic_mean": round(float(ic.mean()), 4),
        "daily_ic_nw_t": round(newey_west_t(ic.to_numpy(), lag=h), 2),
        "daily_ic_pos_pct": round(float((ic > 0).mean() * 100), 1),
        "ic_vs_relative_target": round(float(ic_rel.mean()), 4),
        "ic_vs_relative_nw_t": round(newey_west_t(ic_rel.to_numpy(), lag=h), 2),
        "fold_ic": [round(float(v), 4) for v in fold_ic.to_numpy()],
        "fold_ic_positive": f"{int((fold_ic > 0).sum())}/{len(fold_ic)}",
        "fold_ic_positive_frac": round(float((fold_ic > 0).mean()), 3),
        "spread_top_minus_bottom_r": round(float(spread.mean()), 4),
        "spread_ci95": block_bootstrap_ci(spread.to_numpy(), block=h),
        "deciles": [
            {"decile": int(k), "long_r": round(float(v.long_r), 4), "short_r": round(float(v.short_r), 4),
             "tp_hit_pct": round(float(v.hit) * 100, 1), "rel_target": round(float(v.rel), 4)}
            for k, v in dec_tbl.iterrows()
        ],
        "top_k_long_r": round(float(top_r.mean()), 4),
        "pool_long_r": round(float(pool_r.mean()), 4),
        "top_k_long_lift_r": round(float((top_r - pool_r.reindex(top_r.index)).mean()), 4),
        "top_k_long_lift_ci95": block_bootstrap_ci((top_r - pool_r.reindex(top_r.index)).to_numpy(), block=2),
        "bottom_k_short_r": round(float(bot_r.mean()), 4),
        "pool_short_r": round(float(pool_sr.mean()), 4),
        "bottom_k_short_lift_r": round(float((bot_r - pool_sr.reindex(bot_r.index)).mean()), 4),
        "bottom_k_short_lift_ci95": block_bootstrap_ci((bot_r - pool_sr.reindex(bot_r.index)).to_numpy(), block=2),
        "regime_ic": regimes,
        "_top_r_series": top_r,
    }


def barrier_sweep(oos: pd.DataFrame, score: str, top_k: int = 10) -> list[dict[str, Any]]:
    dates = np.sort(oos["date"].unique())[::MAIN_H]
    reb = oos[oos["date"].isin(dates)].dropna(subset=[score])
    top = reb.sort_values(score, ascending=False).groupby("date").head(top_k)
    out = []
    for s, t in BARRIER_GRID:
        c = r_col(MAIN_H, s, t)
        tr = top.groupby("date")[c].mean()
        pr = reb.groupby("date")[c].mean()
        # convert R to % return so different stop sizes are comparable
        pct_top = (top[c] * s * top["atr_pct"]).groupby(top["date"]).mean()
        pct_pool = (reb[c] * s * reb["atr_pct"]).groupby(reb["date"]).mean()
        out.append({"stop_atr": s, "target_atr": t, "top_k_r": round(float(tr.mean()), 4),
                    "pool_r": round(float(pr.mean()), 4), "top_k_ret_pct": round(float(pct_top.mean() * 100), 3),
                    "pool_ret_pct": round(float(pct_pool.mean() * 100), 3)})
    return out


# ---------------------------------------------------------------------------- selection-rule backtest
def _corr_filter_pick(cands: pd.DataFrame, rets: pd.DataFrame, dt: pd.Timestamp, n: int, max_per_sector: int = 2,
                      max_corr: float = 0.8) -> pd.DataFrame:
    window = rets.loc[:dt].tail(60)
    chosen: list[int] = []
    chosen_syms: list[str] = []
    sec_count: dict[str, int] = {}
    for i, row in cands.iterrows():
        if len(chosen) >= n:
            break
        if sec_count.get(row["sector"], 0) >= max_per_sector:
            continue
        sym = row["symbol"]
        if chosen_syms and sym in window.columns:
            cols = [s for s in chosen_syms if s in window.columns]
            if cols:
                c = window[cols].corrwith(window[sym]).abs().max()
                if np.isfinite(c) and c > max_corr:
                    continue
        chosen.append(i)
        chosen_syms.append(sym)
        sec_count[row["sector"]] = sec_count.get(row["sector"], 0) + 1
    return cands.loc[chosen]


def selection_backtest(
    oos: pd.DataFrame,
    score: str,
    regime_df: pd.DataFrame,
    rets: pd.DataFrame,
    floors: dict[int, float] | None,
    top_k: int = 10,
    use_rules: bool = True,
    enable_shorts: bool = True,
    seed: int = 11,
) -> dict[str, Any]:
    df = oos.dropna(subset=[LR]).copy()
    dates = np.sort(df["date"].unique())[::MAIN_H]
    spy = regime_df["spy_close"].astype(float)
    rng = np.random.default_rng(seed)
    rows = []
    for dt in dates:
        day = df[df["date"] == dt]
        if use_rules:
            pool = day[(day["avg_dollar_vol_m"] >= 100.0) & (day["close"] >= 10.0)]
            pool = pool[~((pool["days_to_earn"] >= 0) & (pool["days_to_earn"] <= 15))]
        else:
            pool = day
        breadth = float(day["breadth_50"].iloc[0])
        spy_below = float(day["spy_dist_ema200"].iloc[0]) < 0
        n_long = top_k if (not use_rules or breadth >= 0.30) else 3
        fid = int(day["fold_id"].iloc[0])
        floor = floors.get(fid, -np.inf) if (floors and use_rules) else -np.inf

        if score == "random":
            longs = pool.sample(n=min(n_long, len(pool)), random_state=int(rng.integers(1e9)))
            shorts = pool.iloc[0:0]
        else:
            lc = pool[pool[score] >= floor].sort_values(score, ascending=False).head(60)
            longs = _corr_filter_pick(lc, rets, dt, n_long) if use_rules else lc.head(n_long)
            shorts = pool.iloc[0:0]
            if enable_shorts and use_rules and spy_below and breadth < 0.40:
                sc = pool[pool["is_shortable"]].sort_values(score, ascending=True).head(60)
                shorts = _corr_filter_pick(sc, rets, dt, top_k)

        l_ret = (longs[LR] * MAIN_STOP * longs["atr_pct"] - COST_ROUND_TRIP).to_numpy()
        s_ret = (shorts[SR] * MAIN_STOP * shorts["atr_pct"] - COST_ROUND_TRIP).to_numpy()
        port = (l_ret.sum() + s_ret.sum()) / top_k  # each slot = 1/top_k of capital; unused slots in cash
        i0 = spy.index.searchsorted(dt)
        i1 = min(i0 + MAIN_H, len(spy) - 1)
        spy_r = float(spy.iloc[i1] / spy.iloc[i0] - 1.0)
        pool_eq = float((day[LR] * MAIN_STOP * day["atr_pct"]).mean())
        rows.append({
            "date": dt, "ret": port, "spy": spy_r, "pool": pool_eq, "n_long": len(longs), "n_short": len(shorts),
            "long_r": float(longs[LR].mean()) if len(longs) else np.nan,
            "short_r": float(shorts[SR].mean()) if len(shorts) else np.nan,
            "long_hits": float((longs[LR] > 0).mean()) if len(longs) else np.nan,
            "long_r_list": longs[LR].tolist(), "short_r_list": shorts[SR].tolist(),
        })
    bt = pd.DataFrame(rows)
    return summarize_bt(bt)


def summarize_bt(bt: pd.DataFrame) -> dict[str, Any]:
    per_year = 252.0 / MAIN_H

    def stats(r: pd.Series) -> dict[str, float]:
        eq = (1 + r).cumprod()
        yrs = len(r) / per_year
        cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0.0
        sharpe = r.mean() / r.std() * np.sqrt(per_year) if r.std() > 0 else 0.0
        dd = (eq / eq.cummax() - 1).min()
        return {"total_return_pct": round(float((eq.iloc[-1] - 1) * 100), 1), "cagr_pct": round(float(cagr * 100), 2),
                "sharpe": round(float(sharpe), 2), "max_dd_pct": round(float(dd * 100), 1)}

    all_l = [x for xs in bt["long_r_list"] for x in xs]
    all_s = [x for xs in bt["short_r_list"] for x in xs]
    excess = bt["ret"] - bt["spy"] * (bt["n_long"] / 10.0)
    return {
        "strategy": stats(bt["ret"]),
        "spy_buy_hold": stats(bt["spy"]),
        "equal_weight_pool": stats(bt["pool"]),
        "periods": int(len(bt)),
        "zero_pick_periods": int((bt["n_long"] + bt["n_short"] == 0).sum()),
        "short_active_periods": int((bt["n_short"] > 0).sum()),
        "long_picks": len(all_l),
        "short_picks": len(all_s),
        "avg_long_r": round(float(np.mean(all_l)), 4) if all_l else 0.0,
        "avg_short_r": round(float(np.mean(all_s)), 4) if all_s else 0.0,
        "long_win_pct": round(float(np.mean(np.array(all_l) > 0) * 100), 1) if all_l else 0.0,
        "avg_gross_exposure": round(float(((bt["n_long"] + bt["n_short"]) / 10.0).mean()), 2),
        "beta_adj_excess_per_period_pct": round(float(excess.mean() * 100), 3),
        "beta_adj_excess_ci95": [round(v * 100, 3) for v in block_bootstrap_ci(excess.to_numpy(), block=2)],
        "equity_curve": [
            {"date": str(pd.Timestamp(d).date()), "strategy": round(float(s), 4), "spy": round(float(p), 4), "pool": round(float(q), 4)}
            for d, s, p, q in zip(bt["date"], (1 + bt["ret"]).cumprod(), (1 + bt["spy"]).cumprod(), (1 + bt["pool"]).cumprod())
        ],
    }


# ---------------------------------------------------------------------------- orchestration
def run_full_evaluation(oos: pd.DataFrame, meta: dict, regime_df: pd.DataFrame, panel_ret: pd.DataFrame,
                        log: Callable[[str], None] = print) -> dict[str, Any]:
    oos = oos.copy()
    oos["date"] = pd.to_datetime(oos["date"])
    rets = panel_ret.pivot(index="date", columns="symbol", values="ret_1d").sort_index()
    rets.index = pd.to_datetime(rets.index)

    results: dict[str, Any] = {"pass_criteria": PASS_CRITERIA, "variants": {}, "n_experiments": 0}
    top_series = {}
    for v in VARIANTS:
        if v not in oos.columns:
            continue
        h = int(v.split("_h")[-1]) if "_h" in v and v.startswith("rel_full_h") else MAIN_H
        m = variant_metrics(oos, v, h=h)
        top_series[v] = m.pop("_top_r_series")
        results["variants"][v] = m
        log(f"{v:20s} IC={m['daily_ic_mean']:+.4f} NWt={m['daily_ic_nw_t']:+.2f} spread={m['spread_top_minus_bottom_r']:+.3f} "
            f"{m['spread_ci95']} topK lift={m['top_k_long_lift_r']:+.3f} folds+={m['fold_ic_positive']}")

    # short side of the legacy baseline uses its own short model
    if "baseline_v2_short" in oos.columns:
        bs = oos.dropna(subset=["baseline_v2_short", SR])
        dates = np.sort(bs["date"].unique())[::MAIN_H]
        reb = bs[bs["date"].isin(dates) & bs["is_shortable"]]
        top_s = reb.sort_values("baseline_v2_short", ascending=False).groupby("date").head(10)
        diff = top_s.groupby("date")[SR].mean() - reb.groupby("date")[SR].mean()
        results["variants"]["baseline_v2_long"]["own_short_model_top_k_short_r"] = round(float(top_s[SR].mean()), 4)
        results["variants"]["baseline_v2_long"]["own_short_model_lift_r"] = round(float(diff.mean()), 4)
        results["variants"]["baseline_v2_long"]["own_short_model_lift_ci95"] = block_bootstrap_ci(diff.to_numpy(), block=2)

    # paired vs momentum on the top-k long R per rebalance date
    mom = top_series.get("mom_12_1_score")
    for v, s in top_series.items():
        if mom is None or v == "mom_12_1_score":
            continue
        j = pd.concat([s, mom], axis=1, join="inner").dropna()
        d = (j.iloc[:, 0] - j.iloc[:, 1]).to_numpy()
        results["variants"][v]["paired_vs_mom_r"] = round(float(d.mean()), 4)
        results["variants"][v]["paired_vs_mom_ci95"] = block_bootstrap_ci(d, block=2)

    # pass/fail
    for v, m in results["variants"].items():
        checks = {
            "ic_nw_t>3": m["daily_ic_nw_t"] > PASS_CRITERIA["daily_ic_nw_t_min"],
            "spread_ci>0": m["spread_ci95"][0] > PASS_CRITERIA["spread_ci_low_min"],
            "beats_mom_ci>0": (m.get("paired_vs_mom_ci95") or [0, 0])[0] > PASS_CRITERIA["paired_vs_mom_ci_low_min"],
            "most_folds_ic>0": m["fold_ic_positive_frac"] > PASS_CRITERIA["fold_ic_positive_frac_min"],
        }
        m["checks"] = checks
        m["passes_all"] = all(checks.values())
    results["n_experiments"] = len(results["variants"]) + len(BARRIER_GRID)

    # choose best ML variant by primary metric (NW t of daily IC) among 15-day variants
    cand = [v for v in results["variants"] if v not in ("mom_12_1_score",) and not v.startswith("rel_full_h")]
    best = max(cand, key=lambda v: results["variants"][v]["daily_ic_nw_t"])
    results["best_variant"] = best
    log(f"best variant by NW t: {best}")

    results["barrier_sweep"] = {best: barrier_sweep(oos, best), "mom_12_1_score": barrier_sweep(oos, "mom_12_1_score")}

    # selection-rule backtests (EV floor = 90th pct of the variant's validation predictions per fold)
    floors = {int(f["fold"]): float(f[best]["q90"]) for f in meta["folds"] if best in f and "q90" in f[best]}
    sel = {}
    sel[f"{best}_rules"] = selection_backtest(oos, best, regime_df, rets, floors)
    sel[f"{best}_rules_no_floor"] = selection_backtest(oos, best, regime_df, rets, None)
    sel[f"{best}_plain_top10"] = selection_backtest(oos, best, regime_df, rets, None, use_rules=False)
    sel["baseline_v2_plain_top10"] = selection_backtest(oos, "baseline_v2_long", regime_df, rets, None, use_rules=False)
    sel["ens_plain_top10"] = selection_backtest(oos, "ens", regime_df, rets, None, use_rules=False)
    sel["mom_12_1_plain_top10"] = selection_backtest(oos, "mom_12_1_score", regime_df, rets, None, use_rules=False)
    sel["random_plain_top10"] = selection_backtest(oos, "random", regime_df, rets, None, use_rules=False)
    sel["mom_12_1_rules"] = selection_backtest(oos, "mom_12_1_score", regime_df, rets, None, enable_shorts=False)
    sel["random_rules"] = selection_backtest(oos, "random", regime_df, rets, None)
    if "ens" in oos.columns and best != "ens":
        sel["ens_rules_no_floor"] = selection_backtest(oos, "ens", regime_df, rets, None)
    results["selection_backtests"] = sel
    for k, v in sel.items():
        s = v["strategy"]
        log(f"{k:28s} total={s['total_return_pct']:+.1f}% CAGR={s['cagr_pct']:+.2f}% Sharpe={s['sharpe']:.2f} "
            f"DD={s['max_dd_pct']:.1f}% longR={v['avg_long_r']:+.3f} shortR={v['avg_short_r']:+.3f} SPY={v['spy_buy_hold']['total_return_pct']:+.1f}%")
    results["folds"] = meta["folds"]
    return results
