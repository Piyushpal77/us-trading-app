from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from app.config import MODELS_DIR, ScreenerConfig
from app.data.universe import UniverseMember
from app.features.indicators import FEATURE_COLS, FEATURE_LABELS, compute_features
from app.ml.evaluation import compute_screener_evaluation, train_logreg_fold
from app.ml.labeling import compute_triple_barrier_labels


US_EASTERN = ZoneInfo("America/New_York")


def _grade_from_score(score: float) -> str:
    if score >= 115:
        return "A+"
    if score >= 95:
        return "A"
    if score >= 78:
        return "B+"
    if score >= 60:
        return "B"
    return "C"


def _make_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        subsample_for_bin=50000,
        min_child_samples=60,
        colsample_bytree=0.75,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )


@dataclass
class TrainedArtifacts:
    model_long: lgb.LGBMClassifier
    model_short: lgb.LGBMClassifier
    config: ScreenerConfig
    trained_at: str
    data_mode: str
    backtest: dict[str, Any]
    feature_importance: list[dict[str, Any]]


def build_panel_dataset(
    universe: list[UniverseMember],
    history: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    cfg: ScreenerConfig,
    include_labels: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    meta_map = {m.symbol: m for m in universe}

    for sym, df in history.items():
        if len(df) < 120:
            continue
        feats = compute_features(df, benchmark_df)
        feats["symbol"] = sym
        feats["date"] = feats.index
        m = meta_map.get(sym)
        feats["sector"] = m.sector if m else "Equities"
        feats["index_membership"] = m.index_membership if m else "SP500"
        feats["is_shortable"] = bool(m.is_shortable) if m else True

        if include_labels:
            labels = compute_triple_barrier_labels(
                df,
                feats["atr_14"],
                target_atr_mult=cfg.target_atr_mult,
                stop_atr_mult=cfg.stop_atr_mult,
                horizon_days=cfg.horizon_days,
            )
            feats = pd.concat([feats, labels], axis=1)

        frames.append(feats.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_walk_forward_backtest(
    panel: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    cfg: ScreenerConfig,
    n_splits: int = 6,
) -> dict[str, Any]:
    valid = panel.dropna(subset=["y_long", "y_short", "atr_14"]).copy()
    valid = valid[valid["avg_dollar_vol_m"] >= cfg.min_avg_dollar_vol_m]
    valid = valid.sort_values("date")

    unique_dates = np.sort(valid["date"].unique())
    if len(unique_dates) < 180:
        return {
            "equity_curve": [],
            "metrics": {
                "precision_at_5_long": 0.0,
                "precision_at_5_short": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "oos_days": 0,
            },
        }

    split_points = np.linspace(int(len(unique_dates) * 0.45), len(unique_dates) - 1, n_splits + 1, dtype=int)
    oos_preds: list[pd.DataFrame] = []

    for s in range(n_splits):
        train_end_idx = max(0, split_points[s] - cfg.horizon_days)
        test_start_idx = split_points[s]
        test_end_idx = split_points[s + 1]

        train_cutoff = unique_dates[train_end_idx]
        test_start = unique_dates[test_start_idx]
        test_end = unique_dates[test_end_idx]

        tr = valid[valid["date"] <= train_cutoff]
        te = valid[(valid["date"] >= test_start) & (valid["date"] <= test_end)].copy()
        if len(tr) < 500 or te.empty:
            continue

        m_long = _make_lgbm()
        m_long.fit(tr[FEATURE_COLS], tr["y_long"].astype(int))
        te["prob_long"] = m_long.predict_proba(te[FEATURE_COLS])[:, 1]

        m_short = _make_lgbm()
        m_short.fit(tr[FEATURE_COLS], tr["y_short"].astype(int))
        te["prob_short"] = m_short.predict_proba(te[FEATURE_COLS])[:, 1]

        lr_p_l, lr_p_s = train_logreg_fold(tr, te)
        te["lr_prob_long"] = lr_p_l
        te["lr_prob_short"] = lr_p_s
        te["fold_id"] = s

        oos_preds.append(te)

    if not oos_preds:
        return {"equity_curve": [], "metrics": {}}

    oos = pd.concat(oos_preds, ignore_index=True)
    oos_dates = np.sort(oos["date"].unique())
    rebal_dates = oos_dates[:: max(5, cfg.horizon_days // 2)]

    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    curve: list[dict[str, Any]] = []
    long_hits: list[float] = []
    short_hits: list[float] = []
    all_rets: list[float] = []

    b_series = benchmark_df["close"] if not benchmark_df.empty else pd.Series(dtype=float)
    b_base = None

    for dt in rebal_dates:
        day_df = oos[oos["date"] == dt].copy()
        day_df["target_gain_pct"] = (cfg.target_atr_mult * day_df["atr_14"] / day_df["close"]) * 100.0
        elig = day_df[day_df["target_gain_pct"] >= cfg.min_target_gain_pct]
        long_pool = elig[elig["prob_long"] >= cfg.min_win_prob]
        top_l = long_pool.nlargest(cfg.top_k, "prob_long") if not long_pool.empty else pd.DataFrame()
        short_df = elig[(elig["is_shortable"]) & (elig["prob_short"] >= cfg.min_win_prob)]
        top_s = short_df.nlargest(min(3, cfg.top_k), "prob_short") if not short_df.empty else pd.DataFrame()

        if not top_l.empty:
            long_hits.extend(top_l["y_long"].tolist())
            step_rets = top_l["long_ret"].tolist()
        else:
            step_rets = []
        if not top_s.empty:
            short_hits.extend(top_s["y_short"].tolist())
            step_rets.extend(top_s["short_ret"].tolist())

        if step_rets:
            avg_r = float(np.mean(step_rets))
            all_rets.extend(step_rets)
            equity *= 1.0 + avg_r * 0.35
            peak = max(peak, equity)
            dd = (equity - peak) / peak
            max_dd = min(max_dd, dd)

        b_val = 100.0
        if not b_series.empty:
            idx_loc = b_series.index.searchsorted(pd.Timestamp(dt))
            idx_loc = min(idx_loc, len(b_series) - 1)
            px = float(b_series.iloc[idx_loc])
            if b_base is None:
                b_base = px
            b_val = 100.0 * (px / b_base)

        curve.append(
            {
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "strategy": round(equity, 2),
                "benchmark": round(b_val, 2),
            }
        )

    wins = [r for r in all_rets if r > 0]
    losses = [-r for r in all_rets if r <= 0]
    pf = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else 2.0

    screener_eval = compute_screener_evaluation(oos, None, cfg, panel=valid)

    return {
        "equity_curve": curve,
        "screener_eval": screener_eval,
        "metrics": {
            "precision_at_5_long": round(float(np.mean(long_hits) * 100), 1) if long_hits else 0.0,
            "precision_at_5_short": round(float(np.mean(short_hits) * 100), 1) if short_hits else 0.0,
            "win_rate": round(float(len(wins) / max(1, len(all_rets)) * 100), 1),
            "profit_factor": round(float(pf), 2),
            "max_drawdown_pct": round(float(max_dd * 100), 1),
            "total_trades": len(all_rets),
            "oos_days": len(oos_dates),
        },
    }


def train_and_save_models(
    universe: list[UniverseMember],
    history: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    cfg: ScreenerConfig,
    data_mode: str = "LIVE_EOD",
) -> TrainedArtifacts:
    panel = build_panel_dataset(universe, history, benchmark_df, cfg, include_labels=True)
    if panel.empty:
        raise ValueError("No sufficient historical data to train models.")

    backtest = run_walk_forward_backtest(panel, benchmark_df, cfg)

    valid = panel.dropna(subset=["y_long", "y_short", "atr_14"]).copy()
    valid = valid[valid["avg_dollar_vol_m"] >= cfg.min_avg_dollar_vol_m]

    model_long = _make_lgbm()
    model_long.fit(valid[FEATURE_COLS], valid["y_long"].astype(int))

    valid_short = valid[valid["is_shortable"]]
    if len(valid_short) < 200:
        valid_short = valid
    model_short = _make_lgbm()
    model_short.fit(valid_short[FEATURE_COLS], valid_short["y_short"].astype(int))

    importances = model_long.feature_importances_
    total_imp = float(np.sum(importances)) or 1.0
    feat_imp = sorted(
        [
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "importance_pct": round(float(imp / total_imp * 100), 2),
            }
            for col, imp in zip(FEATURE_COLS, importances)
        ],
        key=lambda x: x["importance_pct"],
        reverse=True,
    )[:12]

    artifacts = TrainedArtifacts(
        model_long=model_long,
        model_short=model_short,
        config=cfg,
        trained_at=datetime.now(US_EASTERN).isoformat(timespec="seconds"),
        data_mode=data_mode,
        backtest=backtest,
        feature_importance=feat_imp,
    )
    save_artifacts(artifacts)
    return artifacts


def save_artifacts(artifacts: TrainedArtifacts, models_dir: Path = MODELS_DIR) -> None:
    with open(models_dir / "model_long.pkl", "wb") as f:
        pickle.dump(artifacts.model_long, f)
    with open(models_dir / "model_short.pkl", "wb") as f:
        pickle.dump(artifacts.model_short, f)
    meta = {
        "config": artifacts.config.model_dump(),
        "trained_at": artifacts.trained_at,
        "data_mode": artifacts.data_mode,
        "backtest": artifacts.backtest,
        "feature_importance": artifacts.feature_importance,
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_artifacts(models_dir: Path = MODELS_DIR) -> TrainedArtifacts | None:
    lp = models_dir / "model_long.pkl"
    sp = models_dir / "model_short.pkl"
    mp = models_dir / "metadata.json"
    if not (lp.exists() and sp.exists() and mp.exists()):
        return None
    with open(lp, "rb") as f:
        m_long = pickle.load(f)
    with open(sp, "rb") as f:
        m_short = pickle.load(f)
    meta = json.loads(mp.read_text(encoding="utf-8"))
    return TrainedArtifacts(
        model_long=m_long,
        model_short=m_short,
        config=ScreenerConfig(**meta["config"]),
        trained_at=meta["trained_at"],
        data_mode=meta.get("data_mode", "LIVE_EOD"),
        backtest=meta.get("backtest", {}),
        feature_importance=meta.get("feature_importance", []),
    )


def _extract_top_shap(
    explainer: shap.TreeExplainer, row_df: pd.DataFrame, top_n: int = 4
) -> list[dict[str, Any]]:
    row_df = row_df.astype(float)
    vals = explainer.shap_values(row_df)
    if isinstance(vals, list):
        sv = np.asarray(vals[-1])[0]
    elif getattr(vals, "ndim", 1) == 3:
        sv = vals[0, :, -1]
    else:
        sv = np.asarray(vals)[0]

    idxs = np.argsort(np.abs(sv))[::-1][:top_n]
    drivers: list[dict[str, Any]] = []
    for idx in idxs:
        col = FEATURE_COLS[int(idx)]
        impact = float(sv[int(idx)])
        raw_val = float(row_df[col].iloc[0]) if np.isfinite(row_df[col].iloc[0]) else 0.0
        drivers.append(
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "shap_impact": round(impact, 3),
                "direction": "bullish" if impact > 0 else "bearish",
                "value": round(raw_val, 3),
            }
        )
    return drivers


def scan_latest(
    artifacts: TrainedArtifacts,
    universe: list[UniverseMember],
    history: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    cfg: ScreenerConfig,
    held_symbols: list[str] | None = None,
    scan_mode: str = "LIVE_EOD",
) -> dict[str, Any]:
    meta_map = {m.symbol: m for m in universe}
    latest_rows: list[pd.Series] = []
    latest_date: pd.Timestamp | None = None

    for sym, df in history.items():
        if len(df) < 120:
            continue
        feats = compute_features(df, benchmark_df)
        last = feats.iloc[-1].copy()
        if not np.isfinite(last["atr_14"]) or last["atr_14"] <= 0:
            continue
        dt = feats.index[-1]
        if latest_date is None or dt > latest_date:
            latest_date = dt
        last["symbol"] = sym
        m = meta_map.get(sym)
        last["name"] = m.name if m else sym
        last["sector"] = m.sector if m else "Equities"
        last["index_membership"] = m.index_membership if m else "SP500"
        last["is_shortable"] = bool(m.is_shortable) if m else True
        latest_rows.append(last)

    if not latest_rows:
        return {
            "scan_date": datetime.now(US_EASTERN).strftime("%Y-%m-%d"),
            "scan_mode": scan_mode,
            "top_longs": [],
            "top_shorts": [],
            "holdings_alerts": [],
            "universe_scanned": 0,
        }

    snap = pd.DataFrame(latest_rows).reset_index(drop=True)
    liquid = snap[snap["avg_dollar_vol_m"] >= cfg.min_avg_dollar_vol_m].copy()
    if liquid.empty:
        liquid = snap.copy()

    liquid["prob_long"] = artifacts.model_long.predict_proba(liquid[FEATURE_COLS])[:, 1]
    liquid["prob_short"] = artifacts.model_short.predict_proba(liquid[FEATURE_COLS])[:, 1]

    rr = cfg.target_atr_mult / cfg.stop_atr_mult
    liquid["score_long"] = liquid["prob_long"] * rr * 100.0
    liquid["score_short"] = liquid["prob_short"] * rr * 100.0
    liquid["target_gain_pct"] = (cfg.target_atr_mult * liquid["atr_14"] / liquid["close"]) * 100.0

    eligible = liquid[liquid["target_gain_pct"] >= cfg.min_target_gain_pct].copy()

    exp_long = shap.TreeExplainer(artifacts.model_long)
    exp_short = shap.TreeExplainer(artifacts.model_short)

    def _build_card(row: pd.Series, side: str, rank: int) -> dict[str, Any]:
        entry = float(row["close"])
        atr = float(row["atr_14"])
        if side == "LONG":
            tp = entry + cfg.target_atr_mult * atr
            sl = entry - cfg.stop_atr_mult * atr
            prob = float(row["prob_long"])
            score = float(row["score_long"])
            drivers = _extract_top_shap(exp_long, row[FEATURE_COLS].to_frame().T)
        else:
            tp = entry - cfg.target_atr_mult * atr
            sl = entry + cfg.stop_atr_mult * atr
            prob = float(row["prob_short"])
            score = float(row["score_short"])
            drivers = _extract_top_shap(exp_short, row[FEATURE_COLS].to_frame().T)

        target_gain_pct = ((tp - entry) / entry * 100.0) if side == "LONG" else ((entry - tp) / entry * 100.0)
        return {
            "rank": rank,
            "symbol": str(row["symbol"]),
            "name": str(row.get("name") or row["symbol"]),
            "sector": str(row["sector"]),
            "index_membership": str(row["index_membership"]),
            "is_shortable": bool(row["is_shortable"]),
            "side": side,
            "grade": _grade_from_score(score),
            "score": round(score, 1),
            "win_prob_pct": round(prob * 100, 1),
            "expected_rr": round(rr, 2),
            "target_gain_pct": round(target_gain_pct, 2),
            "entry": round(entry, 2),
            "target": round(tp, 2),
            "stop_loss": round(sl, 2),
            "atr_pct": round(float(row["atr_pct"]) * 100, 2),
            "rsi_14": round(float(row["rsi_14"]), 1) if np.isfinite(row["rsi_14"]) else 50.0,
            "vol_ratio_20": round(float(row["vol_ratio_20"]), 2) if np.isfinite(row["vol_ratio_20"]) else 1.0,
            "avg_dollar_vol_m": round(float(row["avg_dollar_vol_m"]), 1),
            "drivers": drivers,
        }

    long_pool = eligible[eligible["prob_long"] >= cfg.min_win_prob] if not eligible.empty else pd.DataFrame()
    top_l_df = long_pool.nlargest(cfg.top_k, "prob_long") if not long_pool.empty else pd.DataFrame()
    top_longs = [_build_card(r, "LONG", i + 1) for i, (_, r) in enumerate(top_l_df.iterrows())]

    short_pool = eligible[(eligible["is_shortable"]) & (eligible["prob_short"] >= cfg.min_win_prob)] if not eligible.empty else pd.DataFrame()
    top_s_df = short_pool.nlargest(cfg.top_k, "prob_short") if not short_pool.empty else pd.DataFrame()
    top_shorts = [_build_card(r, "SHORT", i + 1) for i, (_, r) in enumerate(top_s_df.iterrows())]

    holdings_alerts: list[dict[str, Any]] = []
    if held_symbols:
        held_set = {s.upper().strip() for s in held_symbols}
        held_df = snap[snap["symbol"].isin(held_set)].copy()
        if not held_df.empty:
            held_df["prob_short"] = artifacts.model_short.predict_proba(held_df[FEATURE_COLS])[:, 1]
            for _, r in held_df.sort_values("prob_short", ascending=False).iterrows():
                prob_b = float(r["prob_short"])
                risk_level = "HIGH" if prob_b >= 0.58 else ("ELEVATED" if prob_b >= 0.45 else "LOW")
                holdings_alerts.append(
                    {
                        "symbol": str(r["symbol"]),
                        "name": str(r.get("name") or r["symbol"]),
                        "sector": str(r["sector"]),
                        "close": round(float(r["close"]), 2),
                        "exit_risk_pct": round(prob_b * 100, 1),
                        "risk_level": risk_level,
                        "rsi_14": round(float(r["rsi_14"]), 1) if np.isfinite(r["rsi_14"]) else 50.0,
                        "drivers": _extract_top_shap(exp_short, r[FEATURE_COLS].to_frame().T, top_n=3),
                    }
                )

    return {
        "scan_date": latest_date.strftime("%Y-%m-%d") if latest_date is not None else datetime.now(US_EASTERN).strftime("%Y-%m-%d"),
        "scan_mode": scan_mode,
        "top_longs": top_longs,
        "top_shorts": top_shorts,
        "holdings_alerts": holdings_alerts,
        "universe_scanned": len(snap),
    }


def score_single_stock(
    artifacts: TrainedArtifacts,
    universe: list[UniverseMember],
    history: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    cfg: ScreenerConfig,
    query: str,
) -> dict[str, Any] | None:
    """Look up a stock in the universe by symbol or company name and return ML recommendation."""
    q = query.strip().upper()
    if not q:
        return None

    meta_map = {m.symbol.upper(): m for m in universe}
    matched_sym: str | None = None
    if q in history:
        matched_sym = q
    else:
        for m in universe:
            if m.symbol.upper() == q or m.name.upper() == q:
                if m.symbol in history:
                    matched_sym = m.symbol
                    break
        if matched_sym is None:
            for m in universe:
                if (q in m.name.upper() or q in m.symbol.upper()) and m.symbol in history:
                    matched_sym = m.symbol
                    break

    if matched_sym is None or len(history[matched_sym]) < 120:
        return None

    df = history[matched_sym]
    feats = compute_features(df, benchmark_df)
    last = feats.iloc[-1]
    atr = float(last["atr_14"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    row_df = last[FEATURE_COLS].to_frame().T.astype(float)
    prob_long = float(artifacts.model_long.predict_proba(row_df)[0, 1])
    prob_short = float(artifacts.model_short.predict_proba(row_df)[0, 1])

    m = meta_map.get(matched_sym)
    is_shortable = bool(m.is_shortable) if m else True
    side = "SHORT" if (is_shortable and prob_short > prob_long) else "LONG"

    entry = float(last["close"])
    rr = cfg.target_atr_mult / cfg.stop_atr_mult
    if side == "LONG":
        tp = entry + cfg.target_atr_mult * atr
        sl = entry - cfg.stop_atr_mult * atr
        prob = prob_long
        exp = shap.TreeExplainer(artifacts.model_long)
    else:
        tp = entry - cfg.target_atr_mult * atr
        sl = entry + cfg.stop_atr_mult * atr
        prob = prob_short
        exp = shap.TreeExplainer(artifacts.model_short)

    score = prob * rr * 100.0
    target_gain_pct = ((tp - entry) / entry * 100.0) if side == "LONG" else ((entry - tp) / entry * 100.0)
    sigma_h = atr * float(np.sqrt(cfg.horizon_days))

    return {
        "symbol": matched_sym,
        "name": m.name if m else matched_sym,
        "sector": m.sector if m else "Equities",
        "index_membership": m.index_membership if m else "SP500",
        "as_of": feats.index[-1].strftime("%Y-%m-%d"),
        "side": side,
        "grade": _grade_from_score(score),
        "score": round(score, 1),
        "win_prob_pct": round(prob * 100.0, 1),
        "long_prob_pct": round(prob_long * 100.0, 1),
        "short_prob_pct": round(prob_short * 100.0, 1),
        "expected_rr": round(rr, 2),
        "target_gain_pct": round(target_gain_pct, 2),
        "entry": round(entry, 2),
        "target": round(tp, 2),
        "stop_loss": round(sl, 2),
        "range_low": round(entry - sigma_h, 2),
        "range_high": round(entry + sigma_h, 2),
        "horizon_days": cfg.horizon_days,
        "rsi_14": round(float(last["rsi_14"]), 1) if np.isfinite(last["rsi_14"]) else 50.0,
        "atr_pct": round(float(last["atr_pct"]) * 100.0, 2),
        "drivers": _extract_top_shap(exp, row_df),
    }
