from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import DATA_DIR


US_EASTERN = ZoneInfo("America/New_York")
LEDGER_CSV_PATH = DATA_DIR / "recommendations_log.csv"

LEDGER_COLUMNS = [
    "rec_date",
    "scan_mode",
    "symbol",
    "name",
    "sector",
    "side",
    "rank",
    "grade",
    "score",
    "win_prob_pct",
    "entry_price",
    "target_price",
    "target_gain_pct",
    "stop_loss",
    "horizon_days",
    "status",
    "resolved_date",
    "exit_price",
    "pnl_pct",
    "days_elapsed",
    "last_eod_date",
    "last_eod_price",
    "eod_trail",
]


def load_ledger(path: Path = LEDGER_CSV_PATH) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            for col in LEDGER_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[LEDGER_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def save_ledger(df: pd.DataFrame, path: Path = LEDGER_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[LEDGER_COLUMNS].to_csv(path, index=False)


def sort_ledger_by_win_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Sort ledger rows by rec_date DESC, then rank ASC (1..K), then score DESC."""
    if df.empty:
        return df
    out = df.copy()
    out["_rk"] = pd.to_numeric(out["rank"], errors="coerce").fillna(999.0)
    out["_sc"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    out = out.sort_values(["rec_date", "_rk", "_sc"], ascending=[False, True, False]).drop(columns=["_rk", "_sc"])
    return out.reset_index(drop=True)


def record_and_update_ledger(
    scan_result: dict[str, Any],
    history: dict[str, pd.DataFrame],
    horizon_days: int,
    path: Path = LEDGER_CSV_PATH,
) -> pd.DataFrame:
    df = load_ledger(path)
    df = df[df["scan_mode"] == "LIVE_EOD"].copy()
    scan_date = str(scan_result.get("scan_date") or datetime.now(US_EASTERN).strftime("%Y-%m-%d"))
    scan_mode = str(scan_result.get("scan_mode") or "LIVE_EOD")

    new_rows: list[dict[str, Any]] = []
    if scan_mode == "LIVE_EOD":
        for item in (scan_result.get("top_longs") or []) + (scan_result.get("top_shorts") or []):
            ent = float(item["entry"])
            tgt = float(item["target"])
            tgt_gain = ((tgt - ent) / ent * 100.0) if str(item["side"]) == "LONG" else ((ent - tgt) / ent * 100.0)
            new_rows.append(
                {
                    "rec_date": scan_date,
                    "scan_mode": scan_mode,
                    "symbol": str(item["symbol"]),
                    "name": str(item.get("name") or item["symbol"]),
                    "sector": str(item.get("sector", "")),
                    "side": str(item["side"]),
                    "rank": str(item["rank"]),
                    "grade": str(item["grade"]),
                    "score": str(item["score"]),
                    "win_prob_pct": str(item["win_prob_pct"]),
                    "entry_price": f"{ent:.2f}",
                    "target_price": f"{tgt:.2f}",
                    "target_gain_pct": f"{tgt_gain:.2f}",
                    "stop_loss": f"{float(item['stop_loss']):.2f}",
                    "horizon_days": str(horizon_days),
                    "status": "OPEN",
                    "resolved_date": "",
                    "exit_price": "",
                    "pnl_pct": "0.00",
                    "days_elapsed": "0",
                    "last_eod_date": scan_date,
                    "last_eod_price": f"{ent:.2f}",
                    "eod_trail": f"{scan_date}:{ent:.2f}",
                }
            )

    if new_rows:
        df = df[~((df["rec_date"] == scan_date) & (df["scan_mode"] == scan_mode))].copy()
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    for idx, row in df.iterrows():
        if row["status"] != "OPEN":
            continue
        sym = str(row["symbol"])
        hist = history.get(sym)
        if hist is None or hist.empty:
            continue

        rec_ts = pd.Timestamp(row["rec_date"]).normalize()
        fwd = hist[hist.index > rec_ts].sort_index()
        if fwd.empty:
            continue

        entry = float(row["entry_price"])
        target = float(row["target_price"])
        stop = float(row["stop_loss"])
        side = str(row["side"])
        max_days = int(float(row["horizon_days"] or horizon_days))

        trail_parts = [f"{row['rec_date']}:{entry:.2f}"]
        status = "OPEN"
        resolved_date = ""
        exit_price = ""
        days_elapsed = 0
        last_dt = str(row["rec_date"])
        last_px = entry

        for dt, bar in fwd.iterrows():
            days_elapsed += 1
            dt_str = dt.strftime("%Y-%m-%d")
            hi = float(bar["high"])
            lo = float(bar["low"])
            cl = float(bar["close"])
            last_dt = dt_str
            last_px = cl
            trail_parts.append(f"{dt_str}:{cl:.2f}")

            if side == "LONG":
                if lo <= stop:
                    status = "STOP_HIT"
                    resolved_date = dt_str
                    exit_price = f"{stop:.2f}"
                    last_px = stop
                    break
                if hi >= target:
                    status = "TARGET_HIT"
                    resolved_date = dt_str
                    exit_price = f"{target:.2f}"
                    last_px = target
                    break
            else:
                if hi >= stop:
                    status = "STOP_HIT"
                    resolved_date = dt_str
                    exit_price = f"{stop:.2f}"
                    last_px = stop
                    break
                if lo <= target:
                    status = "TARGET_HIT"
                    resolved_date = dt_str
                    exit_price = f"{target:.2f}"
                    last_px = target
                    break

            if days_elapsed >= max_days:
                status = "EXPIRED"
                resolved_date = dt_str
                exit_price = f"{cl:.2f}"
                break

        pnl = ((last_px - entry) / entry * 100.0) if side == "LONG" else ((entry - last_px) / entry * 100.0)
        df.at[idx, "status"] = status
        df.at[idx, "resolved_date"] = resolved_date
        df.at[idx, "exit_price"] = exit_price
        df.at[idx, "pnl_pct"] = f"{pnl:.2f}"
        df.at[idx, "days_elapsed"] = str(days_elapsed)
        df.at[idx, "last_eod_date"] = last_dt
        df.at[idx, "last_eod_price"] = f"{last_px:.2f}"
        df.at[idx, "eod_trail"] = " | ".join(trail_parts)

    df = sort_ledger_by_win_prob(df)
    save_ledger(df, path)
    return df


def get_trade_chart_data(
    symbol: str,
    rec_date: str,
    side: str,
    history: dict[str, pd.DataFrame],
    pre_days: int = 30,
    path: Path = LEDGER_CSV_PATH,
) -> dict[str, Any] | None:
    df = load_ledger(path)
    match = df[
        (df["symbol"] == symbol.upper().strip())
        & (df["rec_date"] == rec_date)
        & (df["side"] == side.upper().strip())
    ]
    if match.empty:
        return None

    trade = match.iloc[0].to_dict()
    hist = history.get(symbol.upper().strip())
    if hist is None or hist.empty:
        return {"trade": trade, "candles": [], "post_days": []}

    rec_ts = pd.Timestamp(rec_date).normalize()
    end_ts = pd.Timestamp(trade["resolved_date"]).normalize() if trade.get("resolved_date") else hist.index.max()

    pre = hist[hist.index <= rec_ts].tail(pre_days)
    post = hist[(hist.index > rec_ts) & (hist.index <= end_ts)]
    combined = pd.concat([pre, post]).sort_index()

    entry = float(trade["entry_price"])
    candles = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": round(float(r["open"]), 2),
            "high": round(float(r["high"]), 2),
            "low": round(float(r["low"]), 2),
            "close": round(float(r["close"]), 2),
            "volume": int(r["volume"]),
            "is_rec_date": idx == rec_ts,
            "is_post_rec": idx > rec_ts,
        }
        for idx, r in combined.iterrows()
    ]

    post_rows = []
    for idx, r in post.iterrows():
        cl = float(r["close"])
        chg = ((cl - entry) / entry * 100.0) if trade["side"] == "LONG" else ((entry - cl) / entry * 100.0)
        post_rows.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(r["open"]), 2),
                "high": round(float(r["high"]), 2),
                "low": round(float(r["low"]), 2),
                "close": round(cl, 2),
                "pnl_pct": round(chg, 2),
            }
        )

    return {"trade": trade, "candles": candles, "post_days": post_rows}


def render_ledger_html(df: pd.DataFrame) -> str:
    """Render a standalone 2-tab spreadsheet (Long Trades & Short Trades) with chart hyperlinks."""
    status_colors = {
        "TARGET_HIT": "#10b981",
        "STOP_HIT": "#f43f5e",
        "EXPIRED": "#f59e0b",
        "OPEN": "#38bdf8",
    }

    def _build_rows(sub_df: pd.DataFrame) -> str:
        if sub_df.empty:
            return "<tr><td colspan='15' style='text-align:center;padding:24px;color:#94a3b8'>No recommendations logged yet.</td></tr>"
        out = []
        for _, r in sub_df.iterrows():
            st = str(r["status"])
            st_color = status_colors.get(st, "#94a3b8")
            pnl_val = float(r["pnl_pct"]) if r["pnl_pct"] else 0.0
            pnl_color = "#10b981" if pnl_val > 0 else ("#f43f5e" if pnl_val < 0 else "#cbd5e1")
            tgt_gain = str(r.get("target_gain_pct") or "0.00").strip()
            sym = str(r["symbol"])
            nm = str(r.get("name") or sym).strip() or sym
            rec_dt = str(r["rec_date"])
            side = str(r["side"])
            chart_href = f"/?open_trade={escape(sym)}&rec_date={escape(rec_dt)}&side={escape(side)}"
            out.append(
                "<tr>"
                f"<td>{escape(rec_dt)}</td>"
                f"<td><a class='sym-link' href='{chart_href}' target='_blank' rel='noopener noreferrer'>{escape(nm)}</a> <span style='color:#64748b;font-size:10px'>({escape(sym)})</span></td>"
                f"<td>{escape(str(r['sector']))}</td>"
                f"<td>#{escape(str(r['rank']))} ({escape(str(r['grade']))})</td>"
                f"<td>{escape(str(r['win_prob_pct']))}%</td>"
                f"<td class='num'>${escape(str(r['entry_price']))}</td>"
                f"<td class='num pos'>${escape(str(r['target_price']))}</td>"
                f"<td class='num pos'>+{escape(tgt_gain)}%</td>"
                f"<td class='num neg'>${escape(str(r['stop_loss']))}</td>"
                f"<td><span class='badge' style='border-color:{st_color};color:{st_color}'>{escape(st)}</span></td>"
                f"<td>{escape(str(r['resolved_date'] or '—'))}</td>"
                f"<td class='num'>{'$' + escape(str(r['exit_price'])) if r['exit_price'] else '—'}</td>"
                f"<td class='num' style='color:{pnl_color};font-weight:600'>{pnl_val:+.2f}%</td>"
                f"<td class='num'>{escape(str(r['days_elapsed']))}/{escape(str(r['horizon_days']))}</td>"
                f"<td class='num'>${escape(str(r['last_eod_price']))} ({escape(str(r['last_eod_date']))})</td>"
                "</tr>"
            )
        return "\n".join(out)

    df = sort_ledger_by_win_prob(df)
    long_df = df[df["side"] == "LONG"]
    short_df = df[df["side"] == "SHORT"]
    long_rows = _build_rows(long_df)
    short_rows = _build_rows(short_df)
    headers = (
        "<tr>"
        "<th>Rec Date</th><th>Company (Chart ↗)</th><th>Sector</th><th>Rank / Grade</th>"
        "<th>Win Prob</th><th>Entry</th><th>Target</th><th>Target Gain %</th><th>Stop Loss</th>"
        "<th>Status</th><th>Resolved</th><th>Exit</th><th>P&amp;L %</th><th>Days</th><th>Latest EOD</th>"
        "</tr>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>US Trading App — Recommendations Spreadsheet</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: #020617; color: #e2e8f0; padding: 24px; }}
    .hdr {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 12px; }}
    h1 {{ margin: 0; font-size: 18px; }}
    .sub {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .btn {{ background: #0284c7; color: #fff; text-decoration: none; padding: 8px 14px; border-radius: 8px; font-size: 12px; font-weight: 600; }}
    .btn:hover {{ background: #0ea5e9; }}
    .tabs {{ display: flex; gap: 8px; margin-bottom: 14px; }}
    .tab {{ padding: 8px 16px; border-radius: 8px; border: 1px solid #1e293b; background: #0f172a; color: #94a3b8; font-size: 13px; font-weight: 600; cursor: pointer; }}
    .tab.active-long {{ background: #064e3b; color: #6ee7b7; border-color: #10b981; }}
    .tab.active-short {{ background: #4c0519; color: #fda4af; border-color: #f43f5e; }}
    .wrap {{ overflow-x: auto; border: 1px solid #1e293b; border-radius: 10px; background: #0f172a; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; white-space: nowrap; }}
    th {{ background: #1e293b; color: #94a3b8; text-align: left; padding: 10px 12px; font-weight: 600; position: sticky; top: 0; }}
    td {{ padding: 9px 12px; border-top: 1px solid #1e293b; }}
    tr:hover td {{ background: #1e293b66; }}
    .sym-link {{ font-weight: 700; color: #38bdf8; text-decoration: underline; text-underline-offset: 3px; }}
    .sym-link:hover {{ color: #7dd3fc; }}
    .num {{ font-family: ui-monospace, monospace; text-align: right; }}
    .pos {{ color: #34d399; }}
    .neg {{ color: #fb7185; }}
    .badge {{ border: 1px solid; padding: 2px 7px; border-radius: 999px; font-size: 10px; font-weight: 700; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <div class="hdr">
    <div>
      <h1>US Trading App — S&amp;P 500 &amp; Nasdaq 100 Spreadsheet</h1>
      <div class="sub">Total logged: {len(df)} ({len(long_df)} Long · {len(short_df)} Short) · Click any symbol to open its price chart</div>
    </div>
    <a class="btn" href="/api/ledger/csv" download="recommendations_log.csv">Download CSV (Excel)</a>
  </div>

  <div class="tabs">
    <button id="btn-long" class="tab active-long" onclick="showTab('long')">Long Trades ({len(long_df)})</button>
    <button id="btn-short" class="tab" onclick="showTab('short')">Short Trades ({len(short_df)})</button>
  </div>

  <div id="tab-long" class="wrap">
    <table><thead>{headers}</thead><tbody>{long_rows}</tbody></table>
  </div>
  <div id="tab-short" class="wrap hidden">
    <table><thead>{headers}</thead><tbody>{short_rows}</tbody></table>
  </div>

  <script>
    function showTab(which) {{
      document.getElementById('tab-long').classList.toggle('hidden', which !== 'long');
      document.getElementById('tab-short').classList.toggle('hidden', which !== 'short');
      document.getElementById('btn-long').className = 'tab' + (which === 'long' ? ' active-long' : '');
      document.getElementById('btn-short').className = 'tab' + (which === 'short' ? ' active-short' : '');
    }}
  </script>
</body>
</html>"""
