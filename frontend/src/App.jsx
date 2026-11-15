import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Cpu,
  Database,
  Download,
  ExternalLink,
  FileSpreadsheet,
  ListChecks,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldAlert,
  Trash2,
  TrendingDown,
  TrendingUp,
  X,
} from "lucide-react";
import PriceChart from "./components/PriceChart";
import EquityChart from "./components/EquityChart";

const GRADE_STYLES = {
  "A+": "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  A: "bg-teal-500/20 text-teal-300 border-teal-500/40",
  "B+": "bg-sky-500/20 text-sky-300 border-sky-500/40",
  B: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  C: "bg-slate-500/20 text-slate-300 border-slate-500/40",
};

function ModeBadge({ mode }) {
  if (mode === "LIVE_EOD") {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
        <Activity className="w-3.5 h-3.5" /> LIVE US EOD
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-sky-500/15 text-sky-300 border border-sky-500/30">
      <Cpu className="w-3.5 h-3.5" /> DEMO SANDBOX
    </span>
  );
}

function SignalCard({ item, timing, onSelect }) {
  const isLong = item.side === "LONG";
  const isEnterNow = timing?.status === "OK" && timing?.signal === "ENTER_NOW";
  const gainPct =
    item.target_gain_pct ??
    (
      ((isLong ? item.target - item.entry : item.entry - item.target) /
        item.entry) *
      100
    ).toFixed(1);

  return (
    <div
      onClick={() => onSelect(item)}
      className={`group cursor-pointer rounded-xl border-2 p-4 transition-all ${
        isEnterNow
          ? isLong
            ? "flash-enter-long"
            : "flash-enter-short"
          : "border border-slate-800 bg-slate-900/70 hover:bg-slate-900 hover:border-slate-700"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-mono text-slate-400">#{item.rank}</span>
            <h3 className="text-base font-bold tracking-tight text-white group-hover:text-sky-300 transition-colors">
              {item.name || item.symbol}
            </h3>
            <span className="text-[10px] font-mono text-slate-400">({item.symbol})</span>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded border ${
                GRADE_STYLES[item.grade] || GRADE_STYLES.C
              }`}
            >
              GRADE {item.grade}
            </span>
            {item.conviction_tier && (
              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded border ${
                  item.conviction_tier === "HIGH"
                    ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                    : item.conviction_tier === "MEDIUM"
                      ? "bg-sky-500/20 text-sky-300 border-sky-500/40"
                      : "bg-slate-800 text-slate-400 border-slate-700"
                }`}
              >
                {item.conviction_tier}
              </span>
            )}
            {item.index_membership && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                {item.index_membership}
              </span>
            )}
            {timing?.status === "OK" && (
              <span
                className={`text-[11px] font-black px-2.5 py-0.5 rounded-full border tracking-wide ${
                  timing.signal === "ENTER_NOW"
                    ? isLong
                      ? "bg-emerald-400 text-slate-950 border-emerald-200 shadow-md shadow-emerald-400/50 animate-pulse"
                      : "bg-rose-500 text-white border-rose-200 shadow-md shadow-rose-500/50 animate-pulse"
                    : timing.signal === "NEUTRAL_LIMIT"
                      ? "bg-sky-500/15 text-sky-300 border-sky-500/40"
                      : "bg-amber-500/15 text-amber-300 border-amber-500/40"
                }`}
              >
                {timing.signal === "ENTER_NOW" ? "⚡ ENTER NOW" : timing.signal}
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            {item.sector} · Avg Vol ${item.avg_dollar_vol_m}M/day
            {item.days_to_earn !== undefined && item.days_to_earn !== null && ` · Earnings in ${item.days_to_earn}d`}
          </p>
        </div>

        <div className="text-right">
          <div
            className={`inline-flex items-center gap-1 text-sm font-bold ${
              isLong ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            {isLong ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
            {item.ev_r !== undefined ? `EV ${item.ev_r >= 0 ? "+" : ""}${item.ev_r}R` : `${item.win_prob_pct}% Win`}
          </div>
          <div className="text-[11px] text-slate-400">
            {item.win_prob_pct}% TP · {item.expected_rr}R
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-slate-800/80 text-xs">
        <div>
          <span className="text-slate-500 block">Entry</span>
          <span className="font-mono font-semibold text-slate-200">${item.entry}</span>
        </div>
        <div>
          <span className="text-slate-500 block">Target</span>
          <span className="font-mono font-semibold text-emerald-400">
            ${item.target}{" "}
            <span className="text-[11px] text-emerald-300/90">(+{gainPct}%)</span>
          </span>
        </div>
        <div>
          <span className="text-slate-500 block">Stop Loss</span>
          <span className="font-mono font-semibold text-rose-400">${item.stop_loss}</span>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-slate-800/80">
        <div className="text-[11px] text-slate-400 mb-1.5 font-medium">Top ML Drivers (SHAP)</div>
        <div className="flex flex-wrap gap-1.5">
          {item.drivers?.map((d, i) => (
            <span
              key={i}
              className={`text-[11px] px-2 py-0.5 rounded-md border ${
                d.direction === "bullish"
                  ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/20"
                  : "bg-rose-500/10 text-rose-300 border-rose-500/20"
              }`}
              title={`Feature value: ${d.value}, SHAP: ${d.shap_impact}`}
            >
              {d.label} ({d.shap_impact > 0 ? "+" : ""}
              {d.shap_impact})
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("opportunities");
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState(null);
  const [scan, setScan] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [holdings, setHoldings] = useState([]);
  const [ledger, setLedger] = useState({ count: 0, longs: [], shorts: [] });
  const [trackedTradeChart, setTrackedTradeChart] = useState(null);
  const [selectedStock, setSelectedStock] = useState(null);
  const [stockCandles, setStockCandles] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [busyMsg, setBusyMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [newHolding, setNewHolding] = useState({ symbol: "", entry_price: "", qty: "10" });
  const [universeList, setUniverseList] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [lookupResult, setLookupResult] = useState(null);
  const [lookupLoading, setLookupLoading] = useState(false);
  const [entryTiming, setEntryTiming] = useState(null);
  const [stockEntryTiming, setStockEntryTiming] = useState(null);
  const [hourlyScan, setHourlyScan] = useState(null);
  const [hourlyAuto, setHourlyAuto] = useState(true);
  const [hourlyChecking, setHourlyChecking] = useState(false);
  const [alertDismissed, setAlertDismissed] = useState(false);

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Request failed");
    }
    return res.json();
  }

  async function loadAll() {
    try {
      const [h, c, s, b, hld, ldg, univ] = await Promise.all([
        api("/api/health"),
        api("/api/config"),
        api("/api/scan/latest"),
        api("/api/backtest"),
        api("/api/holdings"),
        api("/api/ledger"),
        api("/api/universe").catch(() => []),
      ]);
      setHealth(h);
      setConfig(c);
      setScan(s);
      setBacktest(b);
      setHoldings(hld);
      setLedger(ldg);
      setUniverseList(univ || []);
    } catch (e) {
      setErrorMsg(e.message);
    }
  }

  async function handleLookup(e) {
    e.preventDefault();
    const q = searchQuery.trim();
    if (!q) return;
    setLookupLoading(true);
    setErrorMsg("");
    try {
      const res = await api(`/api/analyze?q=${encodeURIComponent(q)}`);
      setLookupResult(res);
      api(`/api/entry-timing/${encodeURIComponent(res.symbol)}?side=${encodeURIComponent(res.side)}`)
        .then(setEntryTiming)
        .catch(() => setEntryTiming(null));
    } catch (err) {
      setLookupResult(null);
      setEntryTiming(null);
      setErrorMsg(err.message);
    } finally {
      setLookupLoading(false);
    }
  }

  async function checkHourlyTiming(notifyOnNew = false) {
    setHourlyChecking(true);
    try {
      const prevKeys = new Set((hourlyScan?.enter_now || []).map((x) => `${x.side}:${x.symbol}`));
      const res = await api("/api/entry-timing/scan");
      setHourlyScan(res);
      const nowList = res?.enter_now || [];
      const newlyEntered = nowList.filter((x) => !prevKeys.has(`${x.side}:${x.symbol}`));
      if (nowList.length > 0 && notifyOnNew && newlyEntered.length > 0) {
        setAlertDismissed(false);
        try {
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = "sine";
          osc.frequency.value = 880;
          gain.gain.value = 0.06;
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.18);
        } catch {}
        if (typeof Notification !== "undefined" && Notification.permission === "granted") {
          const names = newlyEntered.map((x) => `${x.symbol} (${x.side} @ $${x.current_price})`).join(", ");
          new Notification("⚡ ENTER NOW Alert — US Trading App", { body: names });
        }
      }
    } catch {
      // silent background check failure
    } finally {
      setHourlyChecking(false);
    }
  }

  useEffect(() => {
    loadAll().then(() => checkHourlyTiming(false));
    const params = new URLSearchParams(window.location.search);
    const sym = params.get("open_trade");
    const recDate = params.get("rec_date");
    const side = params.get("side") || "LONG";
    if (sym && recDate) {
      setTab("tracker");
      handleOpenTrackedTrade({ symbol: sym, rec_date: recDate, side });
    }
  }, []);

  useEffect(() => {
    if (!hourlyAuto) return;
    const timer = setInterval(() => {
      checkHourlyTiming(true);
    }, 60 * 60 * 1000);
    return () => clearInterval(timer);
  }, [hourlyAuto, hourlyScan]);

  async function handleSelectStock(item) {
    setSelectedStock(item);
    setStockEntryTiming(null);
    try {
      const [data, timing] = await Promise.all([
        api(`/api/stock/${item.symbol}?days=120`),
        api(`/api/entry-timing/${encodeURIComponent(item.symbol)}?side=${encodeURIComponent(item.side)}`).catch(
          () => null
        ),
      ]);
      setStockCandles(data.candles || []);
      setStockEntryTiming(timing);
    } catch (e) {
      setStockCandles([]);
    }
  }

  async function handleOpenTrackedTrade(row) {
    try {
      const data = await api(
        `/api/ledger/chart/${encodeURIComponent(row.symbol)}?rec_date=${encodeURIComponent(
          row.rec_date
        )}&side=${encodeURIComponent(row.side)}`
      );
      setTrackedTradeChart(data);
    } catch (e) {
      setErrorMsg(e.message);
    }
  }

  async function handleSync() {
    setErrorMsg("");
    setBusyMsg("Syncing S&P 500 & Nasdaq 100 OHLCV from Yahoo Finance into SQLite...");
    try {
      await api("/api/pipeline/sync", {
        method: "POST",
        body: JSON.stringify({
          universe_limit: config?.universe_limit || 550,
          years: config?.history_years || 10,
        }),
      });
      await loadAll();
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setBusyMsg("");
    }
  }

  async function handleTrain() {
    setErrorMsg("");
    setBusyMsg("Training Long & Short LightGBM models + Walk-Forward Backtest...");
    try {
      await api("/api/pipeline/train", { method: "POST" });
      await loadAll();
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setBusyMsg("");
    }
  }

  async function handleScan(provisional = false) {
    setErrorMsg("");
    setBusyMsg(provisional ? "Running 15:50 ET provisional snapshot scan..." : "Running US EOD ML scan...");
    try {
      const res = await api("/api/scan", {
        method: "POST",
        body: JSON.stringify({ simulate_1550_provisional: provisional }),
      });
      setScan(res);
      if (!provisional) {
        const ldg = await api("/api/ledger");
        setLedger(ldg);
      }
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setBusyMsg("");
    }
  }

  async function handleSaveConfig(e) {
    e.preventDefault();
    try {
      const updated = await api("/api/config", {
        method: "POST",
        body: JSON.stringify(config),
      });
      setConfig(updated);
      setShowSettings(false);
    } catch (err) {
      setErrorMsg(err.message);
    }
  }

  async function handleAddHolding(e) {
    e.preventDefault();
    if (!newHolding.symbol.trim()) return;
    try {
      const updated = await api("/api/holdings", {
        method: "POST",
        body: JSON.stringify({
          symbol: newHolding.symbol.trim().toUpperCase(),
          entry_price: Number(newHolding.entry_price) || 0,
          qty: Number(newHolding.qty) || 1,
        }),
      });
      setHoldings(updated);
      setNewHolding({ symbol: "", entry_price: "", qty: "10" });
      await handleScan(false);
    } catch (err) {
      setErrorMsg(err.message);
    }
  }

  async function handleDeleteHolding(sym) {
    try {
      const updated = await api(`/api/holdings/${sym}`, { method: "DELETE" });
      setHoldings(updated);
      await handleScan(false);
    } catch (err) {
      setErrorMsg(err.message);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-indigo-500 to-sky-400 flex items-center justify-center font-bold text-slate-950">
              US
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-base sm:text-lg font-bold tracking-tight text-white">
                  US Trading App · S&amp;P 500 &amp; Nasdaq 100
                </h1>
                <ModeBadge mode={scan?.scan_mode || health?.data_mode || "DEMO"} />
              </div>
              <p className="text-xs text-slate-400">
                LightGBM + Triple-Barrier ({config?.target_atr_mult ?? 2.5}x / {config?.stop_atr_mult ?? 1.5}x ATR,{" "}
                {config?.horizon_days ?? 15}D) · Scanned {scan?.universe_scanned ?? 0} stocks · Bar{" "}
                {scan?.scan_date ?? "—"}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleSync}
              disabled={!!busyMsg}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 disabled:opacity-50 cursor-pointer"
            >
              <Database className="w-3.5 h-3.5 text-sky-400" /> Sync US Data
            </button>
            <button
              onClick={handleTrain}
              disabled={!!busyMsg}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 disabled:opacity-50 cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5 text-emerald-400" /> Train ML
            </button>
            <button
              onClick={() => handleScan(false)}
              disabled={!!busyMsg}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-sky-600 hover:bg-sky-500 text-white disabled:opacity-50 cursor-pointer"
            >
              <Play className="w-3.5 h-3.5" /> Run EOD Scan
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 cursor-pointer"
              title="Screener & Triple-Barrier Settings"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-6 border-t border-slate-800/60 text-xs font-semibold">
          {[
            { id: "opportunities", label: `Top ${config?.top_k ?? 10} Longs & Shorts`, icon: TrendingUp },
            { id: "tracker", label: `Trade Tracker (${ledger.count})`, icon: ListChecks },
            { id: "holdings", label: `Holdings Exit Alerts (${holdings.length})`, icon: ShieldAlert },
            { id: "backtest", label: "Walk-Forward Backtest & SHAP", icon: BarChart3 },
          ].map((t) => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`py-2.5 inline-flex items-center gap-1.5 border-b-2 transition-colors cursor-pointer ${
                  active
                    ? "border-sky-400 text-sky-300"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {t.label}
              </button>
            );
          })}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
        {busyMsg && (
          <div className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-3 text-sm text-sky-200 flex items-center gap-2.5">
            <RefreshCw className="w-4 h-4 animate-spin" />
            {busyMsg}
          </div>
        )}
        {errorMsg && (
          <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200 flex items-center justify-between">
            <span className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" /> {errorMsg}
            </span>
            <button onClick={() => setErrorMsg("")} className="text-rose-300 hover:text-white cursor-pointer">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {hourlyScan?.enter_now?.length > 0 && !alertDismissed && (
          <div className="rounded-xl border-2 border-slate-200/40 bg-slate-950/90 px-4 py-3.5 flex flex-wrap items-center justify-between gap-3 shadow-xl">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-full bg-white text-slate-950 font-black tracking-wide animate-pulse">
                ⚡ ENTER NOW ({hourlyScan.enter_now.length})
              </span>
              {hourlyScan.enter_now.map((x) => {
                const isL = x.side === "LONG";
                return (
                  <button
                    key={`${x.side}-${x.symbol}`}
                    type="button"
                    onClick={() => {
                      const match = [...(scan?.top_longs || []), ...(scan?.top_shorts || [])].find(
                        (c) => c.symbol === x.symbol && c.side === x.side
                      );
                      if (match) handleSelectStock(match);
                    }}
                    className={`px-3 py-1 rounded-lg border-2 font-mono font-black cursor-pointer ${
                      isL
                        ? "flash-enter-long text-emerald-200"
                        : "flash-enter-short text-rose-200"
                    }`}
                  >
                    {x.symbol} · {x.side} @ ${x.current_price}
                  </button>
                );
              })}
            </div>
            <button
              type="button"
              onClick={() => setAlertDismissed(true)}
              className="text-slate-300 hover:text-white cursor-pointer"
              title="Dismiss alert banner"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2 text-slate-300">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-semibold text-white">Hourly Entry Monitor:</span>
            <span className="text-slate-400">
              {hourlyScan?.checked_at
                ? `Last checked ${hourlyScan.checked_at.slice(11, 19)} ET · ${hourlyScan.enter_now?.length || 0} ENTER_NOW`
                : "Checking 1h bars..."}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => checkHourlyTiming(true)}
              disabled={hourlyChecking}
              className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-medium disabled:opacity-50 cursor-pointer"
            >
              {hourlyChecking ? "Checking 1H..." : "Check 1H Now"}
            </button>
            <button
              type="button"
              onClick={() => setHourlyAuto((v) => !v)}
              className={`px-2.5 py-1 rounded-lg border font-semibold cursor-pointer ${
                hourlyAuto
                  ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-300"
                  : "bg-slate-800 border-slate-700 text-slate-400"
              }`}
            >
              Auto (1H): {hourlyAuto ? "ON" : "OFF"}
            </button>
            {typeof Notification !== "undefined" && Notification.permission === "default" && (
              <button
                type="button"
                onClick={() => Notification.requestPermission()}
                className="px-2.5 py-1 rounded-lg bg-sky-600/20 hover:bg-sky-600/30 border border-sky-500/40 text-sky-300 font-medium cursor-pointer"
              >
                Enable Desktop Alerts
              </button>
            )}
          </div>
        </div>

        {scan?.market_indices?.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {scan.market_indices.map((idx) => {
              const dirColor =
                idx.direction === "BULLISH"
                  ? "text-emerald-300 bg-emerald-500/15 border-emerald-500/30"
                  : idx.direction === "BEARISH"
                    ? "text-rose-300 bg-rose-500/15 border-rose-500/30"
                    : "text-amber-300 bg-amber-500/15 border-amber-500/30";
              const chgColor = idx.change_1d_pct >= 0 ? "text-emerald-400" : "text-rose-400";
              const projColor = idx.projected_change_pct >= 0 ? "text-emerald-400" : "text-rose-400";
              return (
                <div
                  key={idx.symbol}
                  className="rounded-xl border border-slate-800 bg-slate-900/80 px-4 py-3 flex flex-wrap items-center justify-between gap-3"
                >
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-white">{idx.name}</span>
                      <span className="text-[11px] font-mono text-slate-400">{idx.symbol}</span>
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${dirColor}`}>
                        {idx.direction}
                      </span>
                    </div>
                    <div className="mt-1 flex items-baseline gap-2.5">
                      <span className="text-lg font-bold font-mono text-slate-100">
                        {Number(idx.value).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </span>
                      <span className={`text-xs font-mono font-semibold ${chgColor}`}>
                        {idx.change_1d_pct >= 0 ? "+" : ""}{idx.change_1d_pct}% (1D)
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[11px] text-slate-400">{idx.horizon_days}D Est. Target</div>
                    <div className="text-sm font-bold font-mono text-slate-100">
                      {Number(idx.projected_value).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}{" "}
                      <span className={`text-xs ${projColor}`}>
                        ({idx.projected_change_pct >= 0 ? "+" : ""}{idx.projected_change_pct}%)
                      </span>
                    </div>
                    <div className="text-[10px] font-mono text-slate-500">
                      Range: {Number(idx.range_low).toLocaleString("en-US", { maximumFractionDigits: 0 })} – {Number(idx.range_high).toLocaleString("en-US", { maximumFractionDigits: 0 })}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {tab === "opportunities" && (
          <div className="space-y-5">
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 space-y-3">
              <form onSubmit={handleLookup} className="flex flex-wrap items-center gap-2.5">
                <div className="relative flex-1 min-w-[240px]">
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    list="us-universe-options"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Analyze any S&P 500 / Nasdaq 100 stock (e.g. Microsoft or MSFT)..."
                    className="w-full pl-9 pr-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-sky-500"
                  />
                  <datalist id="us-universe-options">
                    {universeList.map((u) => (
                      <option key={u.symbol} value={u.symbol}>
                        {u.name} ({u.symbol})
                      </option>
                    ))}
                  </datalist>
                </div>
                <button
                  type="submit"
                  disabled={lookupLoading}
                  className="px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold disabled:opacity-50 cursor-pointer"
                >
                  {lookupLoading ? "Analyzing..." : "Analyze Stock"}
                </button>
                {lookupResult && (
                  <button
                    type="button"
                    onClick={() => setLookupResult(null)}
                    className="px-2.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs cursor-pointer"
                  >
                    Clear
                  </button>
                )}
              </form>

              {lookupResult && (
                <div className="rounded-xl border border-sky-500/30 bg-slate-950/70 p-4 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-base font-bold text-white">{lookupResult.name}</span>
                        <span className="text-xs font-mono text-slate-400">({lookupResult.symbol})</span>
                        <span
                          className={`px-2 py-0.5 rounded-full text-[11px] font-bold border ${
                            lookupResult.side === "LONG"
                              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                              : "bg-rose-500/20 text-rose-300 border-rose-500/40"
                          }`}
                        >
                          {lookupResult.side} ({lookupResult.win_prob_pct}% Win)
                        </span>
                        <span className="text-[11px] text-slate-400">
                          Long {lookupResult.long_prob_pct}% vs Short {lookupResult.short_prob_pct}%
                        </span>
                      </div>
                      <div className="text-xs text-slate-400 mt-0.5">
                        {lookupResult.sector} · As of {lookupResult.as_of} · RSI(14) {lookupResult.rsi_14} · {lookupResult.horizon_days}D Range ${lookupResult.range_low} – ${lookupResult.range_high}
                      </div>
                    </div>
                    <button
                      onClick={() => handleSelectStock(lookupResult)}
                      className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sky-300 text-xs font-medium cursor-pointer"
                    >
                      View Chart
                    </button>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                    <div className="rounded-lg bg-slate-900/90 p-2.5">
                      <div className="text-slate-400 text-[11px]">Current Price</div>
                      <div className="font-mono font-bold text-slate-100">${lookupResult.entry}</div>
                    </div>
                    <div className="rounded-lg bg-slate-900/90 p-2.5">
                      <div className="text-slate-400 text-[11px]">Target Price</div>
                      <button
                        type="button"
                        onClick={() => handleSelectStock(lookupResult)}
                        className="font-mono font-bold text-emerald-400 hover:text-emerald-300 underline underline-offset-2 cursor-pointer text-left"
                      >
                        ${lookupResult.target} (+{lookupResult.target_gain_pct}%) · Graph ↗
                      </button>
                    </div>
                    <div className="rounded-lg bg-slate-900/90 p-2.5">
                      <div className="text-slate-400 text-[11px]">Stop Loss</div>
                      <div className="font-mono font-bold text-rose-400">${lookupResult.stop_loss}</div>
                    </div>
                    <div className="rounded-lg bg-slate-900/90 p-2.5">
                      <div className="text-slate-400 text-[11px]">Grade / R:R</div>
                      <div className="font-mono font-bold text-slate-100">
                        {lookupResult.grade} · 1:{lookupResult.expected_rr}
                      </div>
                    </div>
                  </div>
                  {entryTiming && entryTiming.status === "OK" && (
                    <div className="rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-semibold text-slate-400">1H / 4H Entry Timing:</span>
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${
                            entryTiming.signal === "ENTER_NOW"
                              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                              : entryTiming.signal === "NEUTRAL_LIMIT"
                                ? "bg-sky-500/20 text-sky-300 border-sky-500/40"
                                : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                          }`}
                        >
                          {entryTiming.signal}
                        </span>
                        <span className="text-slate-300">{entryTiming.reason}</span>
                      </div>
                      <div className="font-mono text-[11px] text-slate-300">
                        Limit Entry: <span className="text-sky-300 font-bold">${entryTiming.suggested_entry}</span> · 1h RSI {entryTiming.rsi_14_1h} · 4h RSI {entryTiming.rsi_14_4h}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" /> Top {config?.top_k ?? 5} Bullish Setups (S&amp;P 500 / NDX 100)
                </h2>
                <span className="text-xs text-slate-400">Sorted by Calibrated EV ↓</span>
              </div>
              {(scan?.top_longs || []).length > 0 &&
                !(scan?.top_longs || []).some((x) => x.conviction_tier === "HIGH") && (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
                    No High-Conviction Picks Today — showing best available setups ranked by calibrated EV.
                  </div>
                )}
              <div className="space-y-3">
                {(scan?.top_longs || []).map((item) => (
                  <SignalCard
                    key={`L-${item.symbol}`}
                    item={item}
                    timing={hourlyScan?.items?.[`${item.side}:${item.symbol}`]}
                    onSelect={handleSelectStock}
                  />
                ))}
              </div>
            </section>

            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-bold uppercase tracking-wider text-rose-400 flex items-center gap-2">
                  <TrendingDown className="w-4 h-4" />{" "}
                  {(backtest?.screener_eval?.short_summary?.avg_r ?? 0) <= 0
                    ? `Bearish Watchlist (Top ${config?.top_k ?? 5})`
                    : `Top ${config?.top_k ?? 5} Bearish Setups (Shortable)`}
                </h2>
                <span className="text-xs text-slate-400">
                  {(backtest?.screener_eval?.short_summary?.avg_r ?? 0) <= 0
                    ? "Low/Negative OOS Edge · Context Only"
                    : "S&P 500 & Nasdaq 100"}
                </span>
              </div>
              <div className="space-y-3">
                {(scan?.top_shorts || []).map((item) => (
                  <SignalCard
                    key={`S-${item.symbol}`}
                    item={item}
                    timing={hourlyScan?.items?.[`${item.side}:${item.symbol}`]}
                    onSelect={handleSelectStock}
                  />
                ))}
              </div>
            </section>
          </div>
          </div>
        )}

        {tab === "tracker" && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-bold text-white">Daily EOD Recommendation Tracker</h2>
                <p className="text-xs text-slate-400">
                  Click any hyperlinked symbol to view its pre-entry context, Entry / Stop / Target lines, and daily EOD progression.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href="/api/ledger/view"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" /> Open Spreadsheet (2 Tabs) <ExternalLink className="w-3 h-3" />
                </a>
                <a
                  href="/api/ledger/csv"
                  download="recommendations_log.csv"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                >
                  <Download className="w-3.5 h-3.5 text-sky-400" /> Download CSV
                </a>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {[
                { title: "Long Trades", rows: ledger.longs || [], accent: "text-emerald-400", Icon: TrendingUp },
                { title: "Short Trades", rows: ledger.shorts || [], accent: "text-rose-400", Icon: TrendingDown },
              ].map((sec) => {
                const SecIcon = sec.Icon;
                return (
                  <div key={sec.title} className="rounded-xl border border-slate-800 bg-slate-900/70 overflow-hidden">
                    <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                      <h3 className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 ${sec.accent}`}>
                        <SecIcon className="w-4 h-4" /> {sec.title}
                      </h3>
                      <span className="text-xs text-slate-400">{sec.rows.length} logged</span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-slate-950/70 text-slate-400 border-b border-slate-800">
                          <tr>
                            <th className="px-3 py-2.5">Date</th>
                            <th className="px-3 py-2.5">Company</th>
                            <th className="px-3 py-2.5 text-right">Entry</th>
                            <th className="px-3 py-2.5 text-right">Target (Gain %)</th>
                            <th className="px-3 py-2.5 text-right">Stop</th>
                            <th className="px-3 py-2.5">Status</th>
                            <th className="px-3 py-2.5 text-right">P&amp;L</th>
                            <th className="px-3 py-2.5 text-right">Days</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/70">
                          {sec.rows.length === 0 ? (
                            <tr>
                              <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                                No EOD recommendations logged yet. Run Sync US Data + EOD Scan to populate.
                              </td>
                            </tr>
                          ) : (
                            sec.rows.map((r) => {
                              const pnl = Number(r.pnl_pct || 0);
                              const stColor =
                                r.status === "TARGET_HIT"
                                  ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                                  : r.status === "STOP_HIT"
                                  ? "bg-rose-500/15 text-rose-300 border-rose-500/30"
                                  : r.status === "EXPIRED"
                                  ? "bg-amber-500/15 text-amber-300 border-amber-500/30"
                                  : "bg-sky-500/15 text-sky-300 border-sky-500/30";
                              return (
                                <tr key={`${r.rec_date}-${r.symbol}-${r.side}`} className="hover:bg-slate-800/40">
                                  <td className="px-3 py-2.5 font-mono text-slate-400 whitespace-nowrap">{r.rec_date}</td>
                                  <td className="px-3 py-2.5 whitespace-nowrap">
                                    <button
                                      type="button"
                                      onClick={() => handleOpenTrackedTrade(r)}
                                      className="font-bold text-sky-400 hover:text-sky-300 underline decoration-sky-500/40 underline-offset-2 cursor-pointer"
                                    >
                                      {r.name || r.symbol}
                                    </button>
                                    <span className="ml-1.5 text-[10px] font-mono text-slate-500">({r.symbol}) · {r.grade}</span>
                                  </td>
                                  <td className="px-3 py-2.5 text-right font-mono text-slate-200">${r.entry_price}</td>
                                  <td className="px-3 py-2.5 text-right font-mono text-emerald-400 whitespace-nowrap">
                                    ${r.target_price}{" "}
                                    <span className="text-[10px] text-emerald-300/80">(+{r.target_gain_pct}%)</span>
                                  </td>
                                  <td className="px-3 py-2.5 text-right font-mono text-rose-400">${r.stop_loss}</td>
                                  <td className="px-3 py-2.5 whitespace-nowrap">
                                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${stColor}`}>
                                      {r.status}
                                    </span>
                                  </td>
                                  <td
                                    className={`px-3 py-2.5 text-right font-mono font-semibold ${
                                      pnl > 0 ? "text-emerald-400" : pnl < 0 ? "text-rose-400" : "text-slate-300"
                                    }`}
                                  >
                                    {pnl >= 0 ? "+" : ""}
                                    {pnl.toFixed(2)}%
                                  </td>
                                  <td className="px-3 py-2.5 text-right font-mono text-slate-400">
                                    {r.days_elapsed}/{r.horizon_days}
                                  </td>
                                </tr>
                              );
                            })
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab === "holdings" && (
          <div className="space-y-5">
            <form
              onSubmit={handleAddHolding}
              className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 flex flex-wrap items-end gap-3"
            >
              <div>
                <label className="block text-xs text-slate-400 mb-1">US Ticker</label>
                <input
                  value={newHolding.symbol}
                  onChange={(e) => setNewHolding({ ...newHolding, symbol: e.target.value })}
                  placeholder="e.g. NVDA"
                  className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-sm text-white w-36"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Avg Entry ($)</label>
                <input
                  type="number"
                  step="0.01"
                  value={newHolding.entry_price}
                  onChange={(e) => setNewHolding({ ...newHolding, entry_price: e.target.value })}
                  placeholder="135.00"
                  className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-sm text-white w-32"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Shares</label>
                <input
                  type="number"
                  value={newHolding.qty}
                  onChange={(e) => setNewHolding({ ...newHolding, qty: e.target.value })}
                  className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-sm text-white w-24"
                />
              </div>
              <button
                type="submit"
                className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-semibold bg-sky-600 hover:bg-sky-500 text-white cursor-pointer"
              >
                <Plus className="w-4 h-4" /> Track Holding
              </button>
            </form>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {(scan?.holdings_alerts || []).map((a) => {
                const badge =
                  a.risk_level === "HIGH"
                    ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                    : a.risk_level === "ELEVATED"
                    ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                    : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
                return (
                  <div key={a.symbol} className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 space-y-3">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="font-bold text-white">
                          {a.name || a.symbol}{" "}
                          <span className="text-xs font-mono font-normal text-slate-400">({a.symbol})</span>
                        </h3>
                        <p className="text-xs text-slate-400">
                          {a.sector} · Last ${a.close}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${badge}`}>
                          {a.risk_level} EXIT RISK ({a.exit_risk_pct}%)
                        </span>
                        <button
                          onClick={() => handleDeleteHolding(a.symbol)}
                          className="text-slate-500 hover:text-rose-400 cursor-pointer"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {a.drivers?.map((d, i) => (
                        <span
                          key={i}
                          className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700"
                        >
                          {d.label} ({d.shap_impact > 0 ? "+" : ""}
                          {d.shap_impact})
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab === "backtest" && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {[
                { label: "Precision@5 (Long)", val: `${backtest?.metrics?.precision_at_5_long ?? 0}%` },
                { label: "Precision@5 (Short)", val: `${backtest?.metrics?.precision_at_5_short ?? 0}%` },
                { label: "OOS Win Rate", val: `${backtest?.metrics?.win_rate ?? 0}%` },
                { label: "Profit Factor", val: `${backtest?.metrics?.profit_factor ?? 0}` },
                { label: "Max Drawdown", val: `${backtest?.metrics?.max_drawdown_pct ?? 0}%` },
                { label: "OOS Trades", val: `${backtest?.metrics?.total_trades ?? 0}` },
              ].map((m, i) => (
                <div key={i} className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
                  <div className="text-[11px] text-slate-400">{m.label}</div>
                  <div className="text-lg font-bold text-white mt-1 font-mono">{m.val}</div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-bold text-white">Out-of-Sample Walk-Forward Equity Curve</h3>
                  <div className="flex items-center gap-4 text-xs text-slate-400">
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 bg-sky-400 inline-block" /> ML Top-K Strategy
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 bg-slate-500 inline-block" /> S&amp;P 500
                    </span>
                  </div>
                </div>
                <EquityChart curve={backtest?.equity_curve || []} />
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <h3 className="text-sm font-bold text-white mb-3">Global Top Features (LightGBM)</h3>
                <div className="space-y-2.5">
                  {(backtest?.feature_importance || []).slice(0, 8).map((f) => (
                    <div key={f.feature}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300">{f.label}</span>
                        <span className="font-mono text-slate-400">{f.importance_pct}%</span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-sky-500 rounded-full"
                          style={{ width: `${Math.min(100, f.importance_pct * 8)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {backtest?.screener_eval?.long_summary && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="rounded-xl border border-emerald-500/30 bg-slate-900/70 p-3.5">
                    <div className="text-[11px] text-slate-400">Long Top-K vs Base Rate</div>
                    <div className="text-lg font-bold text-emerald-400 mt-1 font-mono">
                      {backtest.screener_eval.long_summary.top_k_hit_pct}% vs {backtest.screener_eval.long_summary.base_rate_pct}%
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Lift +{backtest.screener_eval.long_summary.lift_pp} pp ({backtest.screener_eval.long_summary.lift_ratio}x) · Folds {backtest.screener_eval.long_summary.folds_beating_base}
                    </div>
                  </div>
                  <div className="rounded-xl border border-rose-500/30 bg-slate-900/70 p-3.5">
                    <div className="text-[11px] text-slate-400">Short Top-K vs Base Rate</div>
                    <div className="text-lg font-bold text-rose-400 mt-1 font-mono">
                      {backtest.screener_eval.short_summary.top_k_hit_pct}% vs {backtest.screener_eval.short_summary.base_rate_pct}%
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Lift +{backtest.screener_eval.short_summary.lift_pp} pp ({backtest.screener_eval.short_summary.lift_ratio}x) · Folds {backtest.screener_eval.short_summary.folds_beating_base}
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
                    <div className="text-[11px] text-slate-400">Long Rank IC (Spearman)</div>
                    <div className="text-lg font-bold text-white mt-1 font-mono">
                      {backtest.screener_eval.long_summary.rank_ic.mean} (t={backtest.screener_eval.long_summary.rank_ic.t_stat})
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Avg R: {backtest.screener_eval.long_summary.avg_r} vs {backtest.screener_eval.long_summary.base_avg_r} base
                      {backtest.screener_eval.long_summary.avg_r_lift_ci_95 &&
                        ` (Lift ${backtest.screener_eval.long_summary.avg_r_lift >= 0 ? "+" : ""}${backtest.screener_eval.long_summary.avg_r_lift}R [${backtest.screener_eval.long_summary.avg_r_lift_ci_95[0]}, ${backtest.screener_eval.long_summary.avg_r_lift_ci_95[1]}])`}
                      {backtest.screener_eval.long_summary.brier_score !== undefined &&
                        ` · Brier ${backtest.screener_eval.long_summary.brier_score} (BSS ${backtest.screener_eval.long_summary.brier_skill_score ?? 0})`}
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
                    <div className="text-[11px] text-slate-400">Short Rank IC (Spearman)</div>
                    <div className="text-lg font-bold text-white mt-1 font-mono">
                      {backtest.screener_eval.short_summary.rank_ic.mean} (t={backtest.screener_eval.short_summary.rank_ic.t_stat})
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Avg R: {backtest.screener_eval.short_summary.avg_r} vs {backtest.screener_eval.short_summary.base_avg_r} base
                      {backtest.screener_eval.short_summary.avg_r_lift_ci_95 &&
                        ` (Lift ${backtest.screener_eval.short_summary.avg_r_lift >= 0 ? "+" : ""}${backtest.screener_eval.short_summary.avg_r_lift}R [${backtest.screener_eval.short_summary.avg_r_lift_ci_95[0]}, ${backtest.screener_eval.short_summary.avg_r_lift_ci_95[1]}])`}
                      {backtest.screener_eval.short_summary.brier_score !== undefined &&
                        ` · Brier ${backtest.screener_eval.short_summary.brier_score} (BSS ${backtest.screener_eval.short_summary.brier_skill_score ?? 0})`}
                    </div>
                  </div>
                </div>

                {(backtest.screener_eval.long_summary?.daily_ic_nw || backtest.screener_eval.paired_vs_baselines) && (
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 text-xs">
                    {backtest.screener_eval.long_summary?.daily_ic_nw && (
                      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                        <div className="text-[11px] text-slate-400">Every-Day Long Rank IC (Newey-West lag=15)</div>
                        <div className="font-mono font-bold text-white mt-1">
                          {backtest.screener_eval.long_summary.daily_ic_nw.mean} (NW t={backtest.screener_eval.long_summary.daily_ic_nw.nw_t_stat})
                        </div>
                      </div>
                    )}
                    {backtest.screener_eval.paired_vs_baselines?.lgbm_minus_logreg_long_r && (
                      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                        <div className="text-[11px] text-slate-400">LightGBM EV − LogReg (Long Avg R)</div>
                        <div className="font-mono font-bold text-white mt-1">
                          {backtest.screener_eval.paired_vs_baselines.lgbm_minus_logreg_long_r.mean_diff >= 0 ? "+" : ""}
                          {backtest.screener_eval.paired_vs_baselines.lgbm_minus_logreg_long_r.mean_diff}R{" "}
                          <span className="text-slate-400 font-normal">
                            [95% CI {backtest.screener_eval.paired_vs_baselines.lgbm_minus_logreg_long_r.ci_95[0]},{" "}
                            {backtest.screener_eval.paired_vs_baselines.lgbm_minus_logreg_long_r.ci_95[1]}]
                          </span>
                        </div>
                      </div>
                    )}
                    {backtest.screener_eval.paired_vs_baselines?.lgbm_minus_mom_long_r && (
                      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                        <div className="text-[11px] text-slate-400">LightGBM EV − 12-1 Mom (Long Avg R)</div>
                        <div className="font-mono font-bold text-white mt-1">
                          {backtest.screener_eval.paired_vs_baselines.lgbm_minus_mom_long_r.mean_diff >= 0 ? "+" : ""}
                          {backtest.screener_eval.paired_vs_baselines.lgbm_minus_mom_long_r.mean_diff}R{" "}
                          <span className="text-slate-400 font-normal">
                            [95% CI {backtest.screener_eval.paired_vs_baselines.lgbm_minus_mom_long_r.ci_95[0]},{" "}
                            {backtest.screener_eval.paired_vs_baselines.lgbm_minus_mom_long_r.ci_95[1]}]
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                    <h3 className="text-sm font-bold text-white mb-3">Model vs Baselines (Top-K Hit Rate &amp; Avg R)</h3>
                    <table className="w-full text-xs">
                      <thead className="text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="text-left py-1.5">Model</th>
                          <th className="text-right py-1.5">Long Hit</th>
                          <th className="text-right py-1.5">Long R</th>
                          <th className="text-right py-1.5">Short Hit</th>
                          <th className="text-right py-1.5">Short R</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-mono">
                        {Object.entries(backtest.screener_eval.baselines || {}).map(([k, v]) => (
                          <tr key={k}>
                            <td className="py-1.5 font-sans text-slate-200 capitalize">{k.replace(/_/g, " ")}</td>
                            <td className="py-1.5 text-right text-emerald-400">{v.long_hit_pct}%</td>
                            <td className="py-1.5 text-right text-slate-300">{v.long_avg_r}</td>
                            <td className="py-1.5 text-right text-rose-400">{v.short_hit_pct}%</td>
                            <td className="py-1.5 text-right text-slate-300">{v.short_avg_r}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                    <h3 className="text-sm font-bold text-white mb-3">Long Probability Decile Lift (D1 → D10)</h3>
                    <div className="grid grid-cols-10 gap-1.5 items-end h-28 pt-2">
                      {(backtest.screener_eval.deciles_long || []).map((d) => (
                        <div key={d.decile} className="flex flex-col items-center gap-1">
                          <span className="text-[10px] font-mono text-slate-300">{d.tb_hit_rate_pct}%</span>
                          <div
                            className="w-full rounded-t bg-emerald-500/70"
                            style={{ height: `${Math.max(12, d.tb_hit_rate_pct * 1.6)}px` }}
                          />
                          <span className="text-[10px] font-mono text-slate-500">D{d.decile}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {backtest?.v3_eval?.variants && (
                  <div className="space-y-4 pt-2">
                    <div className="rounded-xl border border-sky-500/30 bg-slate-900/80 p-4 overflow-x-auto">
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                        <div>
                          <h3 className="text-sm font-bold text-sky-300">
                            Claude Feedback V3 — 14-Fold Purged Walk-Forward A/B Comparison (2020–2026)
                          </h3>
                          <p className="text-[11px] text-slate-400">
                            Pre-registered bar: NW t &gt; 3.0, D10−D1 CI &gt; 0, Paired vs 12-1 Mom CI &gt; 0 ({backtest.v3_eval.n_experiments} total experiments counted)
                          </p>
                        </div>
                      </div>
                      <table className="w-full text-xs">
                        <thead className="text-slate-400 border-b border-slate-800">
                          <tr>
                            <th className="text-left py-1.5">Variant</th>
                            <th className="text-right py-1.5">Daily IC (NW t)</th>
                            <th className="text-right py-1.5">Rel IC (NW t)</th>
                            <th className="text-right py-1.5">Folds IC&gt;0</th>
                            <th className="text-right py-1.5">D10−D1 Spread</th>
                            <th className="text-right py-1.5">Top-10 Long R (Lift)</th>
                            <th className="text-right py-1.5">Paired vs Mom</th>
                            <th className="text-right py-1.5">Passes Bar</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 font-mono">
                          {Object.entries(backtest.v3_eval.variants).map(([k, v]) => (
                            <tr key={k}>
                              <td className="py-1.5 font-sans text-slate-200">{v.label || k}</td>
                              <td className="py-1.5 text-right text-slate-300">
                                {v.daily_ic_mean} ({v.daily_ic_nw_t})
                              </td>
                              <td className="py-1.5 text-right text-slate-300">
                                {v.ic_vs_relative_target} ({v.ic_vs_relative_nw_t})
                              </td>
                              <td className="py-1.5 text-right text-slate-300">{v.fold_ic_positive}</td>
                              <td className="py-1.5 text-right text-slate-300">
                                {v.spread_top_minus_bottom_r}R
                              </td>
                              <td className="py-1.5 text-right text-emerald-400">
                                {v.top_k_long_r}R ({v.top_k_long_lift_r >= 0 ? "+" : ""}{v.top_k_long_lift_r}R)
                              </td>
                              <td className="py-1.5 text-right text-slate-300">
                                {v.paired_vs_mom_r !== undefined
                                  ? `${v.paired_vs_mom_r >= 0 ? "+" : ""}${v.paired_vs_mom_r}R`
                                  : "—"}
                              </td>
                              <td className="py-1.5 text-right">
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[10px] font-sans font-bold ${
                                    v.passes_all
                                      ? "bg-emerald-500/20 text-emerald-300"
                                      : "bg-amber-500/15 text-amber-300"
                                  }`}
                                >
                                  {v.passes_all ? "PASS" : "WATCHLIST ONLY"}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {backtest?.v3_eval?.selection_backtests && (
                      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 overflow-x-auto">
                        <h3 className="text-sm font-bold text-white mb-2">
                          V3 Portfolio Backtests (115 Non-Overlapping 15D Periods, 20 bps Round-Trip Cost · SPY Buy &amp; Hold: +153.4% / 14.55% CAGR)
                        </h3>
                        <table className="w-full text-xs">
                          <thead className="text-slate-400 border-b border-slate-800">
                            <tr>
                              <th className="text-left py-1.5">Strategy</th>
                              <th className="text-right py-1.5">Total Return</th>
                              <th className="text-right py-1.5">CAGR</th>
                              <th className="text-right py-1.5">Sharpe</th>
                              <th className="text-right py-1.5">Max DD</th>
                              <th className="text-right py-1.5">Avg Long R</th>
                              <th className="text-right py-1.5">Beta-Adj Excess / 15D [95% CI]</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/60 font-mono">
                            {Object.entries(backtest.v3_eval.selection_backtests).map(([k, v]) => (
                              <tr key={k}>
                                <td className="py-1.5 font-sans text-slate-200">{k}</td>
                                <td className="py-1.5 text-right text-emerald-400">
                                  {v.strategy?.total_return_pct >= 0 ? "+" : ""}
                                  {v.strategy?.total_return_pct}%
                                </td>
                                <td className="py-1.5 text-right text-slate-200">{v.strategy?.cagr_pct}%</td>
                                <td className="py-1.5 text-right text-slate-200">{v.strategy?.sharpe}</td>
                                <td className="py-1.5 text-right text-rose-400">{v.strategy?.max_dd_pct}%</td>
                                <td className="py-1.5 text-right text-slate-300">{v.avg_long_r}R</td>
                                <td className="py-1.5 text-right text-slate-300">
                                  {v.beta_adj_excess_per_period_pct >= 0 ? "+" : ""}
                                  {v.beta_adj_excess_per_period_pct}% [{v.beta_adj_excess_ci95?.[0]}%, {v.beta_adj_excess_ci95?.[1]}%]
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {selectedStock && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-3xl rounded-2xl border border-slate-800 bg-slate-900 p-5 space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-xl font-bold text-white">
                    {selectedStock.name || selectedStock.symbol}{" "}
                    <span className="text-sm font-mono font-normal text-slate-400">({selectedStock.symbol})</span>
                  </h2>
                  <span
                    className={`text-xs font-bold px-2 py-0.5 rounded border ${
                      GRADE_STYLES[selectedStock.grade] || GRADE_STYLES.C
                    }`}
                  >
                    {selectedStock.side} · GRADE {selectedStock.grade}
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  {selectedStock.sector} · Win Prob {selectedStock.win_prob_pct}% · RSI(14){" "}
                  {selectedStock.rsi_14} · Vol {selectedStock.vol_ratio_20}x
                </p>
              </div>
              <button
                onClick={() => setSelectedStock(null)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <PriceChart
              candles={stockCandles}
              entry={selectedStock.entry}
              target={selectedStock.target}
              stopLoss={selectedStock.stop_loss}
              side={selectedStock.side}
              recDate={scan?.scan_date}
            />

            <div className="grid grid-cols-3 gap-3 text-xs bg-slate-950/70 rounded-xl p-3 border border-slate-800">
              <div>
                <span className="text-slate-400">Entry</span>
                <div className="text-sm font-mono font-bold text-sky-400">${selectedStock.entry}</div>
              </div>
              <div>
                <span className="text-slate-400">
                  Target ({config?.target_atr_mult}x ATR · +{selectedStock.target_gain_pct}%)
                </span>
                <div className="text-sm font-mono font-bold text-emerald-400">${selectedStock.target}</div>
              </div>
              <div>
                <span className="text-slate-400">Stop Loss ({config?.stop_atr_mult}x ATR)</span>
                <div className="text-sm font-mono font-bold text-rose-400">${selectedStock.stop_loss}</div>
              </div>
            </div>
            {stockEntryTiming && stockEntryTiming.status === "OK" && (
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-semibold text-slate-400">1H / 4H Entry Timing:</span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${
                      stockEntryTiming.signal === "ENTER_NOW"
                        ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                        : stockEntryTiming.signal === "NEUTRAL_LIMIT"
                          ? "bg-sky-500/20 text-sky-300 border-sky-500/40"
                          : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                    }`}
                  >
                    {stockEntryTiming.signal}
                  </span>
                  <span className="text-slate-300">{stockEntryTiming.reason}</span>
                </div>
                <div className="font-mono text-[11px] text-slate-300">
                  Suggested Limit: <span className="text-sky-300 font-bold">${stockEntryTiming.suggested_entry}</span> · 1h RSI {stockEntryTiming.rsi_14_1h} · 4h RSI {stockEntryTiming.rsi_14_4h}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {showSettings && config && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <form
            onSubmit={handleSaveConfig}
            className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-5 space-y-4"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold text-white">Triple-Barrier &amp; Universe Settings</h2>
              <button
                type="button"
                onClick={() => setShowSettings(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Target ATR Mult</label>
                <input
                  type="number"
                  step="0.1"
                  value={config.target_atr_mult}
                  onChange={(e) => setConfig({ ...config, target_atr_mult: Number(e.target.value) })}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-white"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Stop ATR Mult</label>
                <input
                  type="number"
                  step="0.1"
                  value={config.stop_atr_mult}
                  onChange={(e) => setConfig({ ...config, stop_atr_mult: Number(e.target.value) })}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-white"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Horizon (Trading Days)</label>
                <input
                  type="number"
                  value={config.horizon_days}
                  onChange={(e) => setConfig({ ...config, horizon_days: Number(e.target.value) })}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-white"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Min Dollar Vol ($M)</label>
                <input
                  type="number"
                  value={config.min_avg_dollar_vol_m}
                  onChange={(e) => setConfig({ ...config, min_avg_dollar_vol_m: Number(e.target.value) })}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-white"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Universe Limit (max 600)</label>
                <input
                  type="number"
                  value={config.universe_limit}
                  onChange={(e) => setConfig({ ...config, universe_limit: Number(e.target.value) })}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-white"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Top K Picks</label>
                <input
                  type="number"
                  value={config.top_k}
                  onChange={(e) => setConfig({ ...config, top_k: Number(e.target.value) })}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-white"
                />
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setShowSettings(false)}
                className="px-3 py-1.5 rounded-lg text-xs bg-slate-800 text-slate-300 cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-sky-600 hover:bg-sky-500 text-white cursor-pointer"
              >
                Save Settings
              </button>
            </div>
          </form>
        </div>
      )}

      {trackedTradeChart && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-2xl border border-slate-800 bg-slate-900 p-5 space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2.5">
                  <h2 className="text-xl font-bold text-white">
                    {trackedTradeChart.trade.name || trackedTradeChart.trade.symbol}{" "}
                    <span className="text-sm font-mono font-normal text-slate-400">
                      ({trackedTradeChart.trade.symbol})
                    </span>
                  </h2>
                  <span className="text-xs font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-200 border border-slate-700">
                    {trackedTradeChart.trade.side} · {trackedTradeChart.trade.status}
                  </span>
                  <span className="text-xs font-mono text-slate-400">
                    Rec: {trackedTradeChart.trade.rec_date}
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  30D Pre-Entry Context + Daily EOD Progression ({trackedTradeChart.trade.days_elapsed}/
                  {trackedTradeChart.trade.horizon_days} days) · P&amp;L:{" "}
                  <span
                    className={
                      Number(trackedTradeChart.trade.pnl_pct) >= 0 ? "text-emerald-400 font-semibold" : "text-rose-400 font-semibold"
                    }
                  >
                    {Number(trackedTradeChart.trade.pnl_pct) >= 0 ? "+" : ""}
                    {trackedTradeChart.trade.pnl_pct}%
                  </span>
                </p>
              </div>
              <button
                onClick={() => setTrackedTradeChart(null)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <PriceChart
              candles={trackedTradeChart.candles}
              entry={trackedTradeChart.trade.entry_price}
              target={trackedTradeChart.trade.target_price}
              stopLoss={trackedTradeChart.trade.stop_loss}
              side={trackedTradeChart.trade.side}
              recDate={trackedTradeChart.trade.rec_date}
            />

            <div className="grid grid-cols-4 gap-3 text-xs bg-slate-950/70 rounded-xl p-3 border border-slate-800">
              <div>
                <span className="text-slate-400">Entry</span>
                <div className="text-sm font-mono font-bold text-sky-400">${trackedTradeChart.trade.entry_price}</div>
              </div>
              <div>
                <span className="text-slate-400">Target (+{trackedTradeChart.trade.target_gain_pct}%)</span>
                <div className="text-sm font-mono font-bold text-emerald-400">${trackedTradeChart.trade.target_price}</div>
              </div>
              <div>
                <span className="text-slate-400">Stop Loss</span>
                <div className="text-sm font-mono font-bold text-rose-400">${trackedTradeChart.trade.stop_loss}</div>
              </div>
              <div>
                <span className="text-slate-400">Latest / Exit</span>
                <div className="text-sm font-mono font-bold text-white">
                  ${trackedTradeChart.trade.exit_price || trackedTradeChart.trade.last_eod_price}
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/60 overflow-hidden">
              <div className="px-3.5 py-2 border-b border-slate-800 text-xs font-semibold text-slate-300">
                Subsequent Daily EOD Prices
              </div>
              <div className="max-h-48 overflow-y-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="px-3 py-2">Date</th>
                      <th className="px-3 py-2 text-right">Open</th>
                      <th className="px-3 py-2 text-right">High</th>
                      <th className="px-3 py-2 text-right">Low</th>
                      <th className="px-3 py-2 text-right">Close (EOD)</th>
                      <th className="px-3 py-2 text-right">vs Entry</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {(trackedTradeChart.post_days || []).length === 0 ? (
                      <tr>
                        <td colSpan={6} className="px-3 py-4 text-center text-slate-500">
                          Recommended on the latest available bar — subsequent EOD bars will appear here as new days are synced.
                        </td>
                      </tr>
                    ) : (
                      trackedTradeChart.post_days.map((d) => (
                        <tr key={d.date}>
                          <td className="px-3 py-1.5 font-mono text-slate-300">{d.date}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-slate-400">${d.open}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-slate-400">${d.high}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-slate-400">${d.low}</td>
                          <td className="px-3 py-1.5 text-right font-mono font-semibold text-white">${d.close}</td>
                          <td
                            className={`px-3 py-1.5 text-right font-mono font-semibold ${
                              d.pnl_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            {d.pnl_pct >= 0 ? "+" : ""}
                            {d.pnl_pct}%
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
