import React from "react";

export default function PriceChart({ candles, entry, target, stopLoss, side, recDate }) {
  if (!candles || candles.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-400 text-sm">
        No price history available.
      </div>
    );
  }

  const W = 720;
  const H = 260;
  const padL = 54;
  const padR = 78;
  const padT = 18;
  const padB = 26;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const prices = candles.flatMap((c) => [c.high, c.low]);
  if (target) prices.push(Number(target));
  if (stopLoss) prices.push(Number(stopLoss));
  if (entry) prices.push(Number(entry));

  const minP = Math.min(...prices) * 0.99;
  const maxP = Math.max(...prices) * 1.01;
  const rangeP = maxP - minP || 1;

  const xFor = (i) => padL + (i / Math.max(1, candles.length - 1)) * plotW;
  const yFor = (p) => padT + ((maxP - p) / rangeP) * plotH;

  const pathD = candles
    .map((c, i) => `${i === 0 ? "M" : "L"}${xFor(i).toFixed(1)},${yFor(c.close).toFixed(1)}`)
    .join(" ");

  const areaD = `${pathD} L${xFor(candles.length - 1).toFixed(1)},${(padT + plotH).toFixed(1)} L${padL},${(padT + plotH).toFixed(1)} Z`;

  const recIdx = recDate ? candles.findIndex((c) => c.date === recDate || c.is_rec_date) : -1;
  const recX = recIdx >= 0 ? xFor(recIdx) : null;
  const recY = recIdx >= 0 ? yFor(candles[recIdx].close) : null;
  const recLabelNearRight = recX !== null && recX > W - padR - 110;

  const levels = [
    target ? { label: `TP $${target}`, y: yFor(Number(target)), color: "#10b981" } : null,
    entry ? { label: `Entry $${entry}`, y: yFor(Number(entry)), color: "#38bdf8" } : null,
    stopLoss ? { label: `SL $${stopLoss}`, y: yFor(Number(stopLoss)), color: "#f43f5e" } : null,
  ].filter(Boolean);

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-64 select-none">
        <defs>
          <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={side === "SHORT" ? "#f43f5e" : "#10b981"} stopOpacity="0.28" />
            <stop offset="100%" stopColor={side === "SHORT" ? "#f43f5e" : "#10b981"} stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {recX !== null && (
          <rect
            x={recX}
            y={padT}
            width={Math.max(0, W - padR - recX)}
            height={plotH}
            fill="#38bdf8"
            fillOpacity="0.06"
          />
        )}

        {[0, 0.25, 0.5, 0.75, 1].map((t, idx) => {
          const y = padT + t * plotH;
          const p = maxP - t * rangeP;
          return (
            <g key={idx}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
              <text x={padL - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#64748b">
                {p.toFixed(0)}
              </text>
            </g>
          );
        })}

        <path d={areaD} fill="url(#priceFill)" />
        <path
          d={pathD}
          fill="none"
          stroke={side === "SHORT" ? "#fb7185" : "#34d399"}
          strokeWidth="2"
        />

        {recX !== null && (
          <g>
            <line
              x1={recX}
              y1={padT}
              x2={recX}
              y2={padT + plotH}
              stroke="#fbbf24"
              strokeWidth="2"
              strokeDasharray="2 4"
            />
            {recY !== null && (
              <circle cx={recX} cy={recY} r="4" fill="#fbbf24" stroke="#020617" strokeWidth="1.5" />
            )}
            <text
              x={recLabelNearRight ? recX - 6 : recX + 6}
              y={padT + 12}
              textAnchor={recLabelNearRight ? "end" : "start"}
              fontSize="10"
              fontWeight="700"
              fill="#fbbf24"
            >
              Rec: {recDate}
            </text>
          </g>
        )}

        {levels.map((lv, idx) => (
          <g key={idx}>
            <line
              x1={padL}
              y1={lv.y}
              x2={W - padR}
              y2={lv.y}
              stroke={lv.color}
              strokeWidth="1.2"
              strokeDasharray="5 4"
            />
            <text x={W - padR + 6} y={lv.y + 3} fontSize="10" fontWeight="600" fill={lv.color}>
              {lv.label}
            </text>
          </g>
        ))}

        {[0, Math.floor(candles.length / 2), candles.length - 1].map((idx) => {
          const c = candles[idx];
          if (!c) return null;
          return (
            <text key={idx} x={xFor(idx)} y={H - 6} textAnchor="middle" fontSize="10" fill="#64748b">
              {c.date}
            </text>
          );
        })}
      </svg>
    </div>
  );
}
