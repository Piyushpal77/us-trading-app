import React from "react";

export default function EquityChart({ curve }) {
  if (!curve || curve.length < 2) {
    return (
      <div className="h-56 flex items-center justify-center text-slate-400 text-sm">
        Not enough out-of-sample history for equity curve.
      </div>
    );
  }

  const W = 720;
  const H = 220;
  const padL = 48;
  const padR = 20;
  const padT = 16;
  const padB = 26;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const vals = curve.flatMap((d) => [d.strategy, d.benchmark]);
  const minV = Math.min(...vals) * 0.97;
  const maxV = Math.max(...vals) * 1.03;
  const rangeV = maxV - minV || 1;

  const xFor = (i) => padL + (i / (curve.length - 1)) * plotW;
  const yFor = (v) => padT + ((maxV - v) / rangeV) * plotH;

  const stratPath = curve
    .map((d, i) => `${i === 0 ? "M" : "L"}${xFor(i).toFixed(1)},${yFor(d.strategy).toFixed(1)}`)
    .join(" ");
  const benchPath = curve
    .map((d, i) => `${i === 0 ? "M" : "L"}${xFor(i).toFixed(1)},${yFor(d.benchmark).toFixed(1)}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-56 select-none">
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
        const y = padT + t * plotH;
        const v = maxV - t * rangeV;
        return (
          <g key={i}>
            <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
            <text x={padL - 8} y={y + 3} textAnchor="end" fontSize="10" fill="#64748b">
              {v.toFixed(0)}
            </text>
          </g>
        );
      })}
      <path d={benchPath} fill="none" stroke="#64748b" strokeWidth="1.6" strokeDasharray="4 4" />
      <path d={stratPath} fill="none" stroke="#38bdf8" strokeWidth="2.2" />

      {[0, Math.floor(curve.length / 2), curve.length - 1].map((idx) => (
        <text key={idx} x={xFor(idx)} y={H - 6} textAnchor="middle" fontSize="10" fill="#64748b">
          {curve[idx].date}
        </text>
      ))}
    </svg>
  );
}
