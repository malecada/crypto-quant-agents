import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Card } from "../components/Card";
import { Badge } from "../components/Badge";
import { Section } from "../components/Section";
import { EquityChart } from "../charts/EquityChart";
import { fmtNum, fmtPct, fmtUsd } from "../lib/format";
import { rebaseTo100, sliceFromDays } from "../lib/rebase";
import type { StrategyPerf } from "../types";

const RANGES = [
  { label: "7d", days: 7 }, { label: "30d", days: 30 },
  { label: "90d", days: 90 }, { label: "all", days: null },
] as const;

function prep(p: StrategyPerf | null, days: number | null) {
  return {
    eq: p ? rebaseTo100(sliceFromDays(p.equity, days)) : [],
    dd: p ? sliceFromDays(p.drawdown, days) : [],
    rs: p ? sliceFromDays(p.rolling_sharpe, days) : [],
  };
}

function CardsRow(props: { name: "quant" | "hybrid"; p: StrategyPerf }) {
  const c = props.p.cards;
  return (
    <div style={{ marginTop: 10 }}>
      <Badge kind={props.name}>{props.name.toUpperCase()}</Badge>{" "}
      {c.upnl_stale && <Badge kind="stale">uPnL stale</Badge>}
      <div className="cards" style={{ marginTop: 6 }}>
        <Card label="Equity" value={fmtUsd(c.equity)} />
        <Card label="Sharpe (live)" value={fmtNum(c.sharpe)} tone={c.sharpe >= 0 ? "pos" : "neg"} />
        <Card label="Max drawdown" value={fmtPct(c.max_drawdown)} tone="neg" />
        <Card label="Unrealized PnL" value={fmtUsd(c.total_upnl)}
          tone={(c.total_upnl ?? 0) >= 0 ? "pos" : "neg"} />
        <Card label="Open positions" value={c.open_positions === null ? "—" : String(c.open_positions)} />
      </div>
    </div>
  );
}

export function PerformanceTab() {
  const q = useQuery({ queryKey: ["performance"], queryFn: api.performance });
  const [days, setDays] = useState<number | null>(null);
  const d = q.data;
  const quant = useMemo(
    () => prep(d?.quant ?? null, days),
    [d, days],
  );
  const hybrid = useMemo(
    () => prep(d?.hybrid ?? null, days),
    [d, days],
  );
  if (q.isLoading) return <div className="muted">loading…</div>;
  if (q.isError || !d) return <div className="badge error">failed: {String(q.error)}</div>;

  return (
    <>
      {d.quant ? <CardsRow name="quant" p={d.quant} />
        : <p className="muted">quant journal unavailable</p>}
      {d.hybrid ? <CardsRow name="hybrid" p={d.hybrid} />
        : <p className="muted">hybrid not configured</p>}

      <Section title="Equity (indexed to 100) · drawdown · rolling Sharpe"
        right={
          <div className="pills">
            {RANGES.map((r) => (
              <button key={r.label} className={`pill ${days === r.days ? "active" : ""}`}
                onClick={() => setDays(r.days)}>{r.label}</button>
            ))}
          </div>
        }>
        <EquityChart
          quantEquity={quant.eq} hybridEquity={hybrid.eq}
          quantDd={quant.dd} hybridDd={hybrid.dd}
          quantRs={quant.rs} hybridRs={hybrid.rs}
          anchors={d.anchors}
        />
        {(d.quant?.rolling_sharpe.length ?? 0) === 0 &&
          <p className="muted">rolling Sharpe appears after 30 live cycles</p>}
      </Section>

      {d.compare && !d.compare.error && (
        <Section title={`Quant vs hybrid — common window ${d.compare.window?.start} → ${d.compare.window?.end} (${d.compare.window?.n} cycles)`}>
          <table>
            <thead><tr><th></th><th>Sharpe</th><th>Return</th><th>Max DD</th></tr></thead>
            <tbody>
              <tr><td><Badge kind="quant">QUANT</Badge></td>
                <td>{fmtNum(d.compare.quant?.sharpe)}</td>
                <td>{fmtPct(d.compare.quant?.ret)}</td>
                <td>{fmtPct(d.compare.quant?.maxdd)}</td></tr>
              <tr><td><Badge kind="hybrid">HYBRID</Badge></td>
                <td>{fmtNum(d.compare.hybrid?.sharpe)}</td>
                <td>{fmtPct(d.compare.hybrid?.ret)}</td>
                <td>{fmtPct(d.compare.hybrid?.maxdd)}</td></tr>
              <tr><td className="muted">Δ (H−Q)</td>
                <td>{fmtNum(d.compare.delta?.sharpe)}</td>
                <td>{fmtPct(d.compare.delta?.ret)}</td>
                <td>{fmtPct(d.compare.delta?.maxdd)}</td></tr>
            </tbody>
          </table>
        </Section>
      )}
      {d.compare?.error && <p className="muted">compare: {d.compare.error}</p>}
    </>
  );
}
