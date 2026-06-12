import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { Section } from "../components/Section";
import { AllocationDonut } from "../charts/AllocationDonut";
import { fmtNum, fmtUsd } from "../lib/format";
import type { StrategyPositions } from "../types";

function StrategyBlock(props: { name: "quant" | "hybrid"; s: StrategyPositions }) {
  const { s } = props;
  return (
    <Section title=""
      right={s.stale
        ? <Badge kind="stale">STALE — live unavailable{s.as_of ? ` · as of ${s.as_of}` : ""}</Badge>
        : <Badge kind="ok">live</Badge>}>
      <div style={{ marginBottom: 8 }}>
        <Badge kind={props.name}>{props.name.toUpperCase()}</Badge>
        {s.error && <span className="muted" style={{ marginLeft: 8 }}>{s.error}</span>}
      </div>
      <div className="cards">
        <Card label="Account equity" value={fmtUsd(s.totals.equity)} />
        <Card label="Total uPnL" value={fmtUsd(s.totals.upnl)}
          tone={(s.totals.upnl ?? 0) >= 0 ? "pos" : "neg"} />
        <Card label="Gross notional" value={fmtUsd(s.totals.notional)} />
      </div>
      <table style={{ marginTop: 10 }}>
        <thead><tr>
          <th>Coin</th><th>Side</th><th>Qty</th><th>Entry</th><th>Mark</th>
          <th>Lev</th><th>Notional</th><th>uPnL $</th><th>uPnL %</th><th>Liq</th>
        </tr></thead>
        <tbody>
          {s.positions.map((p) => (
            <tr key={p.coin}>
              <td>{p.coin}</td>
              <td className={p.side === "LONG" ? "pos" : "neg"}>{p.side}</td>
              <td>{p.qty}</td>
              <td>{fmtUsd(p.entry)}</td>
              <td>{fmtUsd(p.mark)}</td>
              <td>{p.leverage ?? "—"}</td>
              <td>{fmtUsd(p.notional)}</td>
              <td className={(p.upnl_usd ?? 0) >= 0 ? "pos" : "neg"}>{fmtUsd(p.upnl_usd)}</td>
              <td className={(p.upnl_pct ?? 0) >= 0 ? "pos" : "neg"}>
                {p.upnl_pct === null ? "—" : `${fmtNum(p.upnl_pct)}%`}</td>
              <td>{fmtUsd(p.liq_price)}</td>
            </tr>
          ))}
          {!s.positions.length && <tr><td colSpan={10} className="muted">flat — no open positions</td></tr>}
        </tbody>
      </table>
      <h2>Allocation</h2>
      <AllocationDonut data={s.allocation} />
    </Section>
  );
}

export function PositionsTab() {
  const q = useQuery({ queryKey: ["positions"], queryFn: api.positions });
  if (q.isLoading) return <div className="muted">loading…</div>;
  if (q.isError || !q.data) return <div className="badge error">failed: {String(q.error)}</div>;
  return (
    <div className="grid2">
      {q.data.quant
        ? <StrategyBlock name="quant" s={q.data.quant} />
        : <div className="panel muted">quant journal unavailable</div>}
      {q.data.hybrid
        ? <StrategyBlock name="hybrid" s={q.data.hybrid} />
        : <div className="panel muted">hybrid not configured</div>}
    </div>
  );
}
