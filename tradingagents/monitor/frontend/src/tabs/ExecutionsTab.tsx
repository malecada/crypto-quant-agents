import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { Section } from "../components/Section";
import { fmtNum, fmtUsd } from "../lib/format";
import type { Strategy } from "../types";

export function ExecutionsTab() {
  const [strategy, setStrategy] = useState<Strategy>("quant");
  const q = useQuery({
    queryKey: ["trades", strategy],
    queryFn: () => api.trades(strategy),
  });
  if (q.isLoading) return <div className="muted">loading…</div>;
  if (q.isError || !q.data) return <div className="badge error">failed: {String(q.error)}</div>;
  const { executions, analytics } = q.data;
  const inc = analytics.income;
  return (
    <>
      <div className="pills">
        {(["quant", "hybrid"] as Strategy[]).map((s) => (
          <button key={s} className={`pill ${s === strategy ? "active" : ""}`}
            onClick={() => setStrategy(s)}>{s}</button>
        ))}
      </div>

      <Section title="Trade analytics">
        {inc ? (
          <div className="cards">
            <Card label="Realized PnL" value={fmtUsd(inc.realized_pnl_total)}
              tone={inc.realized_pnl_total >= 0 ? "pos" : "neg"} />
            <Card label="Win rate"
              value={inc.win_rate === null ? "—" : `${(inc.win_rate * 100).toFixed(0)}% of ${inc.n_closing_fills}`} />
            <Card label="Fees" value={fmtUsd(inc.fees_total)} />
            <Card label="Funding" value={fmtUsd(inc.funding_total)} />
            <Card label="Slippage mean/max"
              value={`${fmtNum(analytics.slippage.mean)} / ${fmtNum(analytics.slippage.max)}`} />
          </div>
        ) : (
          <p className="muted">
            income analytics unavailable (exchange income API unreachable) —
            slippage mean/max: {fmtNum(analytics.slippage.mean)} / {fmtNum(analytics.slippage.max)} over {analytics.slippage.n} fills
          </p>
        )}
        {inc && Object.keys(inc.realized_pnl_per_coin).length > 0 && (
          <table style={{ marginTop: 10 }}>
            <thead><tr><th>Symbol</th><th>Realized PnL</th></tr></thead>
            <tbody>
              {Object.entries(inc.realized_pnl_per_coin).map(([sym, v]) => (
                <tr key={sym}><td>{sym}</td>
                  <td className={v >= 0 ? "pos" : "neg"}>{fmtUsd(v)}</td></tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted" style={{ marginTop: 6 }}>
          income figures cover the last 1000 exchange income records
        </p>
      </Section>

      <Section title={`Executions (${executions.length})`}>
        <table>
          <thead><tr>
            <th>Cycle</th><th>Coin</th><th>Side</th><th>Qty</th>
            <th>Entry</th><th>Slippage</th><th>Status</th>
          </tr></thead>
          <tbody>
            {executions.map((t, i) => (
              <tr key={i}>
                <td>{String(t.cycle_id ?? "")}</td>
                <td>{String(t.coin ?? "")}</td>
                <td className={t.side === "BUY" ? "pos" : "neg"}>{String(t.side ?? "")}</td>
                <td>{String(t.qty ?? "")}</td>
                <td>{fmtUsd(t.entry_price as number | null)}</td>
                <td>{t.slippage === null || t.slippage === undefined ? "—" : String(t.slippage)}</td>
                <td><Badge kind={t.status === "EXECUTED" ? "ok" : "error"}>{String(t.status ?? "")}</Badge></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    </>
  );
}
