import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Badge } from "../components/Badge";
import { Section } from "../components/Section";
import { fmtNum } from "../lib/format";
import type { Strategy } from "../types";

function Tbl(props: { rows: Record<string, unknown>[]; cols: string[] }) {
  if (!props.rows.length) return <p className="muted">none</p>;
  return (
    <table>
      <thead><tr>{props.cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>
        {props.rows.map((r, i) => (
          <tr key={i}>{props.cols.map((c) => <td key={c}>{String(r[c] ?? "—")}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

export function DecisionsTab() {
  const [strategy, setStrategy] = useState<Strategy>("quant");
  const cyclesQ = useQuery({
    queryKey: ["cycles", strategy],
    queryFn: () => api.cycles(strategy),
  });
  const cycles = cyclesQ.data?.cycles ?? [];
  const [cycleId, setCycleId] = useState<string | null>(null);
  const selected = cycleId ?? cycles[0]?.cycle_id ?? null;
  const detailQ = useQuery({
    queryKey: ["cycle", selected, strategy],
    queryFn: () => api.cycle(selected!, strategy),
    enabled: selected !== null,
  });
  if (cyclesQ.isLoading) return <div className="muted">loading…</div>;
  if (cyclesQ.isError) return <div className="badge error">failed: {String(cyclesQ.error)}</div>;
  return (
    <>
      <div className="pills">
        {(["quant", "hybrid"] as Strategy[]).map((s) => (
          <button key={s} className={`pill ${s === strategy ? "active" : ""}`}
            onClick={() => { setStrategy(s); setCycleId(null); }}>{s}</button>
        ))}
        <select value={selected ?? ""} onChange={(e) => setCycleId(e.target.value)}
          style={{ background: "#161b22", color: "#e6edf3", border: "1px solid #30363d",
                   borderRadius: 6, padding: "4px 8px" }}>
          {cycles.map((c) => (
            <option key={c.cycle_id} value={c.cycle_id}>
              {c.cycle_id} · {c.status ?? "?"}
            </option>
          ))}
        </select>
      </div>
      {detailQ.data && (
        <>
          {strategy === "hybrid" && (
            <Section title="LLM modulator">
              {detailQ.data.modulator.length ? (
                <table>
                  <thead><tr>
                    <th>Coin</th><th>Multiplier</th><th>Effective weight</th>
                    <th>Confidence</th><th>Regime</th><th>Mode</th>
                  </tr></thead>
                  <tbody>
                    {detailQ.data.modulator.map((m) => (
                      <tr key={m.coin}>
                        <td>{m.coin}</td>
                        <td>{fmtNum(m.multiplier)}</td>
                        <td>{fmtNum(m.effective_weight)}</td>
                        <td>{m.llm_confidence === null ? "—" : fmtNum(m.llm_confidence)}</td>
                        <td>{m.regime ?? "—"}</td>
                        <td>{m.fallback
                          ? <Badge kind="stale">pure quant fallback</Badge>
                          : <Badge kind="ok">modulated</Badge>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <p className="muted">no modulator rows (cycle predates modulator journaling)</p>}
            </Section>
          )}
          <Section title="Predictions">
            <Tbl rows={detailQ.data.predictions}
              cols={["coin", "horizon", "pred_value", "ref_price", "signal_h7",
                     "signal_h14", "consensus_signal", "bundle_route"]} />
          </Section>
          <Section title="Sizing">
            <Tbl rows={detailQ.data.sizing}
              cols={["coin", "realized_vol", "kelly", "confidence", "base_size",
                     "leverage", "sma30_multiplier", "final_size_notional"]} />
          </Section>
          <Section title="Risk checks">
            <Tbl rows={detailQ.data.risk_checks}
              cols={["coin", "check_name", "passed", "value", "threshold", "reason"]} />
          </Section>
          <Section title="Shadow decisions">
            <Tbl rows={detailQ.data.shadow_decisions}
              cols={["coin", "live_signal", "backtest_signal", "agree",
                     "live_size", "backtest_size", "size_delta_pct"]} />
          </Section>
        </>
      )}
    </>
  );
}
