import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Badge } from "../components/Badge";
import { Section } from "../components/Section";
import type { CycleRow } from "../types";

function Timeline(props: { name: "quant" | "hybrid"; rows: CycleRow[] }) {
  return (
    <Section title="">
      <Badge kind={props.name}>{props.name.toUpperCase()}</Badge>
      <table style={{ marginTop: 8 }}>
        <thead><tr><th>Cycle</th><th>Status</th><th>Trades</th><th>Data fails</th><th>Error</th></tr></thead>
        <tbody>
          {props.rows.slice(0, 30).map((c) => (
            <tr key={c.cycle_id}>
              <td>{c.cycle_id}</td>
              <td><Badge kind={c.status === "ok" ? "ok" : "error"}>{c.status ?? "?"}</Badge></td>
              <td>{c.n_trades ?? "—"}</td>
              <td className="muted">{c.critical_data_fail_sources || c.supplementary_stale_sources || "—"}</td>
              <td className="muted">{c.error_msg || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

export function HealthTab() {
  const q = useQuery({ queryKey: ["health"], queryFn: api.health });
  if (q.isLoading) return <div className="muted">loading…</div>;
  if (q.isError || !q.data) return <div className="badge error">failed: {String(q.error)}</div>;
  const d = q.data;
  return (
    <>
      <div className="grid2">
        {d.timeline.quant
          ? <Timeline name="quant" rows={d.timeline.quant} />
          : <div className="panel muted">quant journal unavailable</div>}
        {d.timeline.hybrid
          ? <Timeline name="hybrid" rows={d.timeline.hybrid} />
          : <div className="panel muted">hybrid not configured</div>}
      </div>
      <Section title="Pipeline steps (latest quant cycle — hybrid runner has no structured log)">
        <table>
          <thead><tr><th>Step</th><th>Status</th><th>Duration</th><th>Detail</th></tr></thead>
          <tbody>
            {d.steps.map((s, i) => (
              <tr key={i}>
                <td>{String(s.step ?? "")}</td>
                <td><Badge kind={s.status === "ok" ? "ok" : "error"}>{String(s.status ?? "")}</Badge></td>
                <td>{s.duration_ms != null ? `${String(s.duration_ms)} ms` : "—"}</td>
                <td className="muted">{JSON.stringify(s.payload ?? {})}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
      <Section title="Retrains">
        <table>
          <thead><tr><th>Strategy</th><th>Retrain</th><th>Cycle</th><th>DirAcc</th><th>Status</th></tr></thead>
          <tbody>
            {(["quant", "hybrid"] as const).flatMap((name) =>
              (d.retrains[name] ?? []).map((r, i) => (
                <tr key={`${name}${i}`}>
                  <td><Badge kind={name}>{name}</Badge></td>
                  <td>{String(r.retrain_id ?? "")}</td>
                  <td>{String(r.cycle_id ?? "")}</td>
                  <td>{String(r.train_dir_acc ?? "—")}</td>
                  <td>{String(r.status ?? "")}</td>
                </tr>
              )))}
          </tbody>
        </table>
      </Section>
    </>
  );
}
