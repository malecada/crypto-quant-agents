import type {
  CycleDetail, CycleRow, HealthResp, PerformanceResp, PositionsResp,
  Strategy, TradesResp,
} from "./types";

/** Thin fetch wrapper. Browser basic-auth (401 challenge) covers credentials. */
async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json() as Promise<T>;
}

export const api = {
  performance: () => get<PerformanceResp>("/api/performance"),
  positions: () => get<PositionsResp>("/api/positions"),
  trades: (s: Strategy) => get<TradesResp>(`/api/trades?strategy=${s}`),
  cycles: (s: Strategy) => get<{ cycles: CycleRow[] }>(`/api/cycles?strategy=${s}`),
  cycle: (id: string, s: Strategy) =>
    get<CycleDetail>(`/api/cycle/${encodeURIComponent(id)}?strategy=${s}`),
  health: () => get<HealthResp>("/api/health"),
};
