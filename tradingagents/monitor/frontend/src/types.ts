export interface Point { ts: string; value: number }

export interface Cards {
  equity: number; sharpe: number; max_drawdown: number;
  total_upnl: number | null; upnl_stale: boolean;
  open_positions: number | null;
}

export interface StrategyPerf {
  cards: Cards; equity: Point[]; drawdown: Point[]; rolling_sharpe: Point[];
}

export interface CompareSide {
  sharpe: number | null; ret: number | null; maxdd: number | null;
}

export interface CompareBlock {
  quant?: CompareSide;
  hybrid?: CompareSide;
  delta?: CompareSide;
  window?: { start: string; end: string; n: number };
  error?: string;
}

export interface PerformanceResp {
  quant: StrategyPerf | null; hybrid: StrategyPerf | null;
  compare: CompareBlock | null;
  anchors: { quant: number; hybrid: number | null };
}

export interface Position {
  coin: string; side: "LONG" | "SHORT"; qty: number;
  entry: number | null; mark: number | null; leverage: number | null;
  notional: number | null; upnl_usd: number | null; upnl_pct: number | null;
  liq_price: number | null;
}

export interface StrategyPositions {
  positions: Position[];
  totals: { upnl: number | null; notional: number | null; equity: number | null };
  allocation: { label: string; usd: number }[];
  stale: boolean; as_of: string | null; error: string | null;
}

export interface PositionsResp {
  quant: StrategyPositions | null; hybrid: StrategyPositions | null;
}

export interface IncomeSummary {
  realized_pnl_per_coin: Record<string, number>;
  realized_pnl_total: number; fees_total: number; funding_total: number;
  win_rate: number | null; n_closing_fills: number;
}

export interface TradesResp {
  executions: Record<string, unknown>[];
  analytics: {
    income: IncomeSummary | null;
    slippage: { mean: number | null; max: number | null; n: number };
  };
}

export interface CycleRow {
  cycle_id: string; start_ts: string; end_ts: string | null; status: string | null;
  error_msg: string | null; n_trades: number | null;
  critical_data_fail_sources: string | null;
  supplementary_stale_sources: string | null;
}

export interface ModulatorRow {
  cycle_id: string; coin: string; multiplier: number; effective_weight: number;
  llm_confidence: number | null; regime: string | null; fallback: number;
}

export interface CycleDetail {
  predictions: Record<string, unknown>[];
  sizing: Record<string, unknown>[];
  risk_checks: Record<string, unknown>[];
  shadow_decisions: Record<string, unknown>[];
  modulator: ModulatorRow[];
}

export interface HealthResp {
  timeline: { quant: CycleRow[] | null; hybrid: CycleRow[] | null };
  steps: Record<string, unknown>[];
  errors: Record<string, unknown>[];
  retrains: { quant: Record<string, unknown>[] | null; hybrid: Record<string, unknown>[] | null };
}

export type Strategy = "quant" | "hybrid";
