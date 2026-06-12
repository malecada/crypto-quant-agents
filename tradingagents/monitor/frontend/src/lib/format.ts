export function fmtUsd(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 2,
  });
}

/** v is a FRACTION (-0.05 => "-5.00%"). */
export function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(2)}%`;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined) return "—";
  return v.toFixed(digits);
}
