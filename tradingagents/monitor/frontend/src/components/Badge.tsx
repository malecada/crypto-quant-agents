import type { ReactNode } from "react";

export function Badge(props: { kind: "quant" | "hybrid" | "stale" | "error" | "ok"; children: ReactNode }) {
  return <span className={`badge ${props.kind}`}>{props.children}</span>;
}
