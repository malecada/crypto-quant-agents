import type { ReactNode } from "react";

export function Section(props: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>{props.title}</h2>
        {props.right}
      </div>
      {props.children}
    </div>
  );
}
