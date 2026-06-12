export function Card(props: { label: string; value: string; tone?: "pos" | "neg" | "" }) {
  return (
    <div className="card">
      <div className="label">{props.label}</div>
      <div className={`value ${props.tone ?? ""}`}>{props.value}</div>
    </div>
  );
}
