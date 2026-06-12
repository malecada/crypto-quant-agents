import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = ["#58a6ff", "#bc8cff", "#3fb950", "#d29922", "#f85149",
  "#39c5cf", "#db61a2", "#9e6a03", "#6e7681"];

export function AllocationDonut(props: { data: { label: string; usd: number }[] }) {
  const total = props.data.reduce((s, d) => s + d.usd, 0);
  if (!props.data.length || total === 0)
    return <p className="muted">no allocation data</p>;
  return (
    <ResponsiveContainer width="100%" height={240}>
      <PieChart>
        <Pie data={props.data} dataKey="usd" nameKey="label"
          innerRadius={55} outerRadius={90} stroke="#161b22">
          {props.data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
        </Pie>
        <Tooltip formatter={(v) =>
          [`$${Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 })} (${(Number(v) / total * 100).toFixed(1)}%)`]}
          contentStyle={{ background: "#161b22", border: "1px solid #30363d" }} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
