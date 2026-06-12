import { useEffect, useState } from "react";
import { PerformanceTab } from "./tabs/PerformanceTab";
import { PositionsTab } from "./tabs/PositionsTab";
import { ExecutionsTab } from "./tabs/ExecutionsTab";
import { DecisionsTab } from "./tabs/DecisionsTab";
import { HealthTab } from "./tabs/HealthTab";

const TABS = [
  { id: "performance", label: "Performance", el: <PerformanceTab /> },
  { id: "positions", label: "Positions", el: <PositionsTab /> },
  { id: "executions", label: "Executions", el: <ExecutionsTab /> },
  { id: "decisions", label: "Decisions", el: <DecisionsTab /> },
  { id: "health", label: "Health", el: <HealthTab /> },
] as const;

export default function App() {
  const initial = window.location.hash.replace("#", "") || "performance";
  const [tab, setTab] = useState(initial);
  useEffect(() => {
    const onHash = () => setTab(window.location.hash.replace("#", "") || "performance");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const active = TABS.find((t) => t.id === tab) ?? TABS[0];
  return (
    <>
      <div className="topbar">
        <h1>Live Monitor</h1>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={`tab ${t.id === active.id ? "active" : ""}`}
              onClick={() => { window.location.hash = t.id; }}>
              {t.label}
            </button>
          ))}
        </nav>
      </div>
      <div className="container">{active.el}</div>
    </>
  );
}
