import { useEffect, useRef } from "react";
import {
  createChart, ColorType, LineSeries, AreaSeries, LineStyle,
  type IChartApi, type Time,
} from "lightweight-charts";
import type { Point } from "../types";

export interface EquityChartProps {
  quantEquity: Point[]; hybridEquity: Point[];     // already sliced+rebased
  quantDd: Point[]; hybridDd: Point[];             // already sliced (fractions)
  quantRs: Point[]; hybridRs: Point[];             // rolling sharpe (may be [])
  anchors: { quant: number; hybrid: number | null };
}

const toLw = (pts: Point[]) =>
  pts.map((p) => ({ time: (new Date(p.ts).getTime() / 1000) as Time, value: p.value }));

export function EquityChart(props: EquityChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const showRs = props.quantRs.length > 0 || props.hybridRs.length > 0;
    const chart = createChart(ref.current, {
      height: showRs ? 520 : 420,
      layout: {
        background: { type: ColorType.Solid, color: "#161b22" },
        textColor: "#8b949e",
        panes: { separatorColor: "#30363d" },
      },
      grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
      timeScale: { borderColor: "#30363d", timeVisible: false },
      rightPriceScale: { borderColor: "#30363d" },
    });
    chartRef.current = chart;

    chart.addSeries(LineSeries, { color: "#3fb950", lineWidth: 2, title: "quant" }, 0)
      .setData(toLw(props.quantEquity));
    if (props.hybridEquity.length)
      chart.addSeries(LineSeries, { color: "#bc8cff", lineWidth: 2, title: "hybrid" }, 0)
        .setData(toLw(props.hybridEquity));

    const ddOpts = { lineWidth: 1 as const, priceFormat: { type: "percent" as const } };
    chart.addSeries(AreaSeries, {
      ...ddOpts, lineColor: "#3fb950", topColor: "rgba(63,185,80,0)",
      bottomColor: "rgba(63,185,80,0.25)", title: "quant DD",
    }, 1).setData(toLw(props.quantDd.map((p) => ({ ...p, value: p.value * 100 }))));
    if (props.hybridDd.length)
      chart.addSeries(AreaSeries, {
        ...ddOpts, lineColor: "#bc8cff", topColor: "rgba(188,140,255,0)",
        bottomColor: "rgba(188,140,255,0.25)", title: "hybrid DD",
      }, 1).setData(toLw(props.hybridDd.map((p) => ({ ...p, value: p.value * 100 }))));

    if (showRs) {
      const rsQuant = chart.addSeries(LineSeries,
        { color: "#3fb950", lineWidth: 1, title: "quant rSR" }, 2);
      rsQuant.setData(toLw(props.quantRs));
      rsQuant.createPriceLine({
        price: props.anchors.quant, color: "#8b949e",
        lineStyle: LineStyle.Dashed, title: `backtest ${props.anchors.quant}`,
      });
      if (props.hybridRs.length) {
        const rsH = chart.addSeries(LineSeries,
          { color: "#bc8cff", lineWidth: 1, title: "hybrid rSR" }, 2);
        rsH.setData(toLw(props.hybridRs));
        if (props.anchors.hybrid !== null)
          rsH.createPriceLine({
            price: props.anchors.hybrid, color: "#8b949e",
            lineStyle: LineStyle.Dashed, title: `backtest ${props.anchors.hybrid}`,
          });
      }
    }

    chart.timeScale().fitContent();
    const onResize = () => chart.applyOptions({ width: ref.current?.clientWidth ?? 600 });
    onResize();
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); chart.remove(); };
  }, [props.quantEquity, props.hybridEquity, props.quantDd, props.hybridDd, props.quantRs, props.hybridRs, props.anchors]);

  return <div ref={ref} />;
}
