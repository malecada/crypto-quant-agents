export interface Point { ts: string; value: number }

/** Rebase a series so its first point equals 100. */
export function rebaseTo100(points: Point[]): Point[] {
  if (points.length === 0) return [];
  const base = points[0].value;
  if (base === 0) return points.map((p) => ({ ...p, value: 0 }));
  return points.map((p) => ({ ts: p.ts, value: (p.value / base) * 100 }));
}

/** Keep points within `days` of the LAST point's timestamp (null = all).
 *  Points with unparseable timestamps (e.g. the journal's synthetic
 *  "start" row) are dropped so chart time scales never see NaN. */
export function sliceFromDays(points: Point[], days: number | null): Point[] {
  const parseable = points.filter((p) => !Number.isNaN(new Date(p.ts).getTime()));
  if (days === null || parseable.length === 0) return parseable;
  const end = new Date(parseable[parseable.length - 1].ts).getTime();
  const cutoff = end - days * 86_400_000;
  return parseable.filter((p) => new Date(p.ts).getTime() >= cutoff);
}
