"""Generate the two figures embedded in PROGRESS_REPORT.md.

Builds, over the canonical 4.5-year walk-forward window (2021-11 -> 2026-04):
  fig 1 -- cumulative growth of $1: V5 MIX (4-coin) vs equal-weight buy & hold
  fig 2 -- annualised Sharpe: V5 MIX (4-coin) vs equal-weight buy & hold

Strategy returns come from the V5 MIX portfolio daily-return series; the
buy-&-hold baseline is reconstructed from the per-coin `ref_price` (spot price
at each prediction date) in the routing dirs, so both series share one window.

The source `data/` tree is gitignored; this script documents how the committed
PNGs in `report_assets/` were produced. Re-run with the TradingAgents data dir
available to regenerate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else
            "../TradingAgents/data").resolve()
OUT = Path(__file__).resolve().parent.parent / "report_assets"
OUT.mkdir(exist_ok=True)

COINS = ["bitcoin", "ethereum", "binancecoin", "solana"]
ROUTING = {
    "bitcoin": "multi_2coins_walkforward",
    "ethereum": "multi_2coins_pit_wf",
    "binancecoin": "multi_3coins_bnb_wf",
    "solana": "multi_3coins_sol_pit_wf",
}
RF, BARS_PER_YR = 0.0, 252.0   # 252 to match the published canonical Sharpe
NAVY, GREY = "#1b3a6b", "#9aa3ad"


def sharpe(daily: pd.Series) -> float:
    d = daily.dropna()
    return float(np.sqrt(BARS_PER_YR) * d.mean() / d.std(ddof=1))


# --- strategy: V5 MIX 4-coin portfolio daily returns ---
strat = pd.read_csv(DATA / "tmp_v5_4coin_check/daily_returns.csv",
                    index_col=0, parse_dates=True)
strat_ret = strat["portfolio"]

# --- buy & hold: equal-weight basket from per-coin ref_price ---
price = {}
for coin in COINS:
    df = pd.read_csv(DATA / ROUTING[coin] / "preds_lgb_h14.csv",
                     parse_dates=["date"])
    s = (df[df["coin_id"] == coin]
         .set_index("date")["ref_price"].sort_index())
    s.index = s.index.tz_localize(None)   # prediction dates are tz-aware UTC
    price[coin] = s[~s.index.duplicated()]
px = pd.DataFrame(price).reindex(strat_ret.index).ffill().dropna()
bh_ret = (px.pct_change().fillna(0.0)).mean(axis=1)   # daily EW basket return
bh_ret = bh_ret.reindex(strat_ret.index).fillna(0.0)

strat_cum = (1 + strat_ret).cumprod()
bh_cum = (1 + bh_ret).cumprod()

# --- fig 1: cumulative growth ---
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(strat_cum.index, strat_cum.values, color=NAVY, lw=2.0,
        label=f"V5 MIX (4-coin)  ·  Sharpe {sharpe(strat_ret):.2f}")
ax.plot(bh_cum.index, bh_cum.values, color=GREY, lw=1.6, ls="--",
        label=f"Equal-weight buy & hold  ·  Sharpe {sharpe(bh_ret):.2f}")
ax.set_yscale("log")
ax.set_ylabel("Growth of $1 (log scale)")
ax.set_title("V5 MIX vs buy & hold — 4.5-year walk-forward")
ax.legend(frameon=False, loc="upper left")
ax.grid(True, which="both", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "cumulative_vs_buyhold.png", dpi=130)

# --- fig 2: Sharpe bars ---
labels = ["Buy & hold\n(equal weight)", "V5 MIX\n(4-coin)"]
vals = [sharpe(bh_ret), sharpe(strat_ret)]
fig, ax = plt.subplots(figsize=(5.2, 4.4))
bars = ax.bar(labels, vals, color=[GREY, NAVY], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Annualised Sharpe ratio")
ax.set_title("Risk-adjusted return — same 4.5-year window")
ax.grid(True, axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "sharpe_vs_buyhold.png", dpi=130)

print(f"window {strat_ret.index[0].date()} -> {strat_ret.index[-1].date()}  "
      f"bars={len(strat_ret)}")
print(f"V5 MIX   Sharpe {sharpe(strat_ret):.3f}  total {strat_cum.iloc[-1]-1:+.1%}")
print(f"buy&hold Sharpe {sharpe(bh_ret):.3f}  total {bh_cum.iloc[-1]-1:+.1%}")
print(f"wrote {OUT}/cumulative_vs_buyhold.png, {OUT}/sharpe_vs_buyhold.png")
