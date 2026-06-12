#!/usr/bin/env python
"""Run V2 baseline_strategy_v2 and write metrics as the V2 regression golden.

The golden file records per-coin Sharpe and Return so that
tests/regression/test_v2_unchanged.py can detect drift introduced by V3 work.

Usage:
    python scripts/regenerate_v2_golden.py
    python scripts/regenerate_v2_golden.py --pred-dir data/multi_2coins_v2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "tests" / "regression" / "fixtures" / "v2_88bar_metrics.json"
DEFAULT_PRED_DIR = REPO_ROOT / "data" / "multi_2coins_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate V2 regression golden fixture.")
    p.add_argument(
        "--pred-dir",
        default=str(DEFAULT_PRED_DIR),
        help="Directory with preds_lgb_h*.csv files (default: data/multi_2coins_v2)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)

    for h in (7, 14):
        csv = pred_dir / f"preds_lgb_h{h}.csv"
        if not csv.exists():
            print(f"ERROR: Missing {csv}")
            print("  Run scripts/evaluate_models_multi.py first to generate predictions.")
            sys.exit(1)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "baseline_strategy_v2.py"),
        "--pred-dir", str(pred_dir),
        "--symmetric",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"ERROR: baseline_strategy_v2.py exited {result.returncode}")
        sys.exit(result.returncode)

    # baseline_strategy_v2 writes report_v2/metrics.json inside pred_dir
    src = pred_dir / "report_v2" / "metrics.json"
    if not src.exists():
        print(f"WARNING: Expected {src} but it doesn't exist.")
        print("  The script may not have written it for this data directory.")
        sys.exit(1)

    with open(src) as f:
        raw = json.load(f)

    # Build golden in canonical per_coin / portfolio_avg structure
    per_coin: dict = {}
    for coin_key, metrics in raw.items():
        if not isinstance(metrics, dict) or "sharpe_ratio" not in metrics:
            continue
        per_coin[coin_key] = {
            "total_return": metrics["total_return"],
            "annualized_return": metrics.get("annualized_return"),
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics.get("win_rate"),
            "n_trades": metrics.get("n_trades"),
            "profit_factor": metrics.get("profit_factor"),
            "halted": metrics.get("halted", False),
        }

    # Compute simple average of per-coin sharpe + return for portfolio_avg
    sharpes = [v["sharpe_ratio"] for v in per_coin.values() if v["sharpe_ratio"] is not None]
    returns = [v["total_return"] for v in per_coin.values() if v["total_return"] is not None]
    portfolio_avg = {
        "sharpe_ratio": sum(sharpes) / len(sharpes) if sharpes else None,
        "total_return": sum(returns) / len(returns) if returns else None,
    }

    import subprocess as _sub
    try:
        git_sha = _sub.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    golden = {
        "_generated_from": (
            f"{src.relative_to(REPO_ROOT)} recorded at commit {git_sha}. "
            "Regenerate via scripts/regenerate_v2_golden.py when pred CSVs change."
        ),
        "per_coin": per_coin,
        "portfolio_avg": portfolio_avg,
    }

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"\nWrote golden to {GOLDEN_PATH}")
    for coin, m in per_coin.items():
        print(f"  {coin}: Sharpe={m['sharpe_ratio']:.4f}  Return={m['total_return']:+.2%}")
    print(f"  Portfolio avg: Sharpe={portfolio_avg['sharpe_ratio']:.4f}  Return={portfolio_avg['total_return']:+.2%}")


if __name__ == "__main__":
    main()
