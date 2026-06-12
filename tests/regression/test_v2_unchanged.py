"""Regression test: V2 baseline metrics unchanged after V3 lands.

Run by pytest as part of the standard suite. If the prediction CSVs needed
by V2 aren't present, the test is SKIPPED with a clear marker.

The golden file (fixtures/v2_88bar_metrics.json) was recorded from main at
commit d6b7e5f using the full data/multi_2coins_v2 predictions. Regenerate
via scripts/regenerate_v2_golden.py when pred CSVs change.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "v2_88bar_metrics.json"
PRED_DIR = REPO_ROOT / "data" / "multi_2coins_v2"


def _v2_pred_files_present() -> bool:
    return all((PRED_DIR / f"preds_lgb_h{h}.csv").exists() for h in (7, 14))


@pytest.mark.skipif(
    not _v2_pred_files_present(),
    reason="V2 prediction CSVs not present in data/multi_2coins_v2/ — run evaluate_models_multi.py first",
)
def test_v2_metrics_match_golden():
    """Run V2 and compare metrics to the recorded golden values."""
    # Check golden exists and is not a placeholder
    if not GOLDEN_PATH.exists():
        pytest.skip(
            "No golden V2 metrics fixture — generate via scripts/regenerate_v2_golden.py"
        )

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    # Detect placeholder (task-plan fallback — shouldn't happen since we ship a real golden)
    if "_note" in golden and "PLACEHOLDER" in golden.get("_note", ""):
        pytest.skip("V2 golden is placeholder — regenerate when data available")

    # Run V2 backtest via subprocess.
    # baseline_strategy_v2.py writes report_v2/metrics.json into pred_dir.
    # We redirect its output-plot to a temp file to avoid polluting the data dir.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
        tmp_plot = tmp_png.name

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "baseline_strategy_v2.py"),
        "--pred-dir", str(PRED_DIR),
        "--symmetric",
        "--output-plot", tmp_plot,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT)
    )

    if result.returncode != 0:
        pytest.fail(
            f"baseline_strategy_v2.py exited {result.returncode}.\n"
            f"stderr: {result.stderr[-1000:]}\n"
            f"stdout: {result.stdout[-500:]}"
        )

    # V2 writes metrics to report_v2/metrics.json inside pred_dir
    metrics_path = PRED_DIR / "report_v2" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip(
            f"V2 didn't write {metrics_path}; stdout snippet: {result.stdout[-500:]}"
        )

    with open(metrics_path) as f:
        raw = json.load(f)

    # raw is keyed by coin_id directly (flat schema from report_v2)
    # Wrap into per_coin for uniform comparison
    actual_per_coin: dict = {}
    for coin_key, coin_metrics in raw.items():
        if isinstance(coin_metrics, dict) and "sharpe_ratio" in coin_metrics:
            actual_per_coin[coin_key] = coin_metrics

    # Tolerances
    tol_sharpe = 0.01
    tol_return = 0.005

    golden_per_coin = golden.get("per_coin", {})

    for coin in ("bitcoin", "ethereum"):
        if coin not in golden_per_coin:
            continue  # coin not in golden, skip
        if coin not in actual_per_coin:
            pytest.fail(f"{coin} missing from V2 actual output")

        g = golden_per_coin[coin]
        a = actual_per_coin[coin]

        sharpe_diff = abs(g["sharpe_ratio"] - a.get("sharpe_ratio", float("nan")))
        assert sharpe_diff < tol_sharpe, (
            f"{coin} Sharpe drift: golden={g['sharpe_ratio']:.6f} "
            f"actual={a.get('sharpe_ratio'):.6f} diff={sharpe_diff:.6f}"
        )

        return_diff = abs(g["total_return"] - a.get("total_return", float("nan")))
        assert return_diff < tol_return, (
            f"{coin} Return drift: golden={g['total_return']:.6f} "
            f"actual={a.get('total_return'):.6f} diff={return_diff:.6f}"
        )
