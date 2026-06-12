#!/usr/bin/env python
"""Re-score the `confidence` column of existing agent-signal CSVs.

The trader prompt often omits an explicit HIGH/MEDIUM/LOW label, so the
original extract_confidence() returned UNKNOWN for the majority of rows.
SignalProcessor.extract_confidence was updated to *infer* confidence from
the conviction strength of the trader text. This script replays that new
logic over existing CSVs so we avoid re-running the full (4.7 h) signal
generation pass.

Usage:
    python scripts/rescore_confidence.py \\
        --input data/agent_signals_pit \\
        --output data/agent_signals_pit_rescored \\
        --llm-provider openai \\
        --quick-think gpt-4o-mini

The input CSVs must have columns: date, signal, confidence, trader_text.
Output CSVs have the same schema, with `confidence` overwritten.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-score confidence column on existing agent-signal CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Directory with per-coin signal CSVs.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output directory for rescored CSVs.")
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--quick-think", default="gpt-4o-mini")
    p.add_argument("--replay-cache", action="store_true", default=True,
                   help="Use LLM replay cache (deterministic, caches across runs).")
    p.add_argument("--glob", default="*.csv",
                   help="Glob pattern to match input CSVs.")
    return p.parse_args()


def build_signal_processor(llm_provider: str, quick_think: str, replay_cache: bool):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.llm_clients import create_llm_client
    from tradingagents.graph.signal_processing import SignalProcessor

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = llm_provider
    config["quick_think_llm"] = quick_think
    config["replay_cache"] = replay_cache

    client = create_llm_client(
        provider=llm_provider,
        model=quick_think,
        base_url=config.get("backend_url"),
    )
    quick = client.get_llm()

    if replay_cache:
        from tradingagents.llm_clients.replay_cache import CachedChatModel
        cache_db = config.get("replay_cache_db", "./data/llm_replay_cache.db")
        quick = CachedChatModel(quick, db_path=cache_db)

    return SignalProcessor(quick)


def rescore_file(csv_path: Path, out_path: Path, processor) -> dict:
    df = pd.read_csv(csv_path)
    required = {"date", "signal", "confidence", "trader_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    new_conf: list[str] = []
    for i, text in enumerate(df["trader_text"].fillna("").tolist()):
        label = processor.extract_confidence(text) if text else "UNKNOWN"
        new_conf.append(label)
        if (i + 1) % 10 == 0:
            logging.info("  %d/%d rescored", i + 1, len(df))

    before = df["confidence"].value_counts().to_dict()
    df["confidence"] = new_conf
    after = df["confidence"].value_counts().to_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return {"file": csv_path.name, "before": before, "after": after, "n": len(df)}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    t0 = time.time()

    if not args.input.is_dir():
        raise SystemExit(f"input dir not found: {args.input}")

    processor = build_signal_processor(
        args.llm_provider, args.quick_think, args.replay_cache
    )

    files = sorted(args.input.glob(args.glob))
    if not files:
        raise SystemExit(f"no CSVs matched {args.input}/{args.glob}")

    print(f"\n{'=' * 60}")
    print("  Confidence re-scoring")
    print(f"{'=' * 60}")
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.output}")
    print(f"  LLM        : {args.quick_think} ({args.llm_provider})")
    print(f"  Cache      : {args.replay_cache}")
    print(f"  Files      : {len(files)}")
    print()

    summaries = []
    for f in files:
        out_path = args.output / f.name
        logging.info("Rescoring %s -> %s", f.name, out_path)
        summaries.append(rescore_file(f, out_path, processor))

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for s in summaries:
        print(f"  {s['file']} ({s['n']} rows)")
        print(f"    before: {s['before']}")
        print(f"    after : {s['after']}")
    print(f"\n  Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
