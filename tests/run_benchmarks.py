#!/usr/bin/env python3
import argparse
import asyncio
import json
from datetime import datetime

from master_integration_test_simple import SimpleMasterTest


async def run_quick_bench():
    suite = SimpleMasterTest(argparse.Namespace(self_reflection=False))
    try:
        await suite.initialize()
        # Only run performance benchmarks part to keep it light
        result = await suite.test_performance_benchmarks()
        metrics = suite.performance_results
    finally:
        await suite.cleanup()
    return bool(result), metrics


def main():
    parser = argparse.ArgumentParser(description="Quick benchmark runner")
    parser.add_argument("--agent-count", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ok, metrics = asyncio.run(run_quick_bench())
    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ok": ok,
        "metrics": metrics,
        "agent_count": args.agent_count,
        "mode": "quick" if args.quick else "full",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
