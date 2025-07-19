#!/usr/bin/env python3
"""
Benchmark suite for CodeConductor - collects comprehensive data for analysis.

Runs multiple experiments and saves results for dashboard analysis and reporting.
"""

import click
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import subprocess
import sys
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.prompt_optimizer import prompt_optimizer


def run_experiment(
    prompt_path: Path, iters: int, online: bool = False
) -> Dict[str, Any]:
    """
    Run a single experiment and collect metrics.

    Args:
        prompt_path: Path to prompt file
        iters: Number of iterations
        online: Whether to use LM Studio

    Returns:
        Dictionary with experiment results
    """
    print(f"🧪 Running experiment: {prompt_path.name} ({iters} iterations)")

    # Run pipeline
    cmd = [
        sys.executable,
        "pipeline.py",
        "--prompt",
        str(prompt_path),
        "--iters",
        str(iters),
    ]

    if online:
        cmd.append("--online")
    else:
        cmd.append("--mock")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse results from database
    import sqlite3

    db_path = Path("data/metrics.db")

    if not db_path.exists():
        return {"error": "No database found"}

    conn = sqlite3.connect(db_path)

    # Get latest experiment data
    df = pd.read_sql(
        """
        SELECT * FROM metrics 
        WHERE iteration >= (SELECT MAX(iteration) FROM metrics) - ?
        ORDER BY iteration
        """,
        conn,
        params=(iters - 1,),
    )

    conn.close()

    if df.empty:
        return {"error": "No data found"}

    # Calculate metrics
    metrics = {
        "prompt": prompt_path.name,
        "iterations": len(df),
        "avg_reward": df["reward"].mean(),
        "max_reward": df["reward"].max(),
        "min_reward": df["reward"].min(),
        "pass_rate": df["pass_rate"].mean(),
        "blocked_count": len(df[df["blocked"] == 1]),
        "safe_count": len(df[df["blocked"] == 0]),
        "avg_complexity": df["complexity"].mean(),
        "model_source": df["model_source"].mode().iloc[0]
        if not df["model_source"].empty
        else "mock",
        "optimizer_actions": df["optimizer_action"].value_counts().to_dict()
        if "optimizer_action" in df.columns
        else {},
        "convergence": df["reward"].iloc[-5:].mean()
        - df["reward"].iloc[:5].mean(),  # Improvement over time
        "timestamp": datetime.now().isoformat(),
    }

    return metrics


def run_benchmark_suite(
    prompt_dir: str, iters: int, online: bool = False, save_q_table: bool = True
):
    """
    Run comprehensive benchmark suite.

    Args:
        prompt_dir: Directory containing prompt files
        iters: Number of iterations per experiment
        online: Whether to use LM Studio
        save_q_table: Whether to save Q-table snapshots
    """
    prompt_path = Path(prompt_dir)

    if not prompt_path.exists():
        print(f"❌ Prompt directory not found: {prompt_dir}")
        return

    # Find all prompt files
    prompt_files = list(prompt_path.glob("*.md"))

    if not prompt_files:
        print(f"❌ No prompt files found in {prompt_dir}")
        return

    print(f"🚀 Starting benchmark suite with {len(prompt_files)} prompts")
    print(f"📊 {iters} iterations per experiment")
    print(f"🤖 Online mode: {online}")

    # Create results directory
    results_dir = Path("bench/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []

    for i, prompt_file in enumerate(prompt_files, 1):
        print(f"\n--- Experiment {i}/{len(prompt_files)} ---")

        # Save Q-table snapshot every 10 experiments
        if save_q_table and i % 10 == 0:
            q_table_path = results_dir / f"q_table_snapshot_{i}.json"
            prompt_optimizer.save_q_table(str(q_table_path))
            print(f"💾 Q-table snapshot saved: {q_table_path}")

        # Run experiment
        result = run_experiment(prompt_file, iters, online)
        result["experiment_id"] = i
        all_results.append(result)

        # Print summary
        if "error" not in result:
            print(
                f"✅ {result['prompt']}: avg_reward={result['avg_reward']:.2f}, "
                f"pass_rate={result['pass_rate']:.2f}, blocked={result['blocked_count']}"
            )
        else:
            print(f"❌ {result['error']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    json_path = results_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save as CSV
    df_results = pd.DataFrame(all_results)
    csv_path = results_dir / f"benchmark_results_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)

    # Save Q-table final state
    if save_q_table:
        q_table_final = results_dir / f"q_table_final_{timestamp}.json"
        prompt_optimizer.save_q_table(str(q_table_final))

        # Save Q-table summary
        q_summary = prompt_optimizer.get_q_table_summary()
        q_summary_path = results_dir / f"q_table_summary_{timestamp}.json"
        with open(q_summary_path, "w") as f:
            json.dump(q_summary, f, indent=2)

    # Print final summary
    print(f"\n🎉 Benchmark suite completed!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"📊 JSON: {json_path}")
    print(f"📈 CSV: {csv_path}")

    if save_q_table:
        print(f"🧠 Q-table: {q_table_final}")
        print(f"📋 Q-summary: {q_summary_path}")

    # Print aggregate metrics
    if all_results and "error" not in all_results[0]:
        df_summary = pd.DataFrame(all_results)
        print(f"\n📊 Aggregate Results:")
        print(f"   Total experiments: {len(df_summary)}")
        print(f"   Avg reward across all: {df_summary['avg_reward'].mean():.2f}")
        print(f"   Avg pass rate: {df_summary['pass_rate'].mean():.2f}")
        print(f"   Total blocked code: {df_summary['blocked_count'].sum()}")
        print(f"   Avg convergence: {df_summary['convergence'].mean():.2f}")


@click.command()
@click.option(
    "--prompt_dir", default="prompts", help="Directory containing prompt files"
)
@click.option("--iters", default=50, help="Number of iterations per experiment")
@click.option("--online", is_flag=True, help="Use LM Studio instead of mock")
@click.option(
    "--save_q_table", is_flag=True, default=True, help="Save Q-table snapshots"
)
def main(prompt_dir, iters, online, save_q_table):
    """Run CodeConductor benchmark suite."""
    run_benchmark_suite(prompt_dir, iters, online, save_q_table)


if __name__ == "__main__":
    main()
