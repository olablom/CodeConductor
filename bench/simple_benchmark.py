#!/usr/bin/env python3
"""
Simple Performance Benchmark for CodeConductor

Basic performance measurement focusing on key operations.
"""

import time
import statistics
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_database_operations():
    """Benchmark database read/write operations"""

    print("📊 Benchmarking Database Operations...")

    try:
        import sqlite3

        # Test Q-table database
        qtable_times = []
        if Path("data/qtable.db").exists():
            for i in range(10):
                start_time = time.perf_counter()
                conn = sqlite3.connect("data/qtable.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM q_table")
                count = cursor.fetchone()[0]
                conn.close()
                end_time = time.perf_counter()
                qtable_times.append((end_time - start_time) * 1000)

            print(f"  Q-table read: {statistics.mean(qtable_times):.2f}ms avg")

        # Test metrics database
        metrics_times = []
        if Path("data/metrics.db").exists():
            for i in range(10):
                start_time = time.perf_counter()
                conn = sqlite3.connect("data/metrics.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM metrics")
                count = cursor.fetchone()[0]
                conn.close()
                end_time = time.perf_counter()
                metrics_times.append((end_time - start_time) * 1000)

            print(f"  Metrics read: {statistics.mean(metrics_times):.2f}ms avg")

        return {
            "qtable_read_avg": statistics.mean(qtable_times) if qtable_times else 0,
            "metrics_read_avg": statistics.mean(metrics_times) if metrics_times else 0,
        }

    except Exception as e:
        print(f"  ❌ Database benchmark failed: {e}")
        return {}


def benchmark_import_times():
    """Benchmark import times for key modules"""

    print("📦 Benchmarking Import Times...")

    modules = [
        "agents.base_agent",
        "agents.codegen_agent",
        "agents.architect_agent",
        "agents.review_agent",
        "agents.policy_agent",
        "agents.prompt_optimizer",
        "agents.orchestrator",
    ]

    import_times = {}

    for module in modules:
        try:
            start_time = time.perf_counter()
            __import__(module)
            end_time = time.perf_counter()
            import_time = (end_time - start_time) * 1000
            import_times[module] = import_time
            print(f"  {module}: {import_time:.2f}ms")
        except Exception as e:
            print(f"  {module}: ❌ Failed - {e}")
            import_times[module] = -1

    return import_times


def benchmark_memory_usage():
    """Benchmark memory usage patterns"""

    print("🧠 Benchmarking Memory Usage...")

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate some operations
        test_data = []
        for i in range(1000):
            test_data.append({"id": i, "data": "test" * 100})

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        del test_data

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Peak memory: {peak_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Memory increase: {peak_memory - initial_memory:.2f}MB")

        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": peak_memory - initial_memory,
        }

    except ImportError:
        print("  ❌ psutil not available, skipping memory benchmark")
        return {}
    except Exception as e:
        print(f"  ❌ Memory benchmark failed: {e}")
        return {}


def benchmark_file_operations():
    """Benchmark file I/O operations"""

    print("📁 Benchmarking File Operations...")

    # Test JSON read/write
    json_times = []
    test_data = {"test": "data", "numbers": list(range(1000))}

    for i in range(10):
        start_time = time.perf_counter()

        # Write
        with open("bench/test_temp.json", "w") as f:
            json.dump(test_data, f)

        # Read
        with open("bench/test_temp.json", "r") as f:
            loaded_data = json.load(f)

        end_time = time.perf_counter()
        json_times.append((end_time - start_time) * 1000)

    # Cleanup
    if Path("bench/test_temp.json").exists():
        Path("bench/test_temp.json").unlink()

    avg_json_time = statistics.mean(json_times)
    print(f"  JSON read/write: {avg_json_time:.2f}ms avg")

    return {"json_io_avg_ms": avg_json_time}


def run_comprehensive_benchmark():
    """Run all benchmarks and generate report"""

    print("🎯 Starting Comprehensive Performance Benchmark...")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "database": benchmark_database_operations(),
        "imports": benchmark_import_times(),
        "memory": benchmark_memory_usage(),
        "file_ops": benchmark_file_operations(),
    }

    # Calculate overall performance score
    scores = []

    # Database score (lower is better)
    if results["database"]:
        db_score = 100 - min(results["database"].get("qtable_read_avg", 0), 100)
        scores.append(db_score)

    # Import score (lower is better)
    if results["imports"]:
        import_score = 100 - min(
            statistics.mean([t for t in results["imports"].values() if t > 0]), 100
        )
        scores.append(import_score)

    # Memory score (lower increase is better)
    if results["memory"]:
        memory_score = 100 - min(
            results["memory"].get("memory_increase_mb", 0) * 10, 100
        )
        scores.append(memory_score)

    # File ops score (lower is better)
    if results["file_ops"]:
        file_score = 100 - min(results["file_ops"].get("json_io_avg_ms", 0), 100)
        scores.append(file_score)

    if scores:
        overall_score = statistics.mean(scores)
        results["overall_score"] = overall_score
    else:
        results["overall_score"] = 0

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bench/results/benchmark_{timestamp}.json"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Overall Performance Score: {results['overall_score']:.1f}/100")

    if results["database"]:
        print(
            f"Database Performance: {results['database'].get('qtable_read_avg', 0):.2f}ms avg"
        )

    if results["imports"]:
        avg_import = statistics.mean([t for t in results["imports"].values() if t > 0])
        print(f"Import Performance: {avg_import:.2f}ms avg")

    if results["memory"]:
        print(
            f"Memory Efficiency: {results['memory'].get('memory_increase_mb', 0):.2f}MB increase"
        )

    if results["file_ops"]:
        print(
            f"File I/O Performance: {results['file_ops'].get('json_io_avg_ms', 0):.2f}ms avg"
        )

    print(f"\n✅ Results saved to {filename}")

    return results


def main():
    """Main benchmark function"""

    results = run_comprehensive_benchmark()

    print("\n🎉 Benchmark completed!")
    print("Use these results to identify optimization opportunities.")


if __name__ == "__main__":
    main()
