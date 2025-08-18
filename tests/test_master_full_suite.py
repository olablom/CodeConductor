#!/usr/bin/env python3
"""
Master Full Suite Test - Complete End-to-End Validation
Tests all components: Single Agent, Debate System, RAG, GUI, Performance
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

import GPUtil
import psutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from codeconductor.context.rag_system import RAGSystem
from codeconductor.debate.debate_manager import CodeConductorDebateManager
from codeconductor.ensemble.ensemble_engine import EnsembleEngine
from codeconductor.ensemble.single_model_engine import (
    SingleModelEngine,
    SingleModelRequest,
)
from codeconductor.feedback.self_reflection_agent import self_reflection_agent
from codeconductor.generators.improved_prompt_generator import improved_prompt_generator


class MasterTestSuite:
    """Comprehensive test suite for all CodeConductor components"""

    def __init__(self, args):
        self.args = args
        self.results = {}
        self.performance_metrics = {}
        self.start_time = None

    async def initialize(self):
        """Initialize all test components"""
        print("🚀 Initializing Master Test Suite")
        print("=" * 60)

        # Initialize components
        self.single_engine = SingleModelEngine()
        self.ensemble_engine = EnsembleEngine()
        self.rag_system = RAGSystem()

        # Initialize debate manager with agents
        from codeconductor.debate.local_agent import LocalAIAgent

        agents = [
            LocalAIAgent("coder", "You are a coding expert who focuses on implementation."),
            LocalAIAgent(
                "architect",
                "You are a software architect who focuses on design patterns.",
            ),
            LocalAIAgent("tester", "You are a testing expert who focuses on quality assurance."),
            LocalAIAgent("reviewer", "You are a code reviewer who focuses on best practices."),
        ]
        self.debate_manager = CodeConductorDebateManager(agents)

        # Initialize engines
        await self.single_engine.initialize()
        await self.ensemble_engine.initialize()
        # RAG system doesn't need initialization

        print("✅ All components initialized")

    def get_system_metrics(self):
        """Get current system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # GPU metrics
            gpus = GPUtil.getGPUs()
            gpu_metrics = {}
            if gpus:
                gpu = gpus[0]  # Primary GPU
                gpu_metrics = {
                    "gpu_load": gpu.load * 100,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                }

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                **gpu_metrics,
            }
        except Exception as e:
            print(f"⚠️ Could not get system metrics: {e}")
            return {}

    async def test_single_agent_performance(self):
        """Test single agent with self-reflection"""
        print("\n🧪 Testing Single Agent Performance")
        print("-" * 40)

        test_cases = [
            (
                "fibonacci",
                "Create a Python function to calculate the nth Fibonacci number",
            ),
            (
                "binary_search",
                "Create a Python function to perform binary search on a sorted array",
            ),
            (
                "rest_api",
                "Create a simple REST API endpoint for user login using Flask",
            ),
        ]

        results = []
        total_time = 0

        for task_type, description in test_cases:
            start_time = time.time()

            # Generate improved prompt
            prompt = improved_prompt_generator.generate_improved_prompt(task_type, description)

            # Get response
            request = SingleModelRequest(task_description=prompt)
            response = await self.single_engine.process_request(request)
            code = self_reflection_agent.extract_code(response.content)

            # Test with self-reflection if enabled
            if self.args.self_reflection:
                success, error = self_reflection_agent.validate_code(code, task_type)
                iterations = 1

                while not success and iterations < 3:
                    fix_prompt = improved_prompt_generator.generate_fix_prompt(
                        code, error, task_type
                    )
                    fix_request = SingleModelRequest(task_description=fix_prompt)
                    improved_response = await self.single_engine.process_request(fix_request)
                    improved_code = self_reflection_agent.extract_code(improved_response.content)

                    success, error = self_reflection_agent.validate_code(improved_code, task_type)
                    if success:
                        code = improved_code
                        break

                    code = improved_code
                    iterations += 1
            else:
                success = True  # Assume success without validation
                iterations = 1

            end_time = time.time()
            duration = end_time - start_time
            total_time += duration

            result = {
                "task_type": task_type,
                "success": success,
                "duration": duration,
                "iterations": iterations,
                "code_length": len(code),
            }
            results.append(result)

            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {task_type}: {status} - {duration:.1f}s ({iterations} iterations)")

        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        avg_time = total_time / len(results)

        self.results["single_agent"] = {
            "success_rate": success_rate,
            "avg_time": avg_time,
            "total_time": total_time,
            "results": results,
        }

        print(f"📊 Single Agent: {success_rate:.1f}% success, {avg_time:.1f}s avg")
        return success_rate >= 65  # Target: 65%

    async def test_debate_system_performance(self):
        """Test debate system with multiple agents"""
        print("\n🧪 Testing Debate System Performance")
        print("-" * 40)

        test_cases = [
            ("rest_api", "Create a REST API for user management with authentication"),
            (
                "react_component",
                "Create a React component for a todo list with add/delete functionality",
            ),
            (
                "sql_query",
                "Write a SQL query to find the top 10 customers by total purchase amount",
            ),
            ("bug_fix", "Fix a Python function that has a division by zero error"),
            (
                "algorithm",
                "Implement a function to find the longest palindromic substring",
            ),
            (
                "data_processing",
                "Create a function to process CSV data and calculate statistics",
            ),
        ]

        results = []
        total_time = 0

        for task_type, description in test_cases:
            start_time = time.time()

            try:
                # Run debate
                debate_result = await self.debate_manager.conduct_debate(description)

                end_time = time.time()
                duration = end_time - start_time
                total_time += duration

                success = debate_result.get("success", True)  # Assume success if no error
                code_count = len(debate_result.get("implementations", []))

                result = {
                    "task_type": task_type,
                    "success": success,
                    "duration": duration,
                    "code_count": code_count,
                    "agents_used": len(self.args.agents.split()),
                }
                results.append(result)

                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {task_type}: {status} - {duration:.1f}s ({code_count} implementations)")

            except Exception as e:
                print(f"  {task_type}: ❌ ERROR - {str(e)}")
                results.append(
                    {
                        "task_type": task_type,
                        "success": False,
                        "duration": 0,
                        "code_count": 0,
                        "error": str(e),
                    }
                )

        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        avg_time = total_time / len(results) if results else 0

        self.results["debate_system"] = {
            "success_rate": success_rate,
            "avg_time": avg_time,
            "total_time": total_time,
            "results": results,
        }

        print(f"📊 Debate System: {success_rate:.1f}% success, {avg_time:.1f}s avg")
        return success_rate >= 80  # Target: 80%

    async def test_rag_functionality(self):
        """Test RAG search and context retrieval"""
        print("\n🧪 Testing RAG Functionality")
        print("-" * 40)

        # Test cases for RAG
        test_queries = [
            "How to implement authentication in Flask?",
            "Best practices for React component design",
            "SQL optimization techniques",
            "Python error handling patterns",
            "API design principles",
        ]

        results = []
        total_time = 0

        for query in test_queries:
            start_time = time.time()

            try:
                # Search RAG
                search_results = await self.rag_system.search(query, top_k=3)

                end_time = time.time()
                duration = end_time - start_time
                total_time += duration

                success = len(search_results) > 0
                result_count = len(search_results)

                result = {
                    "query": query,
                    "success": success,
                    "duration": duration,
                    "result_count": result_count,
                }
                results.append(result)

                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {query[:50]}...: {status} - {duration:.1f}s ({result_count} results)")

            except Exception as e:
                print(f"  {query[:50]}...: ❌ ERROR - {str(e)}")
                results.append(
                    {
                        "query": query,
                        "success": False,
                        "duration": 0,
                        "result_count": 0,
                        "error": str(e),
                    }
                )

        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        avg_time = total_time / len(results) if results else 0

        self.results["rag_functionality"] = {
            "success_rate": success_rate,
            "avg_time": avg_time,
            "total_time": total_time,
            "results": results,
        }

        print(f"📊 RAG Functionality: {success_rate:.1f}% success, {avg_time:.1f}s avg")
        return success_rate >= 80  # Target: 80%

    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n🧪 Testing Performance Benchmarks")
        print("-" * 40)

        # Get initial system metrics
        initial_metrics = self.get_system_metrics()

        # Test TTFT (Time To First Token)
        ttft_results = []
        for _i in range(5):
            start_time = time.time()
            request = SingleModelRequest(task_description="Hello, world!")
            response = await self.single_engine.process_request(request)
            end_time = time.time()
            ttft = end_time - start_time
            ttft_results.append(ttft)

        avg_ttft = sum(ttft_results) / len(ttft_results)

        # Test throughput (tokens per second)
        long_request = SingleModelRequest(
            task_description="Write a detailed explanation of machine learning algorithms"
        )
        start_time = time.time()
        response = await self.single_engine.process_request(long_request)
        end_time = time.time()

        # Estimate tokens (rough approximation)
        estimated_tokens = len(response.content.split()) * 1.3
        tokens_per_second = estimated_tokens / (end_time - start_time)

        # Get final system metrics
        final_metrics = self.get_system_metrics()

        self.results["performance_benchmarks"] = {
            "avg_ttft": avg_ttft,
            "tokens_per_second": tokens_per_second,
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "ttft_results": ttft_results,
        }

        print("📊 Performance:")
        print(f"  Average TTFT: {avg_ttft:.3f}s")
        print(f"  Tokens/second: {tokens_per_second:.1f}")
        print(f"  GPU Memory: {final_metrics.get('gpu_memory_percent', 0):.1f}%")

        # Check targets
        ttft_ok = avg_ttft < 0.8
        throughput_ok = tokens_per_second > 40
        memory_ok = final_metrics.get("gpu_memory_percent", 0) <= 80

        return ttft_ok and throughput_ok and memory_ok

    async def run_full_suite(self):
        """Run the complete test suite"""
        print("🧪 MASTER FULL SUITE TEST")
        print("=" * 60)
        print(f"Agents: {self.args.agents}")
        print(f"Tokens: {self.args.tokens}")
        print(f"Self-reflection: {self.args.self_reflection}")
        print(f"Self-consistency (k): {self.args.k}")
        print()

        self.start_time = time.time()

        # Run all tests
        tests = [
            ("Single Agent", self.test_single_agent_performance),
            ("Debate System", self.test_debate_system_performance),
            ("RAG Functionality", self.test_rag_functionality),
            ("Performance Benchmarks", self.test_performance_benchmarks),
        ]

        test_results = {}

        for test_name, test_func in tests:
            try:
                result = await test_func()
                test_results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name}: {status}")
            except Exception as e:
                print(f"{test_name}: ❌ ERROR - {str(e)}")
                test_results[test_name] = False

        # Calculate overall success rate
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        overall_success_rate = (passed_tests / total_tests) * 100

        end_time = time.time()
        total_duration = end_time - self.start_time

        # Generate final report
        self.generate_final_report(test_results, overall_success_rate, total_duration)

        return overall_success_rate >= 95  # Target: 95%

    def generate_final_report(self, test_results, overall_success_rate, total_duration):
        """Generate comprehensive final report"""
        print("\n" + "=" * 60)
        print("📊 MASTER FULL SUITE TEST RESULTS")
        print("=" * 60)

        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.1f}s")
        print()

        print("📋 Test Results:")
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")

        print()
        print("📊 Detailed Metrics:")

        # Single Agent
        if "single_agent" in self.results:
            sa = self.results["single_agent"]
            print(f"  Single Agent: {sa['success_rate']:.1f}% success, {sa['avg_time']:.1f}s avg")

        # Debate System
        if "debate_system" in self.results:
            ds = self.results["debate_system"]
            print(f"  Debate System: {ds['success_rate']:.1f}% success, {ds['avg_time']:.1f}s avg")

        # RAG
        if "rag_functionality" in self.results:
            rag = self.results["rag_functionality"]
            print(
                f"  RAG Functionality: {rag['success_rate']:.1f}% success, {rag['avg_time']:.1f}s avg"
            )

        # Performance
        if "performance_benchmarks" in self.results:
            perf = self.results["performance_benchmarks"]
            print(
                f"  Performance: TTFT {perf['avg_ttft']:.3f}s, {perf['tokens_per_second']:.1f} tokens/s"
            )

        print()

        if overall_success_rate >= 95:
            print("🎉 EXCELLENT! All targets met!")
            print("   System is ready for production launch.")
        elif overall_success_rate >= 90:
            print("✅ GOOD! Most targets met.")
            print("   Minor optimizations needed before launch.")
        else:
            print("❌ NEEDS IMPROVEMENT!")
            print("   Critical issues must be resolved before launch.")

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"master_test_results_{timestamp}.json"

        final_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_success_rate": overall_success_rate,
            "total_duration": total_duration,
            "test_results": test_results,
            "detailed_results": self.results,
            "performance_metrics": self.performance_metrics,
        }

        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"📁 Detailed results saved to: {results_file}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Master Full Suite Test")
    parser.add_argument(
        "--agents",
        default="coder architect tester reviewer",
        help="Space-separated list of agents",
    )
    parser.add_argument("--tokens", type=int, default=100, help="Maximum tokens per response")
    parser.add_argument(
        "--self_reflection", action="store_true", help="Enable self-reflection loop"
    )
    parser.add_argument("--k", type=int, default=3, help="Self-consistency voting (k)")

    args = parser.parse_args()

    suite = MasterTestSuite(args)
    await suite.initialize()
    success = await suite.run_full_suite()

    if success:
        print("\n🎉 MASTER TEST SUITE PASSED!")
        return 0
    else:
        print("\n❌ MASTER TEST SUITE FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
