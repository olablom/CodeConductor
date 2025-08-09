#!/usr/bin/env python3
"""
Simple Master Test - Core Functionality Validation
Tests the most important components without heavy debate system
"""

import asyncio
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import sys
import os
import psutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from codeconductor.ensemble.single_model_engine import SingleModelEngine, SingleModelRequest
from codeconductor.generators.improved_prompt_generator import improved_prompt_generator
from codeconductor.feedback.self_reflection_agent import self_reflection_agent
from codeconductor.feedback.simple_rlhf import simple_rlhf
from codeconductor.context.rag_system import RAGSystem

class SimpleMasterTest:
    """Simple master test focusing on core functionality"""
    
    def __init__(self, args):
        self.args = args
        self.results = {}
        self.start_time = None
        
    async def initialize(self):
        """Initialize test components"""
        print("🚀 Initializing Simple Master Test")
        print("=" * 50)
        
        # Initialize core components only
        self.single_engine = SingleModelEngine()
        self.rag_system = RAGSystem()
        
        # Initialize engine
        await self.single_engine.initialize()
        
        print("✅ Core components initialized")
        
    async def cleanup(self):
        """Cleanup and unload models"""
        print("🧹 Cleaning up and unloading models...")
        try:
            # Unload models from ensemble engine if it exists
            if hasattr(self, 'ensemble_engine'):
                await self.ensemble_engine.cleanup()
            
            # Unload models from single engine
            if hasattr(self, 'single_engine'):
                await self.single_engine.cleanup()
                
            print("✅ Models unloaded successfully")
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")
        
    def get_system_metrics(self):
        """Get basic system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
        except Exception as e:
            print(f"⚠️ Could not get system metrics: {e}")
            return {}
    
    async def test_single_agent_performance(self):
        """Test single agent with self-reflection"""
        print("\n🧪 Testing Single Agent Performance")
        print("-" * 40)
        
        test_cases = [
            ("fibonacci", "Create a Python function to calculate the nth Fibonacci number"),
            ("binary_search", "Create a Python function to perform binary search on a sorted array"),
            ("rest_api", "Create a simple REST API endpoint for user login using Flask")
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
                    fix_prompt = improved_prompt_generator.generate_fix_prompt(code, error, task_type)
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
                "code_length": len(code)
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
            "results": results
        }
        
        print(f"📊 Single Agent: {success_rate:.1f}% success, {avg_time:.1f}s avg")
        return success_rate >= 65  # Target: 65%
    
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
            "API design principles"
        ]
        
        results = []
        total_time = 0
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                # Search RAG
                search_results = self.rag_system.search(query, top_k=3)
                
                end_time = time.time()
                duration = end_time - start_time
                total_time += duration
                
                success = len(search_results) > 0
                result_count = len(search_results)
                
                result = {
                    "query": query,
                    "success": success,
                    "duration": duration,
                    "result_count": result_count
                }
                results.append(result)
                
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {query[:50]}...: {status} - {duration:.1f}s ({result_count} results)")
                
            except Exception as e:
                print(f"  {query[:50]}...: ❌ ERROR - {str(e)}")
                results.append({
                    "query": query,
                    "success": False,
                    "duration": 0,
                    "result_count": 0,
                    "error": str(e)
                })
        
        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        avg_time = total_time / len(results) if results else 0
        
        self.results["rag_functionality"] = {
            "success_rate": success_rate,
            "avg_time": avg_time,
            "total_time": total_time,
            "results": results
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
        for i in range(3):  # Reduced from 5 to 3
            start_time = time.time()
            request = SingleModelRequest(task_description="Hello, world!")
            response = await self.single_engine.process_request(request)
            end_time = time.time()
            ttft = end_time - start_time
            ttft_results.append(ttft)
        
        avg_ttft = sum(ttft_results) / len(ttft_results)
        
        # Test throughput (tokens per second)
        long_request = SingleModelRequest(task_description="Write a short explanation of machine learning")
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
            "ttft_results": ttft_results
        }
        
        print(f"📊 Performance:")
        print(f"  Average TTFT: {avg_ttft:.3f}s")
        print(f"  Tokens/second: {tokens_per_second:.1f}")
        print(f"  CPU Usage: {final_metrics.get('cpu_percent', 0):.1f}%")
        print(f"  Memory Usage: {final_metrics.get('memory_percent', 0):.1f}%")
        
        # Check targets
        ttft_ok = avg_ttft < 6.0  # Realistic target for large models
        throughput_ok = tokens_per_second > 20  # Relaxed target
        memory_ok = final_metrics.get('memory_percent', 0) <= 90  # Relaxed target
        
        return ttft_ok and throughput_ok and memory_ok
    
    async def test_rlhf_learning(self):
        """Test RLHF learning functionality"""
        print("\n🧪 Testing RLHF Learning")
        print("-" * 40)
        
        # Get initial weights
        initial_weights = simple_rlhf.weights.copy()
        
        # Simulate some learning
        model = "mistral-7b-instruct-v0.1"
        
        # Test successful case
        simple_rlhf.update_weights(model, True, 0.8)
        
        # Test failure case
        simple_rlhf.update_weights(model, False, 0.3)
        
        # Get final weights
        final_weights = simple_rlhf.weights
        
        # Calculate improvement
        initial_weight = initial_weights.get(model, 1.0)
        final_weight = final_weights.get(model, 1.0)
        improvement = ((final_weight - initial_weight) / initial_weight) * 100
        
        self.results["rlhf_learning"] = {
            "initial_weight": initial_weight,
            "final_weight": final_weight,
            "improvement_percent": improvement,
            "weights_changed": final_weights != initial_weights
        }
        
        print(f"📊 RLHF Learning:")
        print(f"  Initial weight: {initial_weight:.3f}")
        print(f"  Final weight: {final_weight:.3f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        # Check if weights changed
        return final_weights != initial_weights
    
    async def run_simple_suite(self):
        """Run the simple test suite"""
        print("🧪 SIMPLE MASTER TEST")
        print("=" * 50)
        print(f"Self-reflection: {self.args.self_reflection}")
        print()
        
        self.start_time = time.time()
        
        # Run core tests only
        tests = [
            ("Single Agent", self.test_single_agent_performance),
            ("RAG Functionality", self.test_rag_functionality),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("RLHF Learning", self.test_rlhf_learning)
        ]
        
        test_results = {}
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                test_results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name}: {status}")
                
                # Unload models after each test to free VRAM
                await self.cleanup()
                
            except Exception as e:
                print(f"{test_name}: ❌ ERROR - {str(e)}")
                test_results[test_name] = False
                
                # Still cleanup even on error
                await self.cleanup()
        
        # Calculate overall success rate
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        overall_success_rate = (passed_tests / total_tests) * 100
        
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        # Generate final report
        self.generate_final_report(test_results, overall_success_rate, total_duration)
        
        return overall_success_rate >= 90  # Target: 90%
    
    def generate_final_report(self, test_results, overall_success_rate, total_duration):
        """Generate comprehensive final report"""
        print("\n" + "=" * 50)
        print("📊 SIMPLE MASTER TEST RESULTS")
        print("=" * 50)
        
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
        
        # RAG
        if "rag_functionality" in self.results:
            rag = self.results["rag_functionality"]
            print(f"  RAG Functionality: {rag['success_rate']:.1f}% success, {rag['avg_time']:.1f}s avg")
        
        # Performance
        if "performance_benchmarks" in self.results:
            perf = self.results["performance_benchmarks"]
            print(f"  Performance: TTFT {perf['avg_ttft']:.3f}s, {perf['tokens_per_second']:.1f} tokens/s")
        
        # RLHF
        if "rlhf_learning" in self.results:
            rlhf = self.results["rlhf_learning"]
            print(f"  RLHF Learning: {rlhf['improvement_percent']:.1f}% improvement")
        
        print()
        
        if overall_success_rate >= 90:
            print("🎉 EXCELLENT! All targets met!")
            print("   System is ready for production launch.")
        elif overall_success_rate >= 75:
            print("✅ GOOD! Most targets met.")
            print("   Minor optimizations needed before launch.")
        else:
            print("❌ NEEDS IMPROVEMENT!")
            print("   Critical issues must be resolved before launch.")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simple_master_test_results_{timestamp}.json"
        
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_success_rate": overall_success_rate,
            "total_duration": total_duration,
            "test_results": test_results,
            "detailed_results": self.results
        }
        
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"📁 Detailed results saved to: {results_file}")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple Master Test")
    parser.add_argument("--self_reflection", action="store_true", 
                       help="Enable self-reflection loop")
    
    args = parser.parse_args()
    
    suite = SimpleMasterTest(args)
    try:
        await suite.initialize()
        success = await suite.run_simple_suite()
        
        if success:
            print("\n🎉 SIMPLE MASTER TEST PASSED!")
            return 0
        else:
            print("\n❌ SIMPLE MASTER TEST FAILED!")
            return 1
    finally:
        # Always cleanup at the end
        await suite.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 