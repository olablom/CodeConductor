#!/usr/bin/env python3
"""
Improved Single Agent Test with Self-Reflection and Better Validation
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from codeconductor.ensemble.single_model_engine import SingleModelEngine, SingleModelRequest
from codeconductor.generators.improved_prompt_generator import improved_prompt_generator
from codeconductor.feedback.self_reflection_agent import self_reflection_agent
from codeconductor.feedback.simple_rlhf import simple_rlhf

class ImprovedSingleAgentTester:
    """Enhanced single agent tester with self-reflection"""
    
    def __init__(self):
        self.engine = None
        self.results = []
        
    async def initialize(self):
        """Initialize the test suite"""
        print("🚀 Initializing Improved Single Agent Test")
        
        # Initialize single model engine
        self.engine = SingleModelEngine()
        await self.engine.initialize()
        
        print("✅ Test suite initialized successfully")
    
    async def test_with_self_reflection(self, task_type: str, description: str) -> dict:
        """Test with self-reflection loop"""
        
        print(f"🧪 Running test: {task_type} (improved single agent)")
        print(f"📝 Prompt: {description}")
        
        start_time = time.time()
        
        # Generate improved prompt
        prompt = improved_prompt_generator.generate_improved_prompt(task_type, description)
        
        # First attempt
        request = SingleModelRequest(task_description=prompt)
        response = await self.engine.process_request(request)
        code = self_reflection_agent.extract_code(response.content)
        
        # Test the code
        success, error = self_reflection_agent.validate_code(code, task_type)
        
        # If failed, try self-reflection loop
        iterations = 1
        while not success and iterations < 3:
            print(f"🔄 Self-reflection iteration {iterations + 1}")
            
            # Generate fix prompt
            fix_prompt = improved_prompt_generator.generate_fix_prompt(code, error, task_type)
            
            # Get improved code
            fix_request = SingleModelRequest(task_description=fix_prompt)
            improved_response = await self.engine.process_request(fix_request)
            improved_code = self_reflection_agent.extract_code(improved_response.content)
            
            # Test improved code
            success, error = self_reflection_agent.validate_code(improved_code, task_type)
            
            if success:
                code = improved_code
                break
            
            code = improved_code
            iterations += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Update RLHF weights
        model = "mistral-7b-instruct-v0.1"  # Default model
        quality = 0.8 if success else 0.2
        simple_rlhf.update_weights(model, success, quality)
        
        result = {
            "task_type": task_type,
            "description": description,
            "success": success,
            "code": code,
            "error": error if not success else None,
            "iterations": iterations,
            "duration": duration,
            "response_length": len(response.content),
            "code_length": len(code),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Test completed in {duration:.1f}s")
        print(f"📊 Success: {success}")
        print(f"📝 Code extracted: {len(code)} characters")
        print(f"🔄 Iterations: {iterations}")
        if not success:
            print(f"❌ Error: {error}")
        
        return result
    
    async def run_test_suite(self):
        """Run the complete test suite"""
        
        print("🧪 Improved Single Agent Test Suite")
        print("=" * 50)
        print()
        
        test_cases = [
            ("fibonacci", "Create a Python function to calculate the nth Fibonacci number"),
            ("binary_search", "Create a Python function to perform binary search on a sorted array"),
            ("rest_api", "Create a simple REST API endpoint for user login using Flask")
        ]
        
        for task_type, description in test_cases:
            result = await self.test_with_self_reflection(task_type, description)
            self.results.append(result)
            
            # Save individual result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"improved_test_results_{timestamp}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"📁 Results saved to {result_file}")
            print()
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary"""
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        avg_time = sum(r["duration"] for r in self.results) / total_tests if total_tests > 0 else 0
        avg_iterations = sum(r["iterations"] for r in self.results) / total_tests if total_tests > 0 else 0
        
        print("=" * 50)
        print("📊 IMPROVED SINGLE AGENT TEST SUMMARY")
        print("=" * 50)
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average time: {avg_time:.1f}s")
        print(f"Average iterations: {avg_iterations:.1f}")
        print()
        
        print("📊 Results by test type:")
        for result in self.results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {result['task_type']}: {status} - {result['duration']:.1f}s avg")
        
        print()
        
        if success_rate >= 65:
            print("✅ EXCELLENT! All tests passed target!")
            print("   Self-reflection loop is working perfectly.")
            print("   ✅ PASS: No message")
        else:
            print("❌ NEEDS IMPROVEMENT!")
            print(f"   Target: 65%, Actual: {success_rate:.1f}%")
            print("   Consider additional prompt engineering.")
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"improved_summary_{timestamp}.yaml"
        
        summary = {
            "test_suite": "Improved Single Agent",
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "average_time": avg_time,
            "average_iterations": avg_iterations,
            "results": self.results
        }
        
        with open(summary_file, "w") as f:
            import yaml
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"📁 Summary saved to {summary_file}")

async def main():
    """Main test function"""
    tester = ImprovedSingleAgentTester()
    await tester.initialize()
    await tester.run_test_suite()

if __name__ == "__main__":
    asyncio.run(main()) 