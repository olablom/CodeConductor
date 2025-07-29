#!/usr/bin/env python3
"""
Main Automated Test Runner for CodeConductor
Runs smart model loading tests and GUI integration tests in sequence
"""

import asyncio
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import test modules
from test_smart_model_loading import SmartModelLoadingTester
from test_gui_integration import GUIIntegrationTester


class AutomatedTestRunner:
    """Main test runner that executes all automated tests."""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_results = []
        
    async def run_all_tests(self) -> dict:
        """Run all automated tests in sequence."""
        print("🚀 CodeConductor Automated Test Suite")
        print("=" * 60)
        print("Testing smart model loading and GUI integration")
        print("=" * 60)
        
        # Phase 1: Smart Model Loading Tests
        print("\n📋 PHASE 1: Smart Model Loading Tests")
        print("-" * 40)
        
        smart_loading_tester = SmartModelLoadingTester()
        smart_loading_report = await smart_loading_tester.run_all_tests()
        
        self.test_results.append({
            "phase": "Smart Model Loading",
            "report": smart_loading_report
        })
        
        # Phase 2: GUI Integration Tests
        print("\n📋 PHASE 2: GUI Integration Tests")
        print("-" * 40)
        
        gui_tester = GUIIntegrationTester()
        gui_report = await gui_tester.run_all_tests()
        
        self.test_results.append({
            "phase": "GUI Integration",
            "report": gui_report
        })
        
        return self.generate_final_report()
    
    def generate_final_report(self) -> dict:
        """Generate comprehensive final report."""
        execution_time = time.time() - self.start_time
        
        # Aggregate results
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        for phase_result in self.test_results:
            summary = phase_result["report"]["summary"]
            total_tests += summary["total_tests"]
            total_passed += summary["passed"]
            total_failed += summary["failed"]
            total_errors += summary["errors"]
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        final_report = {
            "overall_summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "errors": total_errors,
                "success_rate": overall_success_rate,
                "execution_time": execution_time
            },
            "phase_results": self.test_results,
            "recommendations": self.generate_final_recommendations(),
            "next_steps": self.generate_next_steps()
        }
        
        return final_report
    
    def generate_final_recommendations(self) -> list:
        """Generate final recommendations based on all test results."""
        recommendations = []
        
        # Check overall success rate
        overall_success_rate = self.test_results[0]["report"]["summary"]["success_rate"]
        gui_success_rate = self.test_results[1]["report"]["summary"]["success_rate"]
        
        if overall_success_rate >= 80 and gui_success_rate >= 80:
            recommendations.append("✅ Excellent test results! Ready for manual testing.")
            recommendations.append("🎯 Both smart model loading and GUI integration are solid.")
        elif overall_success_rate >= 60 and gui_success_rate >= 60:
            recommendations.append("⚠️ Good test results with some issues.")
            recommendations.append("🔧 Proceed with manual testing but be aware of potential issues.")
        else:
            recommendations.append("❌ Multiple issues detected.")
            recommendations.append("🔧 Fix implementation issues before manual testing.")
        
        # Specific recommendations based on phase results
        for phase_result in self.test_results:
            phase_name = phase_result["phase"]
            phase_summary = phase_result["report"]["summary"]
            
            if phase_summary["success_rate"] < 60:
                recommendations.append(f"❌ {phase_name} needs attention (success rate: {phase_summary['success_rate']:.1f}%)")
            elif phase_summary["success_rate"] < 80:
                recommendations.append(f"⚠️ {phase_name} has minor issues (success rate: {phase_summary['success_rate']:.1f}%)")
            else:
                recommendations.append(f"✅ {phase_name} is solid (success rate: {phase_summary['success_rate']:.1f}%)")
        
        return recommendations
    
    def generate_next_steps(self) -> list:
        """Generate next steps based on test results."""
        next_steps = []
        
        overall_success_rate = self.test_results[0]["report"]["summary"]["success_rate"]
        gui_success_rate = self.test_results[1]["report"]["summary"]["success_rate"]
        
        if overall_success_rate >= 80 and gui_success_rate >= 80:
            next_steps.extend([
                "1. ✅ Commit current implementation",
                "2. 🚀 Start manual testing in browser",
                "3. 📋 Follow MANUAL_TEST_GUIDE.md",
                "4. 🎯 Test smart model loading functionality",
                "5. 📊 Document any issues found"
            ])
        elif overall_success_rate >= 60 and gui_success_rate >= 60:
            next_steps.extend([
                "1. ⚠️ Review and fix identified issues",
                "2. 🔧 Run tests again after fixes",
                "3. 🚀 Proceed with manual testing carefully",
                "4. 📋 Follow MANUAL_TEST_GUIDE.md",
                "5. 🎯 Pay attention to problematic areas"
            ])
        else:
            next_steps.extend([
                "1. ❌ Fix critical implementation issues",
                "2. 🔧 Address all test failures",
                "3. 🔄 Run automated tests again",
                "4. ⚠️ Do not proceed to manual testing yet",
                "5. 📋 Review error logs and fix issues"
            ])
        
        return next_steps


async def main():
    """Main test runner."""
    print("🎯 CodeConductor Automated Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("codeconductor_app.py").exists():
        print("❌ Error: codeconductor_app.py not found")
        print("   Please run this script from the project root directory")
        sys.exit(1)
    
    # Run all tests
    runner = AutomatedTestRunner()
    final_report = await runner.run_all_tests()
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    overall_summary = final_report["overall_summary"]
    print(f"Total Tests: {overall_summary['total_tests']}")
    print(f"Passed: {overall_summary['passed']} ✅")
    print(f"Failed: {overall_summary['failed']} ❌")
    print(f"Errors: {overall_summary['errors']} 💥")
    print(f"Overall Success Rate: {overall_summary['success_rate']:.1f}%")
    print(f"Total Execution Time: {overall_summary['execution_time']:.2f}s")
    
    # Print phase results
    print("\n📋 PHASE RESULTS")
    print("-" * 40)
    for phase_result in final_report["phase_results"]:
        phase_name = phase_result["phase"]
        phase_summary = phase_result["report"]["summary"]
        print(f"{phase_name}: {phase_summary['success_rate']:.1f}% ({phase_summary['passed']}/{phase_summary['total_tests']} tests)")
    
    # Print recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    for rec in final_report["recommendations"]:
        print(f"• {rec}")
    
    # Print next steps
    print("\n🎯 NEXT STEPS")
    print("-" * 40)
    for step in final_report["next_steps"]:
        print(f"• {step}")
    
    # Final verdict
    print("\n🏆 FINAL VERDICT")
    print("=" * 60)
    
    overall_success_rate = overall_summary["success_rate"]
    if overall_success_rate >= 80:
        print("✅ READY FOR MANUAL TESTING")
        print("   All automated tests passed successfully.")
        print("   Smart model loading and GUI integration are solid.")
        print("   Proceed with manual testing in the browser.")
        return True
    elif overall_success_rate >= 60:
        print("⚠️ PROCEED WITH CAUTION")
        print("   Some automated tests failed but core functionality works.")
        print("   Review issues and proceed with manual testing carefully.")
        return True
    else:
        print("❌ NOT READY FOR MANUAL TESTING")
        print("   Multiple automated test failures detected.")
        print("   Fix implementation issues before manual testing.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        
        if success:
            print("\n🎉 Automated tests completed successfully!")
            print("   You can now proceed with manual testing.")
        else:
            print("\n💥 Automated tests failed!")
            print("   Please fix issues before manual testing.")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1) 