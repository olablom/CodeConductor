#!/usr/bin/env python3
"""
Test script for PolicyLoader functionality
"""

from integrations.policy_loader import PolicyLoader
from agents.policy_agent import PolicyAgent


def test_policy_loader():
    """Test basic PolicyLoader functionality"""
    print("🧪 Testing PolicyLoader...")

    try:
        # Test loading policies
        loader = PolicyLoader()
        print("✅ PolicyLoader loaded successfully!")

        # Test basic getters
        print(f"📋 Enforcement mode: {loader.get_enforcement_mode()}")
        print(f"🔒 Blocked imports: {len(loader.get_blocked_imports())}")
        print(f"🚫 Blocked patterns: {len(loader.get_blocked_patterns())}")
        print(f"⚠️ Forbidden functions: {len(loader.get_forbidden_functions())}")
        print(f"📏 Max line length: {loader.get_max_line_length()}")

        # Test policy summary
        summary = loader.get_policy_summary()
        print(f"📊 Policy summary: {summary}")

        return True

    except Exception as e:
        print(f"❌ PolicyLoader test failed: {e}")
        return False


def test_policy_agent():
    """Test PolicyAgent with YAML policies"""
    print("\n🧪 Testing PolicyAgent with YAML policies...")

    try:
        # Test PolicyAgent initialization
        agent = PolicyAgent()
        print("✅ PolicyAgent loaded successfully!")

        # Test policy summary
        summary = agent.get_policy_summary()
        print(f"📊 Policy summary: {summary}")

        return True

    except Exception as e:
        print(f"❌ PolicyAgent test failed: {e}")
        return False


def test_code_analysis():
    """Test code analysis with sample code"""
    print("\n🧪 Testing code analysis...")

    try:
        loader = PolicyLoader()

        # Test safe code
        safe_code = """
def hello_world():
    return "Hello, World!"

def add_numbers(a: int, b: int) -> int:
    return a + b
"""

        # Test dangerous code
        dangerous_code = """
import os
import subprocess

def dangerous_function():
    os.system("rm -rf /")
    subprocess.call(["sudo", "reboot"])
"""

        # Analyze safe code
        safe_result = loader.analyze_code(safe_code)
        print(f"✅ Safe code analysis: {safe_result['summary']}")

        # Analyze dangerous code
        dangerous_result = loader.analyze_code(dangerous_code)
        print(f"🚫 Dangerous code analysis: {dangerous_result['summary']}")
        print(f"   Should block: {dangerous_result['should_block']}")

        # Show violations
        for violation in dangerous_result["violations"]:
            print(f"   - {violation.description} (line {violation.line_number})")

        return True

    except Exception as e:
        print(f"❌ Code analysis test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎼 CodeConductor v2.0 - PolicyLoader Test")
    print("=" * 50)

    success = True
    success &= test_policy_loader()
    success &= test_policy_agent()
    success &= test_code_analysis()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! YAML-driven policies are working!")
    else:
        print("❌ Some tests failed. Check the errors above.")
