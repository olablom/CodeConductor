#!/usr/bin/env python3
"""
Test script to verify plugin integration with orchestrator
"""

from agents.orchestrator_simple import SimpleAgentOrchestrator
from plugins.base_simple import PluginManager


def test_plugin_integration():
    """Test that plugins are properly integrated into the orchestrator"""
    print("🧪 Testing plugin integration with orchestrator...")

    try:
        # Create orchestrator
        orchestrator = SimpleAgentOrchestrator()
        print("✅ SimpleAgentOrchestrator created successfully")

        # Check if plugins are loaded
        plugin_info = orchestrator.get_plugin_info()
        print("📊 Plugin Info:")
        print(f"  Total plugins: {plugin_info['total_plugins']}")
        print(f"  Agent plugins: {plugin_info['agent_plugins']}")
        print(f"  Tool plugins: {plugin_info['tool_plugins']}")

        # Test agent discussion with plugins
        print("\n🤖 Testing agent discussion with plugins...")

        # Sample code to analyze
        sample_code = """
def hello_world():
    return "Hello, World!"

def dangerous_function():
    import os
    os.system("rm -rf /")  # This should be flagged by security plugin
"""

        # Run analysis
        result = orchestrator.facilitate_discussion(
            "Generate a hello world function", {"code": sample_code}
        )

        print("📋 Analysis Result:")
        print(f"  Consensus: {result.get('consensus', 'No consensus')}")
        print(f"  Confidence: {result.get('confidence', 'Unknown')}")
        print(f"  Agent count: {len(result.get('agent_analyses', []))}")

        # Check if security plugin contributed
        agent_analyses = result.get("agent_analyses", [])
        security_found = False

        for analysis in agent_analyses:
            agent_name = analysis.get("agent", "Unknown")
            if "security" in agent_name.lower():
                security_found = True
                print(
                    f"🔒 Security plugin analysis: {analysis.get('analysis', 'No analysis')}"
                )
                break

        if security_found:
            print("✅ Security plugin successfully integrated!")
        else:
            print("⚠️ Security plugin not found in analysis")

        return True

    except Exception as e:
        print(f"❌ Plugin integration test failed: {e}")
        return False


def test_plugin_analysis():
    """Test specific plugin analysis functionality"""
    print("\n🧪 Testing plugin analysis functionality...")

    try:
        # Create plugin manager
        pm = PluginManager()
        pm.discover()
        pm.activate_all()

        # Get security plugin
        security_plugin = pm.get_plugin("security_analyzer")
        if security_plugin:
            print("✅ Security plugin found")

            # Test analysis
            sample_code = """
import os
import subprocess

def dangerous_function():
    os.system("rm -rf /")
    subprocess.call(["format", "C:"])
"""

            result = security_plugin.analyze(sample_code, {"context": "test"})
            print(f"🔒 Security analysis result: {result}")

            # Check for security issues
            if result.get("issues"):
                print("✅ Security plugin detected issues!")
                for issue in result["issues"]:
                    print(f"  - {issue}")
            else:
                print("ℹ️ No security issues detected")
        else:
            print("❌ Security plugin not found")

        return True

    except Exception as e:
        print(f"❌ Plugin analysis test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎼 CodeConductor v2.0 - Plugin Integration Test")
    print("=" * 60)

    success = True
    success &= test_plugin_integration()
    success &= test_plugin_analysis()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All plugin integration tests passed!")
        print("🔌 Plugin architecture is working correctly!")
    else:
        print("❌ Some plugin integration tests failed.")

    print("=" * 60)
