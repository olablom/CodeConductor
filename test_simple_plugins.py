#!/usr/bin/env python3
"""
Test script for Simplified Plugin System
"""

from plugins.base_simple import PluginManager
from agents.orchestrator_simple import SimpleAgentOrchestrator


def test_plugin_discovery():
    """Test plugin discovery functionality"""
    print("🧪 Testing simplified plugin discovery...")

    try:
        plugin_manager = PluginManager()
        plugin_manager.discover()

        plugins = plugin_manager.get_plugins()
        agent_plugins = plugin_manager.get_agent_plugins()
        tool_plugins = plugin_manager.get_tool_plugins()

        print(f"✅ Discovered {len(plugins)} total plugins")
        print(f"  🔌 Agent plugins: {len(agent_plugins)}")
        print(f"  🛠️ Tool plugins: {len(tool_plugins)}")

        for plugin in plugins:
            print(f"  📦 {plugin.name()} v{plugin.version()}")
            print(f"     {plugin.description()}")
            print(f"     Type: {'Agent' if plugin in agent_plugins else 'Tool'}")

        return True

    except Exception as e:
        print(f"❌ Plugin discovery failed: {e}")
        return False


def test_plugin_activation():
    """Test plugin activation functionality"""
    print("\n🧪 Testing plugin activation...")

    try:
        plugin_manager = PluginManager()
        plugin_manager.discover()
        plugin_manager.activate_all()

        print("✅ All plugins activated successfully")

        # Test deactivation
        plugin_manager.deactivate_all()
        print("✅ All plugins deactivated successfully")

        return True

    except Exception as e:
        print(f"❌ Plugin activation failed: {e}")
        return False


def test_agent_orchestrator():
    """Test AgentOrchestrator with simplified plugins"""
    print("\n🧪 Testing SimpleAgentOrchestrator with plugins...")

    try:
        orchestrator = SimpleAgentOrchestrator(enable_plugins=True)

        # Get agent summary
        agent_summary = orchestrator.get_agent_summary()
        print("📊 Agent Summary:")
        print(f"  Total agents: {agent_summary['total_agents']}")
        print(f"  Core agents: {agent_summary['core_agents']}")
        print(f"  Plugin agents: {agent_summary['plugin_agents']}")

        # Get plugin info
        plugin_info = orchestrator.get_plugin_info()
        print("📦 Plugin Info:")
        print(f"  Total plugins: {plugin_info['total_plugins']}")
        print(f"  Agent plugins: {plugin_info['agent_plugins']}")
        print(f"  Tool plugins: {plugin_info['tool_plugins']}")
        print(f"  Enabled plugins: {plugin_info['enabled_plugins']}")

        if plugin_info["plugins"]:
            for plugin_name, details in plugin_info["plugins"].items():
                print(f"    🔌 {plugin_name}: {details['description']}")

        return True

    except Exception as e:
        print(f"❌ AgentOrchestrator test failed: {e}")
        return False


def test_plugin_analysis():
    """Test plugin analysis functionality"""
    print("\n🧪 Testing plugin analysis...")

    try:
        orchestrator = SimpleAgentOrchestrator(enable_plugins=True)

        # Test discussion with plugins
        test_prompt = "Create a simple calculator function"
        test_context = {
            "code": """
import os
import subprocess

def dangerous_function():
    os.system("rm -rf /")
    password = "secret123"
    return "done"
""",
            "discussion_history": [],
        }

        print("🤖 Running multi-agent discussion with simplified plugins...")
        print(f"📝 Prompt: {test_prompt}")

        result = orchestrator.facilitate_discussion(test_prompt, test_context)

        print("✅ Discussion completed!")
        print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
        print(f"📊 Total analyses: {len(result.get('agent_analyses', {}))}")

        # Check for plugin analyses
        plugin_analyses = [
            name
            for name in result.get("agent_analyses", {}).keys()
            if name.startswith("plugin_")
        ]

        if plugin_analyses:
            print(f"🔌 Plugin analyses found: {plugin_analyses}")
            for plugin_name in plugin_analyses:
                analysis = result["agent_analyses"][plugin_name]
                print(f"  📋 {plugin_name}:")
                if "security_score" in analysis:
                    print(f"    Security score: {analysis['security_score']}")
                if "vulnerabilities" in analysis:
                    print(f"    Vulnerabilities: {len(analysis['vulnerabilities'])}")
                if "recommendations" in analysis:
                    print(f"    Recommendations: {len(analysis['recommendations'])}")
        else:
            print("⚠️ No plugin analyses found")

        return True

    except Exception as e:
        print(f"❌ Plugin analysis test failed: {e}")
        return False


def test_security_plugin():
    """Test the Security Plugin specifically"""
    print("\n🧪 Testing Security Plugin...")

    try:
        from plugins.security_plugin import SecurityPlugin

        # Create plugin instance
        security_plugin = SecurityPlugin()

        # Test activation
        security_plugin.activate()
        print("✅ Security Plugin activated")

        # Test analysis
        test_code = """
import os
import subprocess

def dangerous_function():
    os.system("rm -rf /")
    password = "secret123"
    return "done"
"""

        context = {"prompt": "Test security analysis", "discussion_history": []}

        analysis = security_plugin.analyze(test_code, context)
        print("🔒 Security analysis completed:")
        print(f"  Security score: {analysis.get('security_score', 'N/A')}")
        print(f"  Vulnerabilities found: {analysis.get('vulnerability_count', 'N/A')}")
        print(f"  Severity: {analysis.get('severity', 'N/A')}")

        if analysis.get("recommendations"):
            print("  Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"    - {rec}")

        # Test deactivation
        security_plugin.deactivate()
        print("✅ Security Plugin deactivated")

        return True

    except Exception as e:
        print(f"❌ Security Plugin test failed: {e}")
        return False


def test_formatter_plugin():
    """Test the Formatter Plugin specifically"""
    print("\n🧪 Testing Formatter Plugin...")

    try:
        from plugins.formatter_plugin import FormatterPlugin

        # Create plugin instance
        formatter_plugin = FormatterPlugin()

        # Test activation
        formatter_plugin.activate()
        print("✅ Formatter Plugin activated")

        # Test formatting
        test_code = """
import os
import sys
from typing import List

def badly_formatted_function( x,y ):
    result=x+y
    return result
"""

        result = formatter_plugin.execute(test_code)

        if result.get("success"):
            print("✅ Code formatting completed:")
            print(f"  Original lines: {result['report']['summary']['original_lines']}")
            print(
                f"  Formatted lines: {result['report']['summary']['formatted_lines']}"
            )
            print(
                f"  Violations found: {result['report']['summary']['violations_found']}"
            )
            print(f"  Lines changed: {result['changes_made']['lines_changed']}")

            # Show formatted code
            print("\n📝 Formatted code:")
            print(result["formatted_code"])
        else:
            print(f"❌ Code formatting failed: {result.get('error', 'Unknown error')}")
            return False

        # Test deactivation
        formatter_plugin.deactivate()
        print("✅ Formatter Plugin deactivated")

        return True

    except Exception as e:
        print(f"❌ Formatter Plugin test failed: {e}")
        return False


def test_plugin_configuration():
    """Test plugin configuration functionality"""
    print("\n🧪 Testing plugin configuration...")

    try:
        plugin_manager = PluginManager()

        # Test configuration loading
        plugin_manager.load_config()
        print("✅ Plugin configuration loaded")

        # Test configuration saving
        plugin_manager.save_config()
        print("✅ Plugin configuration saved")

        # Test plugin enabling/disabling
        plugin_manager.discover()
        plugins = plugin_manager.get_plugins()

        if plugins:
            plugin_name = plugins[0].name()

            # Test enabling
            if plugin_manager.enable_plugin(plugin_name):
                print(f"✅ Enabled plugin: {plugin_name}")

            # Test disabling
            if plugin_manager.disable_plugin(plugin_name):
                print(f"✅ Disabled plugin: {plugin_name}")

        return True

    except Exception as e:
        print(f"❌ Plugin configuration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎼 CodeConductor v2.0 - Simplified Plugin System Test")
    print("=" * 60)

    success = True
    success &= test_plugin_discovery()
    success &= test_plugin_activation()
    success &= test_agent_orchestrator()
    success &= test_plugin_analysis()
    success &= test_security_plugin()
    success &= test_formatter_plugin()
    success &= test_plugin_configuration()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All simplified plugin system tests passed!")
        print("🔌 Your elegant plugin architecture is working perfectly!")
    else:
        print("❌ Some simplified plugin system tests failed. Check the errors above.")

    print("=" * 60)
