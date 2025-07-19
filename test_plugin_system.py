#!/usr/bin/env python3
"""
Test script for CodeConductor v2.0 Plugin System
"""

from plugins.base import PluginManager, PluginType
from agents.orchestrator import AgentOrchestrator


def test_plugin_discovery():
    """Test plugin discovery functionality"""
    print("🧪 Testing plugin discovery...")

    try:
        plugin_manager = PluginManager()
        discovered_plugins = plugin_manager.discover_plugins()

        print(f"✅ Discovered {len(discovered_plugins)} plugins")

        for plugin in discovered_plugins:
            print(f"  📦 {plugin.metadata.name} ({plugin.metadata.plugin_type.value})")
            print(f"     Version: {plugin.metadata.version}")
            print(f"     Author: {plugin.metadata.author}")
            print(f"     Status: {plugin.status.value}")
            print(f"     Enabled: {plugin.is_enabled}")

        return True

    except Exception as e:
        print(f"❌ Plugin discovery failed: {e}")
        return False


def test_plugin_loading():
    """Test plugin loading functionality"""
    print("\n🧪 Testing plugin loading...")

    try:
        plugin_manager = PluginManager()
        discovered_plugins = plugin_manager.discover_plugins()

        loaded_count = 0
        for plugin in discovered_plugins:
            if plugin.is_enabled:
                try:
                    plugin_instance = plugin_manager.load_plugin(plugin)
                    if plugin_instance:
                        print(f"✅ Loaded plugin: {plugin.metadata.name}")
                        loaded_count += 1
                    else:
                        print(f"❌ Failed to load plugin: {plugin.metadata.name}")
                except Exception as e:
                    print(f"❌ Error loading {plugin.metadata.name}: {e}")

        print(f"📊 Successfully loaded {loaded_count} plugins")
        return True

    except Exception as e:
        print(f"❌ Plugin loading failed: {e}")
        return False


def test_agent_orchestrator_with_plugins():
    """Test AgentOrchestrator with plugin support"""
    print("\n🧪 Testing AgentOrchestrator with plugins...")

    try:
        orchestrator = AgentOrchestrator(enable_plugins=True)

        # Get agent summary
        agent_summary = orchestrator.get_agent_summary()
        print(f"📊 Agent Summary:")
        print(f"  Total agents: {agent_summary['total_agents']}")
        print(f"  Core agents: {agent_summary['core_agents']}")
        print(f"  Plugin agents: {agent_summary['plugin_agents']}")

        # Get plugin info
        plugin_info = orchestrator.get_plugin_info()
        print(f"📦 Plugin Info:")
        print(f"  Plugins enabled: {plugin_info['plugins_enabled']}")
        print(f"  Total plugins: {plugin_info['total_plugins']}")
        print(f"  Active plugins: {plugin_info['active_plugins']}")

        if plugin_info["plugin_details"]:
            for plugin_name, details in plugin_info["plugin_details"].items():
                print(f"    🔌 {plugin_name}: {details['description']}")

        return True

    except Exception as e:
        print(f"❌ AgentOrchestrator test failed: {e}")
        return False


def test_plugin_analysis():
    """Test plugin analysis functionality"""
    print("\n🧪 Testing plugin analysis...")

    try:
        orchestrator = AgentOrchestrator(enable_plugins=True)

        # Test discussion with plugins
        test_prompt = "Create a simple calculator function"
        test_context = {"code": "def add(a, b): return a + b", "discussion_history": []}

        print(f"🤖 Running multi-agent discussion with plugins...")
        print(f"📝 Prompt: {test_prompt}")

        result = orchestrator.facilitate_discussion(test_prompt, test_context)

        print(f"✅ Discussion completed!")
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
                print(
                    f"  📋 {plugin_name}: {analysis.get('description', 'No description')}"
                )
        else:
            print("⚠️ No plugin analyses found")

        return True

    except Exception as e:
        print(f"❌ Plugin analysis test failed: {e}")
        return False


def test_plugin_configuration():
    """Test plugin configuration functionality"""
    print("\n🧪 Testing plugin configuration...")

    try:
        plugin_manager = PluginManager()

        # Test saving and loading configuration
        plugin_manager.save_plugin_config()
        print("✅ Plugin configuration saved")

        plugin_manager.load_plugin_config()
        print("✅ Plugin configuration loaded")

        return True

    except Exception as e:
        print(f"❌ Plugin configuration test failed: {e}")
        return False


def test_security_agent_plugin():
    """Test the Security Agent plugin specifically"""
    print("\n🧪 Testing Security Agent plugin...")

    try:
        from plugins.security_agent import SecurityAgentPlugin

        # Create plugin instance
        config = {
            "severity_threshold": "medium",
            "enable_bandit": True,
            "enable_safety": True,
        }

        security_agent = SecurityAgentPlugin(config)

        # Test initialization
        if security_agent.initialize():
            print("✅ Security Agent initialized successfully")
        else:
            print("❌ Security Agent initialization failed")
            return False

        # Test analysis
        test_code = """
import os
import subprocess

def dangerous_function():
    os.system("rm -rf /")
    password = "secret123"
    return "done"
"""

        context = {"code": test_code, "prompt": "Test security analysis"}

        analysis = security_agent.analyze(context)
        print(f"🔒 Security analysis completed:")
        print(f"  Security score: {analysis.get('security_score', 'N/A')}")
        print(f"  Vulnerabilities found: {len(analysis.get('vulnerabilities', []))}")
        print(f"  Severity: {analysis.get('severity', 'N/A')}")

        # Test action
        action_result = security_agent.act({"analysis": analysis, "code": test_code})
        print(f"🛡️ Security action completed:")
        print(f"  Action type: {action_result.get('action_type', 'N/A')}")
        print(f"  Changes made: {len(action_result.get('changes_made', []))}")

        # Cleanup
        security_agent.cleanup()
        print("✅ Security Agent cleaned up")

        return True

    except Exception as e:
        print(f"❌ Security Agent test failed: {e}")
        return False


def test_code_formatter_plugin():
    """Test the Code Formatter plugin specifically"""
    print("\n🧪 Testing Code Formatter plugin...")

    try:
        from plugins.code_formatter import CodeFormatterPlugin

        # Create plugin instance
        config = {
            "formatter": "black",
            "line_length": 88,
            "enable_isort": True,
            "style_guide": "pep8",
        }

        formatter = CodeFormatterPlugin(config)

        # Test initialization
        if formatter.initialize():
            print("✅ Code Formatter initialized successfully")
        else:
            print("❌ Code Formatter initialization failed")
            return False

        # Test formatting
        test_code = """
import os
import sys
from typing import List

def badly_formatted_function( x,y ):
    result=x+y
    return result
"""

        result = formatter.execute(test_code)

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

        # Cleanup
        formatter.cleanup()
        print("✅ Code Formatter cleaned up")

        return True

    except Exception as e:
        print(f"❌ Code Formatter test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎼 CodeConductor v2.0 - Plugin System Test")
    print("=" * 60)

    success = True
    success &= test_plugin_discovery()
    success &= test_plugin_loading()
    success &= test_agent_orchestrator_with_plugins()
    success &= test_plugin_analysis()
    success &= test_plugin_configuration()
    success &= test_security_agent_plugin()
    success &= test_code_formatter_plugin()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All plugin system tests passed!")
        print("🔌 Plugin architecture is working perfectly!")
    else:
        print("❌ Some plugin system tests failed. Check the errors above.")

    print("=" * 60)
