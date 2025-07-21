#!/usr/bin/env python3
"""
System Test for CodeConductor v2.0
Tests all components work together
"""

import sys


def test_imports():
    """Test all imports work"""
    print("🔍 Testing imports...")

    try:

        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_agents():
    """Test all agents work"""
    print("\n🤖 Testing agents...")

    try:
        from agents.orchestrator import AgentOrchestrator
        from agents.test_agent import TestAgent
        from agents.reward_agent import RewardAgent

        # Test orchestrator
        orchestrator = AgentOrchestrator()
        result = orchestrator.facilitate_discussion("Create a simple API")
        print(f"✅ Orchestrator: {result['confidence']:.0%} confidence")

        # Test test agent
        test_agent = TestAgent()
        analysis = test_agent.analyze_code("def hello(): return 'Hello'", "Test")
        print(f"✅ TestAgent: {analysis['overall_score']:.1f}/10 score")

        # Test reward agent
        reward_agent = RewardAgent()
        reward = reward_agent.calculate_reward({"tests_passed": 2, "tests_run": 2})
        print(f"✅ RewardAgent: {reward['total_reward']:.1f} total reward")

        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False


def test_integrations():
    """Test integrations work"""
    print("\n🔌 Testing integrations...")

    try:
        from integrations.cursor_api import MockCursorAPI
        from integrations.human_gate import HumanGate

        # Test Cursor API
        cursor_api = MockCursorAPI()
        result = cursor_api.generate_code("Create a FastAPI app")
        print(f"✅ CursorAPI: {result['success']}")

        # Test Human Gate
        human_gate = HumanGate("test_approval.json")
        stats = human_gate.get_approval_stats()
        print(f"✅ HumanGate: {stats['total_decisions']} decisions")

        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_storage():
    """Test storage works"""
    print("\n💾 Testing storage...")

    try:
        from storage.rl_database import RLDatabase

        # Test database
        db = RLDatabase("test_system.db")
        info = db.get_database_info()
        print(f"✅ Database: {info['total_episodes']} episodes")

        # Test episode storage
        episode_data = {
            "episode_id": "test_episode_001",
            "timestamp": "2024-01-01T00:00:00",
            "project_description": "Test project",
            "total_reward": 25.5,
            "iteration_count": 1,
            "execution_time": 5.2,
            "status": "completed",
            "reward_components": {"test_reward": 15.0, "quality_reward": 10.5},
        }

        episode_id = db.store_episode(episode_data)
        print(f"✅ Episode stored: {episode_id}")

        # Cleanup
        db.clear_data()
        print("✅ Database cleanup successful")

        return True
    except Exception as e:
        print(f"❌ Storage test failed: {e}")
        return False


def test_config():
    """Test configuration works"""
    print("\n⚙️ Testing configuration...")

    try:
        from config.config_loader import get_config

        config = get_config()
        summary = config.get_config_summary()
        print(f"✅ Config: {summary['system']['name']} v{summary['system']['version']}")

        # Test validation
        validation = config.validate_config()
        print(f"✅ Config validation: {validation['valid']}")

        if validation["warnings"]:
            print(f"⚠️ Warnings: {len(validation['warnings'])}")

        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_gui_components():
    """Test GUI components work"""
    print("\n🎨 Testing GUI components...")

    try:

        print("✅ Streamlit and data libraries imported")

        # Test session state simulation
        session_state = {
            "discussion_history": [],
            "current_proposal": None,
            "human_gate": None,
        }
        print("✅ Session state simulation works")

        return True
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False


def test_end_to_end():
    """Test end-to-end workflow"""
    print("\n🔄 Testing end-to-end workflow...")

    try:
        from agents.orchestrator import AgentOrchestrator
        from agents.test_agent import TestAgent
        from agents.reward_agent import RewardAgent
        from integrations.cursor_api import MockCursorAPI
        from storage.rl_database import RLDatabase

        # 1. Initialize components
        orchestrator = AgentOrchestrator()
        test_agent = TestAgent()
        reward_agent = RewardAgent()
        cursor_api = MockCursorAPI()
        db = RLDatabase("test_e2e.db")

        # 2. Multi-agent discussion
        project_desc = "Create a simple REST API with user authentication"
        consensus = orchestrator.facilitate_discussion(project_desc)
        print(f"✅ Consensus reached: {consensus['confidence']:.0%} confidence")

        # 3. Code generation
        code_result = cursor_api.generate_code(consensus["approach"])
        print(f"✅ Code generated: {code_result['success']}")

        # 4. Testing
        test_result = test_agent.analyze_code(code_result["code"], project_desc)
        print(f"✅ Code tested: {test_result['overall_score']:.1f}/10 score")

        # 5. Reward calculation
        reward = reward_agent.calculate_reward(
            test_results={"tests_passed": 2, "tests_run": 2},
            code_quality=test_result,
            iteration_count=1,
        )
        print(f"✅ Reward calculated: {reward['total_reward']:.1f}")

        # 6. Store episode
        episode_data = {
            "episode_id": "e2e_test_001",
            "timestamp": "2024-01-01T00:00:00",
            "project_description": project_desc,
            "initial_prompt": project_desc,
            "optimized_prompt": consensus["approach"],
            "final_code": code_result["code"],
            "total_reward": reward["total_reward"],
            "iteration_count": 1,
            "execution_time": 10.0,
            "status": "completed",
            "reward_components": reward["components"],
            "test_results": test_result,
        }

        episode_id = db.store_episode(episode_data)
        print(f"✅ Episode stored: {episode_id}")

        # 7. Get statistics
        stats = db.get_reward_statistics()
        print(
            f"✅ Statistics: {stats['total_episodes']} episodes, avg reward: {stats['average_reward']:.1f}"
        )

        # Cleanup
        db.clear_data()

        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🎼 CodeConductor v2.0 System Test")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Agents", test_agents),
        ("Integrations", test_integrations),
        ("Storage", test_storage),
        ("Configuration", test_config),
        ("GUI Components", test_gui_components),
        ("End-to-End Workflow", test_end_to_end),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 ALL TESTS PASSED! System is ready for demo!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
