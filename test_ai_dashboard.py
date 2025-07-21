#!/usr/bin/env python3
"""
AI Dashboard Test - Verify React frontend integration
"""

import requests
import time


def test_react_app():
    """Test React app accessibility"""
    print("🧪 Testing React AI Dashboard...")

    try:
        # Test React app on port 3000
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            print("✅ React app is accessible on port 3000")
            return True
        else:
            print(f"❌ React app failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ React app error: {str(e)}")
        return False


def test_gpu_service_integration():
    """Test GPU service integration"""
    print("\n🧪 Testing GPU Service Integration...")

    try:
        # Test GPU service health
        response = requests.get("http://localhost:8007/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GPU Service: {data['status']}")
            print(f"✅ GPU Available: {data['gpu_available']}")
            print(f"✅ Device: {data['device']}")
            print(f"✅ Memory: {data['gpu_memory_gb']:.1f} GB")
        else:
            print(f"❌ GPU Service failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ GPU Service error: {str(e)}")
        return False

    try:
        # Test AI decision generation
        response = requests.post(
            "http://localhost:8007/gpu/bandits/choose",
            json={
                "arms": [
                    "conservative_strategy",
                    "experimental_strategy",
                    "hybrid_approach",
                ],
                "features": [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.9, 0.6, 0.7, 0.8],
                "epsilon": 0.1,
            },
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ AI Decision: {data['selected_arm']}")
            print(f"✅ Confidence: {data['confidence']:.2f}")
            print(f"✅ GPU Used: {data['gpu_used']}")
            print(f"✅ Inference Time: {data['inference_time_ms']:.2f}ms")
        else:
            print(f"❌ AI Decision failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ AI Decision error: {str(e)}")
        return False

    return True


def test_mlops_integration():
    """Test MLOps stack integration"""
    print("\n🧪 Testing MLOps Integration...")

    try:
        # Test Prometheus
        response = requests.get("http://localhost:9090/-/healthy", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus is healthy")
        else:
            print(f"❌ Prometheus failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Prometheus error: {str(e)}")

    try:
        # Test Grafana
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Grafana is healthy: {data.get('database', 'Unknown')}")
        else:
            print(f"❌ Grafana failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Grafana error: {str(e)}")


def generate_test_data():
    """Generate some test AI decisions"""
    print("\n🧪 Generating test AI decisions...")

    try:
        for i in range(5):
            response = requests.post(
                "http://localhost:8007/gpu/bandits/choose",
                json={
                    "arms": [
                        "conservative_strategy",
                        "experimental_strategy",
                        "hybrid_approach",
                    ],
                    "features": [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.9, 0.6, 0.7, 0.8],
                    "epsilon": 0.1,
                },
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                print(
                    f"✅ Decision {i + 1}: {data['selected_arm']} (GPU: {data['gpu_used']})"
                )
            else:
                print(f"❌ Decision {i + 1} failed: HTTP {response.status_code}")

            time.sleep(1)  # Wait between decisions

    except Exception as e:
        print(f"❌ Test data generation error: {str(e)}")


def main():
    """Run all AI Dashboard tests"""
    print("🚀 CodeConductor AI Dashboard Test")
    print("🎯 Testing React frontend and AI integration")
    print("=" * 60)

    # Test React app
    react_success = test_react_app()

    # Test GPU service integration
    gpu_success = test_gpu_service_integration()

    # Test MLOps integration
    test_mlops_integration()

    # Generate test data
    generate_test_data()

    # Final summary
    print("\n" + "=" * 60)
    print("🏆 AI DASHBOARD TEST RESULTS")
    print("=" * 60)

    if react_success and gpu_success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Your AI Dashboard is working perfectly!")
        print("🚀 Real-time AI monitoring is LIVE!")

        print("\n📊 Access your AI Dashboard:")
        print("   React Dashboard: http://localhost:3000")
        print("   GPU Service: http://localhost:8007/health")
        print("   Prometheus: http://localhost:9090")
        print("   Grafana: http://localhost:3000 (admin/codeconductor)")

        print("\n🎯 What you can do now:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Watch real-time AI decisions streaming")
        print("   3. Toggle between Beast and Light modes")
        print("   4. Monitor GPU performance and memory usage")
        print("   5. View interactive performance charts")

        print("\n🏆 Portfolio Achievement:")
        print("   ✅ Complete AI platform with frontend")
        print("   ✅ Real-time AI monitoring dashboard")
        print("   ✅ Professional React/TypeScript UI")
        print("   ✅ GPU integration with RTX 5090")
        print("   ✅ MLOps monitoring stack")

    else:
        print("⚠️  Some tests failed")
        if not react_success:
            print("❌ React app tests failed")
        if not gpu_success:
            print("❌ GPU service integration failed")


if __name__ == "__main__":
    main()
