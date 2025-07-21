#!/usr/bin/env python3
"""
MLOps Test - Verify monitoring setup
"""

import requests
import time


def test_gpu_service_metrics():
    """Test GPU service metrics endpoint"""
    print("🧪 Testing GPU Service Metrics...")

    try:
        # Test health endpoint
        response = requests.get("http://localhost:8007/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GPU Service Health: {data['status']}")
            print(f"✅ GPU Available: {data['gpu_available']}")
            print(f"✅ Device: {data['device']}")
            print(f"✅ Memory: {data['gpu_memory_gb']:.1f} GB")
        else:
            print(f"❌ GPU Service Health Failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ GPU Service Health Error: {str(e)}")
        return False

    try:
        # Test metrics endpoint
        response = requests.get("http://localhost:8007/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            print(f"✅ Metrics Endpoint: {len(metrics)} characters")
            if "codeconductor" in metrics:
                print("✅ CodeConductor metrics found")
            else:
                print("⚠️  No CodeConductor metrics found yet")
        else:
            print(f"❌ Metrics Endpoint Failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics Endpoint Error: {str(e)}")
        return False

    return True


def test_prometheus():
    """Test Prometheus connectivity"""
    print("\n🧪 Testing Prometheus...")

    try:
        # Test Prometheus health
        response = requests.get("http://localhost:9090/-/healthy", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus is healthy")
        else:
            print(f"❌ Prometheus Health Failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prometheus Health Error: {str(e)}")
        return False

    try:
        # Test targets
        response = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
        if response.status_code == 200:
            data = response.json()
            targets = data["data"]["activeTargets"]
            print(f"✅ Prometheus has {len(targets)} targets")

            # Check GPU service target
            gpu_target = None
            for target in targets:
                if "8007" in target["labels"]["instance"]:
                    gpu_target = target
                    break

            if gpu_target:
                print(f"✅ GPU Service Target: {gpu_target['health']}")
                if gpu_target["health"] == "up":
                    print("🎉 GPU Service metrics are flowing to Prometheus!")
                else:
                    print("⚠️  GPU Service target is down")
            else:
                print("❌ GPU Service target not found")
        else:
            print(f"❌ Prometheus Targets Failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prometheus Targets Error: {str(e)}")
        return False

    return True


def test_grafana():
    """Test Grafana connectivity"""
    print("\n🧪 Testing Grafana...")

    try:
        # Test Grafana health
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Grafana is healthy: {data.get('database', 'Unknown')}")
        else:
            print(f"❌ Grafana Health Failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Grafana Health Error: {str(e)}")
        return False

    return True


def generate_some_metrics():
    """Generate some test metrics"""
    print("\n🧪 Generating test metrics...")

    try:
        # Make some API calls to generate metrics
        for i in range(3):
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
                print(f"✅ Generated bandit metric {i + 1}: {data['selected_arm']}")
            else:
                print(f"❌ Bandit call {i + 1} failed: HTTP {response.status_code}")

        # Wait for metrics to be scraped
        print("⏳ Waiting for metrics to be scraped...")
        time.sleep(10)

    except Exception as e:
        print(f"❌ Metrics generation error: {str(e)}")


def main():
    """Run all MLOps tests"""
    print("🚀 CodeConductor MLOps Test")
    print("🎯 Testing monitoring setup and metrics flow")
    print("=" * 60)

    # Test GPU service
    gpu_success = test_gpu_service_metrics()

    # Test Prometheus
    prometheus_success = test_prometheus()

    # Test Grafana
    grafana_success = test_grafana()

    # Generate some metrics
    generate_some_metrics()

    # Final test
    print("\n" + "=" * 60)
    print("🏆 MLOPS TEST RESULTS")
    print("=" * 60)

    if gpu_success and prometheus_success and grafana_success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Your MLOps foundation is working!")
        print("🚀 Ready for real-time AI monitoring!")

        print("\n📊 Access your monitoring:")
        print("   Grafana: http://localhost:3000 (admin/codeconductor)")
        print("   Prometheus: http://localhost:9090")
        print("   GPU Service: http://localhost:8007/health")

        print("\n🎯 Next steps:")
        print("   1. Open Grafana and create dashboards")
        print("   2. Start your AI services to see metrics flow")
        print("   3. Build React frontend to consume metrics")

    else:
        print("⚠️  Some tests failed")
        if not gpu_success:
            print("❌ GPU service tests failed")
        if not prometheus_success:
            print("❌ Prometheus tests failed")
        if not grafana_success:
            print("❌ Grafana tests failed")


if __name__ == "__main__":
    main()
