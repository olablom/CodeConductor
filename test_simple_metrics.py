#!/usr/bin/env python3
"""
Simple Metrics Test
"""

import requests


def test_metrics():
    """Test metrics endpoint"""
    print("🧪 Testing metrics endpoint...")

    # Test health first
    try:
        response = requests.get("http://localhost:8007/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint works")
        else:
            print(f"❌ Health failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health error: {e}")
        return

    # Test metrics
    try:
        response = requests.get("http://localhost:8007/metrics", timeout=5)
        print(f"📊 Metrics response: {response.status_code}")
        if response.status_code == 200:
            content = response.text
            print(f"✅ Metrics endpoint works: {len(content)} characters")
            if "codeconductor" in content:
                print("✅ CodeConductor metrics found!")
            else:
                print("⚠️  No CodeConductor metrics yet")
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Metrics error: {e}")


if __name__ == "__main__":
    test_metrics()
