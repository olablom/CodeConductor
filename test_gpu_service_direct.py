#!/usr/bin/env python3
"""
Direct GPU Service Test
"""

import sys

sys.path.append("services/gpu_data_service")

# Import the app directly
from app.main import app


def test_app_routes():
    """Test app routes directly"""
    print("🧪 Testing GPU service routes directly...")

    routes = []
    for route in app.routes:
        if hasattr(route, "path"):
            routes.append(route.path)

    print("📋 Available routes:")
    for route in sorted(routes):
        print(f"  {route}")

    # Check if metrics route exists
    if "/metrics" in routes:
        print("✅ Metrics route found!")
    else:
        print("❌ Metrics route NOT found!")

        # Let's check what's in the app
        print("\n🔍 Checking app structure...")
        print(f"App routes count: {len(routes)}")

        # Check if there are any issues with the metrics endpoint
        try:

            print("✅ Metrics function imported successfully")
        except Exception as e:
            print(f"❌ Metrics function import error: {e}")


if __name__ == "__main__":
    test_app_routes()
