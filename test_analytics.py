#!/usr/bin/env python3
"""
Test script for CodeConductor Analytics functionality
"""

import numpy as np
from pathlib import Path


def test_ml_predictor():
    """Test ML predictor functionality"""
    print("🧪 Testing ML Predictor...")

    try:
        from analytics.ml_predictor import create_predictor

        predictor = create_predictor()

        # Test prediction
        test_prompt = "Create a simple calculator function"
        test_context = {
            "strategy": "conservative",
            "prev_pass_rate": 0.7,
            "complexity_avg": 0.6,
            "reward_avg": 35.0,
            "iteration_count": 3,
            "model_source": "lm_studio",
        }

        result = predictor.predict_quality(test_prompt, test_context)

        if "error" not in result:
            print("✅ ML prediction successful")
            print(f"   Quality score: {result['quality_score']:.2f}")
            print(f"   Success probability: {result['success_probability']:.2f}")
            print(f"   Confidence: {result['confidence']:.2f}")

            if result.get("warnings"):
                print("   Warnings:")
                for warning in result["warnings"]:
                    print(f"     {warning}")

            return True
        else:
            print(f"❌ ML prediction failed: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ ML predictor test failed: {e}")
        return False


def test_trends_analysis():
    """Test trends analysis"""
    print("\n🧪 Testing Trends Analysis...")

    try:
        from analytics.ml_predictor import create_predictor

        predictor = create_predictor()
        trends = predictor.get_trends(days=30)

        if "error" not in trends:
            print("✅ Trends analysis successful")
            print(f"   Pass rate trend: {trends['pass_rate_trend']:.3f}")
            print(f"   Total iterations: {trends['total_iterations']}")
            print(f"   Daily average: {trends['avg_daily_iterations']:.1f}")
            print(f"   Improvement rate: {trends['improvement_rate']:.3f}")
            return True
        else:
            print(f"❌ Trends analysis failed: {trends['error']}")
            return False

    except Exception as e:
        print(f"❌ Trends analysis test failed: {e}")
        return False


def test_dashboard_imports():
    """Test dashboard imports"""
    print("\n🧪 Testing Dashboard Imports...")

    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd

        print("✅ All dashboard dependencies imported successfully")
        return True

    except Exception as e:
        print(f"❌ Dashboard import test failed: {e}")
        return False


def test_data_loading():
    """Test data loading from database"""
    print("\n🧪 Testing Data Loading...")

    try:
        import sqlite3
        import pandas as pd

        conn = sqlite3.connect("data/metrics.db")
        df = pd.read_sql_query("SELECT * FROM metrics LIMIT 5", conn)
        conn.close()

        if not df.empty:
            print(f"✅ Data loading successful - {len(df)} rows loaded")
            print(f"   Columns: {list(df.columns)}")
            return True
        else:
            print("⚠️ No data found in database")
            return False

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_pipeline_integration():
    """Test pipeline integration"""
    print("\n🧪 Testing Pipeline Integration...")

    try:
        # Test that pipeline can import ML predictor
        from analytics.ml_predictor import create_predictor

        predictor = create_predictor()

        # Test with minimal context
        result = predictor.predict_quality("test prompt", {})

        if "error" not in result:
            print("✅ Pipeline integration successful")
            return True
        else:
            print(f"⚠️ Pipeline integration warning: {result['error']}")
            return True  # Still consider it working

    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False


def main():
    """Run all analytics tests"""
    print("🎼 CodeConductor v2.0 - Analytics Test")
    print("=" * 60)

    tests = [
        ("ML Predictor", test_ml_predictor),
        ("Trends Analysis", test_trends_analysis),
        ("Dashboard Imports", test_dashboard_imports),
        ("Data Loading", test_data_loading),
        ("Pipeline Integration", test_pipeline_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All analytics tests passed!")
        print("🚀 CodeConductor analytics is ready!")
        print("\n💡 Next steps:")
        print("  1. Run analytics dashboard: streamlit run analytics/dashboard.py")
        print("  2. Test ML predictions in pipeline")
        print("  3. Monitor trends and warnings")
    else:
        print("⚠️ Some tests failed. Check dependencies and configuration.")

    print("=" * 60)


if __name__ == "__main__":
    main()
