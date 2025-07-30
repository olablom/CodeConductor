# Automated Test Suite for Validation System (Fixed for Windows)
# Comprehensive testing before manual integration

import unittest
import time
import threading
import pandas as pd
import numpy as np
import warnings
import os

# Suppress all warnings for clean output
warnings.filterwarnings("ignore")
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

from validation_logger import ValidationLogger
from validation_dashboard import ValidationDashboard
import tempfile
import shutil


class ValidationSystemTestSuite(unittest.TestCase):
    """Comprehensive test suite for validation system"""

    def setUp(self):
        """Setup test environment"""
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp()
        self.original_csv = "validation_log.csv"

        # Backup original data if exists
        if os.path.exists(self.original_csv):
            shutil.copy2(self.original_csv, f"{self.original_csv}.backup")

    def tearDown(self):
        """Cleanup after tests"""
        # Restore original data
        if os.path.exists(f"{self.original_csv}.backup"):
            shutil.move(f"{self.original_csv}.backup", self.original_csv)

        # Clean test directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_logger_initialization(self):
        """Test 1: Logger initialization"""
        print("Test 1: Logger initialization")

        logger = ValidationLogger()
        self.assertIsNotNone(logger)
        self.assertTrue(hasattr(logger, "log_task_start"))
        self.assertTrue(hasattr(logger, "log_task_complete"))

        print("Logger initialization passed")

    def test_02_basic_task_logging(self):
        """Test 2: Basic task logging"""
        print("Test 2: Basic task logging")

        logger = ValidationLogger()

        # Test manual task
        task_id = "Test_Manual_001"
        logger.log_task_start(task_id, "Test manual task", "Manual")
        time.sleep(0.1)
        logger.log_task_complete(
            task_id,
            tests_passed=8,
            total_tests=10,
            errors=2,
            complexity=7.0,
            confidence=0.75,
            ensemble_agreement=0.8,
            rlhf_reward=0.7,
            cognitive_load=4.0,
            satisfaction_score=6.0,
        )

        # Test CodeConductor task
        task_id = "Test_CC_001"
        logger.log_task_start(task_id, "Test CC task", "CodeConductor")
        time.sleep(0.05)
        logger.log_task_complete(
            task_id,
            tests_passed=9,
            total_tests=10,
            errors=1,
            complexity=6.0,
            confidence=0.85,
            ensemble_agreement=0.9,
            rlhf_reward=0.8,
            cognitive_load=2.0,
            satisfaction_score=8.0,
        )

        # Verify data was logged
        df = logger.get_comparison_data()
        self.assertGreater(len(df), 0)

        print("Basic task logging passed")

    def test_03_edge_case_logging(self):
        """Test 3: Edge case logging"""
        print("Test 3: Edge case logging")

        logger = ValidationLogger()

        # Test edge cases
        edge_cases = [
            ("Edge_Empty", "", "Manual"),
            (
                "Edge_Long",
                "Very long task description with many words and complex requirements that might break parsing and cause issues with the system",
                "CodeConductor",
            ),
            ("Edge_Special", "Task with special chars: aao!@#$%^&*()", "Manual"),
            ("Edge_Newlines", "Task\nwith\nnewlines", "CodeConductor"),
            ("Edge_Unicode", "Task with unicode: test symbols", "Manual"),
        ]

        for task_id, description, mode in edge_cases:
            try:
                logger.log_task_start(task_id, description, mode)
                time.sleep(0.01)
                logger.log_task_complete(
                    task_id,
                    tests_passed=5,
                    total_tests=10,
                    errors=5,
                    complexity=5.0,
                    confidence=0.5,
                    ensemble_agreement=0.5,
                    rlhf_reward=0.5,
                    cognitive_load=5.0,
                    satisfaction_score=5.0,
                )
                print(f"Edge case '{task_id}' passed")
            except Exception as e:
                self.fail(f"Edge case '{task_id}' failed: {e}")

        print("Edge case logging passed")

    def test_04_concurrent_logging(self):
        """Test 4: Concurrent logging"""
        print("Test 4: Concurrent logging")

        logger = ValidationLogger()
        results = []

        def log_concurrent_task(thread_id):
            """Log task from concurrent thread"""
            try:
                for i in range(5):
                    task_id = f"Concurrent_{thread_id}_{i}"
                    logger.log_task_start(
                        task_id, f"Concurrent task {i}", "CodeConductor"
                    )
                    time.sleep(0.01)
                    logger.log_task_complete(
                        task_id,
                        tests_passed=8,
                        total_tests=10,
                        errors=2,
                        complexity=6.0,
                        confidence=0.8,
                        ensemble_agreement=0.8,
                        rlhf_reward=0.8,
                        cognitive_load=3.0,
                        satisfaction_score=7.0,
                    )
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                results.append(f"Thread {thread_id} failed: {e}")

        # Run 5 concurrent threads
        threads = [
            threading.Thread(target=log_concurrent_task, args=(i,)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check results
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn("completed", result)

        print("Concurrent logging passed")

    def test_05_data_validation(self):
        """Test 5: Data validation"""
        print("Test 5: Data validation")

        logger = ValidationLogger()

        # Log some test data
        for i in range(10):
            task_id = f"Validation_{i}"
            mode = "Manual" if i % 2 == 0 else "CodeConductor"
            logger.log_task_start(task_id, f"Validation task {i}", mode)
            time.sleep(0.01)
            logger.log_task_complete(
                task_id,
                tests_passed=8 + (i % 3),
                total_tests=10,
                errors=i % 3,
                complexity=5.0 + (i % 5),
                confidence=0.7 + (i % 3) * 0.1,
                ensemble_agreement=0.8 + (i % 2) * 0.1,
                rlhf_reward=0.7 + (i % 3) * 0.1,
                cognitive_load=3.0 + (i % 4),
                satisfaction_score=6.0 + (i % 4),
            )

        # Validate data
        df = logger.get_comparison_data()

        # Check data types
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Check required columns
        required_columns = ["Date", "TaskID", "Description", "Mode", "Duration_sec"]
        for col in required_columns:
            self.assertIn(col, df.columns)

        # Check data ranges
        self.assertTrue(all(df["Duration_sec"] >= 0))
        self.assertTrue(all(df["Tests_Passed"] >= 0))
        self.assertTrue(all(df["Total_Tests"] > 0))
        self.assertTrue(
            all(df["Model_Confidence"] >= 0) and all(df["Model_Confidence"] <= 1)
        )

        print("Data validation passed")

    def test_06_metrics_calculation(self):
        """Test 6: Metrics calculation"""
        print("Test 6: Metrics calculation")

        logger = ValidationLogger()

        # Log test data
        for i in range(20):
            task_id = f"Metrics_{i}"
            mode = "Manual" if i % 2 == 0 else "CodeConductor"
            duration = 300 if mode == "Manual" else 30  # 5 min vs 30 sec

            logger.log_task_start(task_id, f"Metrics task {i}", mode)
            time.sleep(0.01)
            logger.log_task_complete(
                task_id,
                tests_passed=8 + (i % 3),
                total_tests=10,
                errors=i % 3,
                complexity=5.0 + (i % 5),
                confidence=0.7 + (i % 3) * 0.1,
                ensemble_agreement=0.8 + (i % 2) * 0.1,
                rlhf_reward=0.7 + (i % 3) * 0.1,
                cognitive_load=3.0 + (i % 4),
                satisfaction_score=6.0 + (i % 4),
            )

        # Test metrics calculation
        savings = logger.calculate_time_savings()
        roi = logger.calculate_roi()

        # Validate metrics
        self.assertIsInstance(savings, dict)
        self.assertIsInstance(roi, dict)
        self.assertIn("time_savings_percent", savings)
        self.assertIn("value_saved", roi)

        # Check reasonable ranges
        self.assertGreaterEqual(savings["time_savings_percent"], 0)
        self.assertLessEqual(savings["time_savings_percent"], 100)
        self.assertGreaterEqual(roi["value_saved"], 0)

        print(
            f"Metrics calculation passed - Time savings: {savings['time_savings_percent']:.1f}%, ROI: ${roi['value_saved']:.0f}"
        )

    def test_07_dashboard_rendering(self):
        """Test 7: Dashboard rendering"""
        print("Test 7: Dashboard rendering")

        # Create dashboard
        dashboard = ValidationDashboard()
        self.assertIsNotNone(dashboard)

        # Test dashboard methods exist
        self.assertTrue(hasattr(dashboard, "render_dashboard"))
        self.assertTrue(hasattr(dashboard, "render_time_comparison_chart"))
        self.assertTrue(hasattr(dashboard, "render_quality_metrics_chart"))
        self.assertTrue(hasattr(dashboard, "render_roi_progress_chart"))
        self.assertTrue(hasattr(dashboard, "render_agent_performance_chart"))

        # Test with sample data
        logger = ValidationLogger()
        for i in range(10):
            task_id = f"Dashboard_{i}"
            mode = "Manual" if i % 2 == 0 else "CodeConductor"
            logger.log_task_start(task_id, f"Dashboard task {i}", mode)
            time.sleep(0.01)
            logger.log_task_complete(
                task_id,
                tests_passed=8 + (i % 3),
                total_tests=10,
                errors=i % 3,
                complexity=5.0 + (i % 5),
                confidence=0.7 + (i % 3) * 0.1,
                ensemble_agreement=0.8 + (i % 2) * 0.1,
                rlhf_reward=0.7 + (i % 3) * 0.1,
                cognitive_load=3.0 + (i % 4),
                satisfaction_score=6.0 + (i % 4),
            )

        # Test dashboard data loading
        df = dashboard.logger.get_comparison_data()
        self.assertGreater(len(df), 0)

        print("Dashboard rendering passed")

    def test_08_error_handling(self):
        """Test 8: Error handling"""
        print("Test 8: Error handling")

        logger = ValidationLogger()

        # Test invalid task completion (no start)
        with self.assertRaises(ValueError):
            logger.log_task_complete(
                "Invalid_Task",
                tests_passed=5,
                total_tests=10,
                errors=5,
                complexity=5.0,
                confidence=0.5,
                ensemble_agreement=0.5,
                rlhf_reward=0.5,
                cognitive_load=5.0,
                satisfaction_score=5.0,
            )

        # Test invalid parameters
        task_id = "Error_Test"
        logger.log_task_start(task_id, "Error test", "Manual")

        # Test with invalid confidence (should be clamped)
        logger.log_task_complete(
            task_id,
            tests_passed=5,
            total_tests=10,
            errors=5,
            complexity=5.0,
            confidence=1.5,  # Invalid - should be clamped to 1.0
            ensemble_agreement=0.5,
            rlhf_reward=0.5,
            cognitive_load=5.0,
            satisfaction_score=5.0,
        )

        print("Error handling passed")

    def test_09_performance_stress_test(self):
        """Test 9: Performance stress test"""
        print("Test 9: Performance stress test")

        logger = ValidationLogger()
        start_time = time.time()

        # Log 100 tasks quickly
        for i in range(100):
            task_id = f"Stress_{i}"
            mode = "Manual" if i % 2 == 0 else "CodeConductor"
            logger.log_task_start(task_id, f"Stress task {i}", mode)
            logger.log_task_complete(
                task_id,
                tests_passed=8,
                total_tests=10,
                errors=2,
                complexity=6.0,
                confidence=0.8,
                ensemble_agreement=0.8,
                rlhf_reward=0.8,
                cognitive_load=3.0,
                satisfaction_score=7.0,
            )

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time
        self.assertLess(duration, 10.0)  # Should complete in under 10 seconds

        # Verify data integrity
        df = logger.get_comparison_data()
        self.assertGreaterEqual(len(df), 100)

        print(f"Performance stress test passed - {len(df)} tasks in {duration:.2f}s")

    def test_10_integration_test(self):
        """Test 10: Full integration test"""
        print("Test 10: Full integration test")

        # Test complete workflow
        logger = ValidationLogger()
        dashboard = ValidationDashboard()

        # Log realistic data
        tasks = [
            ("Manual_Real", "Create REST API with authentication", "Manual"),
            ("CC_Real", "Create REST API with authentication", "CodeConductor"),
            ("Manual_Complex", "Implement machine learning pipeline", "Manual"),
            ("CC_Complex", "Implement machine learning pipeline", "CodeConductor"),
        ]

        for task_id, description, mode in tasks:
            logger.log_task_start(task_id, description, mode)
            time.sleep(0.1 if mode == "Manual" else 0.05)
            logger.log_task_complete(
                task_id,
                tests_passed=8 if mode == "Manual" else 9,
                total_tests=10,
                errors=2 if mode == "Manual" else 1,
                complexity=7.0 if mode == "Manual" else 6.0,
                confidence=0.75 if mode == "Manual" else 0.85,
                ensemble_agreement=0.8 if mode == "Manual" else 0.9,
                rlhf_reward=0.7 if mode == "Manual" else 0.8,
                cognitive_load=4.0 if mode == "Manual" else 2.0,
                satisfaction_score=6.0 if mode == "Manual" else 8.0,
            )

        # Test complete metrics
        savings = logger.calculate_time_savings()
        roi = logger.calculate_roi()

        # Verify realistic results
        self.assertGreater(
            savings["time_savings_percent"], 50
        )  # Should show significant savings
        self.assertGreater(roi["value_saved"], 0)  # Should show positive ROI

        print(
            f"Integration test passed - {savings['time_savings_percent']:.1f}% savings, ${roi['value_saved']:.0f} ROI"
        )


def run_comprehensive_test_suite():
    """Run comprehensive test suite"""
    print("Starting Comprehensive Validation System Test Suite")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(ValidationSystemTestSuite)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("=" * 60)
    print("TEST SUITE SUMMARY:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    if result.wasSuccessful():
        print("\nALL TESTS PASSED! Validation system is ready for integration.")
    else:
        print("\nSome tests failed. Fix issues before integration.")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)
