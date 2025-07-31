# 🎯 **VALIDATION LOGGER - Empirical Data Collection System**

import time
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit as st


class ValidationLogger:
    """Comprehensive metrics logging for CodeConductor empirical validation"""

    def __init__(self, log_file: str = "validation_log.csv"):
        self.log_file = log_file
        self.setup_log_file()

    def setup_log_file(self):
        """Initialize CSV log file with headers"""
        headers = [
            "Date",
            "TaskID",
            "Description",
            "Mode",
            "Duration_sec",
            "Tests_Passed",
            "Total_Tests",
            "Errors",
            "Code_Complexity",
            "Model_Confidence",
            "Ensemble_Agreement",
            "RLHF_Reward",
            "Context_Switches",
            "Cognitive_Load",
            "Satisfaction_Score",
        ]

        try:
            with open(self.log_file, "r") as f:
                # File exists, don't overwrite
                pass
        except FileNotFoundError:
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_task_start(self, task_id: str, description: str, mode: str) -> float:
        """Log task start and return start timestamp"""
        start_time = time.time()
        st.session_state[f"{task_id}_start"] = start_time
        st.session_state[f"{task_id}_mode"] = mode
        st.session_state[f"{task_id}_description"] = description
        return start_time

    def log_task_complete(self, task_id: str, **metrics) -> Dict[str, Any]:
        """Log task completion with all metrics"""
        if f"{task_id}_start" not in st.session_state:
            raise ValueError(f"No start time found for task {task_id}")

        start_time = st.session_state[f"{task_id}_start"]
        end_time = time.time()
        duration = end_time - start_time
        mode = st.session_state.get(f"{task_id}_mode", "Unknown")
        description = st.session_state.get(f"{task_id}_description", "")

        # Prepare log entry
        log_entry = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "TaskID": task_id,
            "Description": description,
            "Mode": mode,
            "Duration_sec": f"{duration:.2f}",
            "Tests_Passed": metrics.get("tests_passed", 0),
            "Total_Tests": metrics.get("total_tests", 0),
            "Errors": metrics.get("errors", 0),
            "Code_Complexity": metrics.get("complexity", 0),
            "Model_Confidence": metrics.get("confidence", 0),
            "Ensemble_Agreement": metrics.get("agreement", 0),
            "RLHF_Reward": metrics.get("reward", 0),
            "Context_Switches": metrics.get("context_switches", 0),
            "Cognitive_Load": metrics.get("cognitive_load", 0),
            "Satisfaction_Score": metrics.get("satisfaction", 0),
        }

        # Write to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)

        # Clean up session state
        for key in [f"{task_id}_start", f"{task_id}_mode", f"{task_id}_description"]:
            if key in st.session_state:
                del st.session_state[key]

        return log_entry

    def get_comparison_data(self) -> pd.DataFrame:
        """Load and prepare data for Manual vs CodeConductor comparison"""
        try:
            df = pd.read_csv(self.log_file)
            return df
        except FileNotFoundError:
            return pd.DataFrame()

    def calculate_time_savings(self) -> Dict[str, float]:
        """Calculate time savings between Manual and CodeConductor modes"""
        df = self.get_comparison_data()
        if df.empty:
            return {"time_savings_percent": 0, "total_hours_saved": 0}

        # Group by TaskID and calculate differences
        manual_times = df[df["Mode"] == "Manual"]["Duration_sec"].astype(float)
        cc_times = df[df["Mode"] == "CodeConductor"]["Duration_sec"].astype(float)

        if len(manual_times) == 0 or len(cc_times) == 0:
            return {"time_savings_percent": 0, "total_hours_saved": 0}

        # Calculate savings
        total_manual_time = manual_times.sum()
        total_cc_time = cc_times.sum()
        time_savings = (total_manual_time - total_cc_time) / total_manual_time * 100
        hours_saved = (total_manual_time - total_cc_time) / 3600

        return {
            "time_savings_percent": time_savings,
            "total_hours_saved": hours_saved,
            "manual_avg_time": manual_times.mean(),
            "cc_avg_time": cc_times.mean(),
        }

    def calculate_roi(self, hourly_rate: float = 50.0) -> Dict[str, float]:
        """Calculate ROI based on time savings"""
        savings = self.calculate_time_savings()
        value_saved = savings["total_hours_saved"] * hourly_rate

        return {
            "hours_saved": savings["total_hours_saved"],
            "value_saved": value_saved,
            "hourly_rate": hourly_rate,
            "time_savings_percent": savings["time_savings_percent"],
        }


# Manual task timing helper
def manual_task_timer():
    """Helper for timing manual tasks"""
    task_id = input("Enter Task ID: ")
    description = input("Enter task description: ")

    print(f"Starting manual task: {description}")
    input("Press Enter to start timing...")
    start_time = time.time()

    input("Press Enter when task is complete...")
    end_time = time.time()

    duration = end_time - start_time

    # Get quality metrics
    tests_passed = int(input("Number of tests passed: ") or 0)
    total_tests = int(input("Total number of tests: ") or 0)
    errors = int(input("Number of errors: ") or 0)
    cognitive_load = int(input("Cognitive load (1-10): ") or 5)
    satisfaction = int(input("Satisfaction score (1-10): ") or 5)

    # Log the manual task
    logger = ValidationLogger()
    logger.log_task_start(task_id, description, "Manual")

    log_entry = logger.log_task_complete(
        task_id,
        tests_passed=tests_passed,
        total_tests=total_tests,
        errors=errors,
        cognitive_load=cognitive_load,
        satisfaction=satisfaction,
    )

    print(f"Manual task logged: {duration:.2f} seconds")
    return log_entry


# Integration with CodeConductor
def log_codeconductor_task(task_id: str, description: str, **metrics):
    """Log CodeConductor task with all metrics"""
    logger = ValidationLogger()
    logger.log_task_start(task_id, description, "CodeConductor")
    return logger.log_task_complete(task_id, **metrics)


# Example usage:
if __name__ == "__main__":
    # Test manual timing
    print("Manual Task Timer Test")
    manual_task_timer()

    # Test CodeConductor logging
    print("\nCodeConductor Task Test")
    log_entry = log_codeconductor_task(
        "CC_Test_01",
        "Test CodeConductor task",
        tests_passed=5,
        total_tests=5,
        errors=0,
        complexity=3,
        confidence=0.85,
        agreement=0.92,
        reward=0.78,
        context_switches=2,
        cognitive_load=3,
        satisfaction=8,
    )
    print(f"CodeConductor task logged: {log_entry}")
