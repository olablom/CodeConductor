#!/usr/bin/env python3
"""
Test script for CodeConductor Dashboard

Generates sample data and tests dashboard functionality.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import random


def create_sample_data():
    """Create sample data for dashboard testing"""

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Sample Q-table data
    create_sample_qtable(data_dir / "qtable.db")

    # Sample metrics data
    create_sample_metrics(data_dir / "metrics.db")

    # Sample RL history data
    create_sample_rl_history(data_dir / "rl_history.db")

    print("✅ Sample data created successfully!")


def create_sample_qtable(db_path: Path):
    """Create sample Q-table data"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS q_table (
            state_hash TEXT,
            state_data TEXT,
            action_data TEXT,
            q_value REAL DEFAULT 0.0,
            visit_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (state_hash)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_metrics (
            episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_hash TEXT,
            action_hash TEXT,
            reward REAL,
            next_state_hash TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Generate sample Q-table entries
    states = [
        {"task_type": "api", "complexity": "low", "language": "python"},
        {"task_type": "api", "complexity": "medium", "language": "python"},
        {"task_type": "api", "complexity": "high", "language": "python"},
        {"task_type": "web", "complexity": "low", "language": "javascript"},
        {"task_type": "web", "complexity": "medium", "language": "javascript"},
        {"task_type": "web", "complexity": "high", "language": "javascript"},
    ]

    actions = [
        {"agent_combination": "codegen+reviewer", "prompt_strategy": "detailed"},
        {"agent_combination": "architect+codegen", "prompt_strategy": "concise"},
        {"agent_combination": "all_agents", "prompt_strategy": "iterative"},
    ]

    # Insert Q-table data
    for state in states:
        for action in actions:
            state_hash = (
                f"{state['task_type']}_{state['complexity']}_{state['language']}"
            )
            state_data = json.dumps(state)
            action_data = json.dumps(action)
            q_value = random.uniform(-1.0, 2.0)
            visit_count = random.randint(1, 50)

            cursor.execute(
                """
                INSERT OR REPLACE INTO q_table (state_hash, state_data, action_data, q_value, visit_count)
                VALUES (?, ?, ?, ?, ?)
            """,
                (state_hash, state_data, action_data, q_value, visit_count),
            )

    # Insert learning metrics
    for episode in range(1, 51):
        state = random.choice(states)
        action = random.choice(actions)
        reward = random.uniform(-0.5, 1.5)

        cursor.execute(
            """
            INSERT INTO learning_metrics (episode_id, state_hash, action_hash, reward)
            VALUES (?, ?, ?, ?)
        """,
            (episode, json.dumps(state), json.dumps(action), reward),
        )

    conn.commit()
    conn.close()

    print(f"✅ Q-table data created: {db_path}")


def create_sample_metrics(db_path: Path):
    """Create sample pipeline metrics data"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER,
            timestamp DATETIME,
            reward REAL,
            pass_rate REAL,
            complexity REAL,
            arm_selected TEXT,
            features TEXT,
            model_source TEXT,
            blocked INTEGER,
            block_reasons TEXT,
            optimizer_state TEXT,
            optimizer_action TEXT
        )
    """)

    # Generate sample metrics
    model_sources = ["mock", "local", "cloud"]
    optimizer_actions = ["explore", "exploit", "random"]

    for iteration in range(1, 101):
        reward = random.uniform(-0.5, 1.5)
        pass_rate = random.uniform(0.0, 1.0)
        complexity = random.uniform(1.0, 10.0)
        blocked = random.choice([0, 1])

        cursor.execute(
            """
            INSERT INTO metrics (
                iteration, reward, pass_rate, complexity, blocked,
                model_source, optimizer_action, timestamp, arm_selected, features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                iteration,
                reward,
                pass_rate,
                complexity,
                blocked,
                random.choice(model_sources),
                random.choice(optimizer_actions),
                datetime.now() - timedelta(minutes=100 - iteration),
                f"arm_{random.randint(1, 5)}",
                json.dumps({"feature1": random.random(), "feature2": random.random()}),
            ),
        )

    conn.commit()
    conn.close()

    print(f"✅ Metrics data created: {db_path}")


def create_sample_rl_history(db_path: Path):
    """Create sample RL history data"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            project_description TEXT,
            initial_prompt TEXT,
            optimized_prompt TEXT,
            final_code TEXT,
            total_reward REAL,
            iteration_count INTEGER,
            execution_time REAL,
            status TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reward_components (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            component_name TEXT NOT NULL,
            reward_value REAL NOT NULL,
            metadata TEXT,
            FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
        )
    """)

    # Generate sample episodes
    for episode in range(1, 21):
        episode_id = f"episode_{episode:03d}"
        total_reward = random.uniform(-1.0, 2.0)
        execution_time = random.uniform(10.0, 60.0)

        cursor.execute(
            """
            INSERT INTO episodes (
                episode_id, timestamp, total_reward, iteration_count, 
                execution_time, status
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                episode_id,
                datetime.now().isoformat(),
                total_reward,
                random.randint(5, 20),
                execution_time,
                "completed",
            ),
        )

        # Add reward components
        components = [
            ("code_quality", random.uniform(-0.5, 1.0)),
            ("test_coverage", random.uniform(0.0, 1.0)),
            ("complexity", random.uniform(-0.3, 0.5)),
            ("safety_score", random.uniform(0.5, 1.0)),
        ]

        for component_name, reward_value in components:
            cursor.execute(
                """
                INSERT INTO reward_components (
                    episode_id, component_name, reward_value
                ) VALUES (?, ?, ?)
            """,
                (episode_id, component_name, reward_value),
            )

    conn.commit()
    conn.close()

    print(f"✅ RL history data created: {db_path}")


def test_dashboard_imports():
    """Test that dashboard can be imported"""
    try:
        import sys

        sys.path.append("ui")

        from dashboard import DashboardDataLoader, DashboardVisualizer

        print("✅ Dashboard imports successful!")
        return True

    except Exception as e:
        print(f"❌ Dashboard import failed: {e}")
        return False


def main():
    """Main test function"""

    print("🧪 Testing CodeConductor Dashboard...")

    # Test imports
    if not test_dashboard_imports():
        return

    # Create sample data
    create_sample_data()

    print("\n🎉 Dashboard test completed!")
    print("\nTo run the dashboard:")
    print("  streamlit run ui/dashboard.py")

    print("\nSample data includes:")
    print("  - 18 Q-table entries (6 states × 3 actions)")
    print("  - 50 learning episodes")
    print("  - 100 pipeline iterations")
    print("  - 20 RL episodes with reward components")


if __name__ == "__main__":
    main()
