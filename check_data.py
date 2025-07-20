#!/usr/bin/env python3
"""Check database data"""

import sqlite3
from pathlib import Path


def check_data():
    """Check if databases contain data"""

    print("🔍 Checking database data...")

    # Check Q-table
    qtable_db = Path("data/qtable.db")
    if qtable_db.exists():
        conn = sqlite3.connect(qtable_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM q_table")
        q_count = cursor.fetchone()[0]
        print(f"📊 Q-table entries: {q_count}")

        cursor.execute("SELECT COUNT(*) FROM learning_metrics")
        lm_count = cursor.fetchone()[0]
        print(f"📈 Learning metrics: {lm_count}")
        conn.close()
    else:
        print("❌ Q-table database not found")

    # Check metrics
    metrics_db = Path("data/metrics.db")
    if metrics_db.exists():
        conn = sqlite3.connect(metrics_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metrics")
        m_count = cursor.fetchone()[0]
        print(f"📊 Metrics entries: {m_count}")
        conn.close()
    else:
        print("❌ Metrics database not found")

    # Check RL history
    rl_db = Path("data/rl_history.db")
    if rl_db.exists():
        conn = sqlite3.connect(rl_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes")
        e_count = cursor.fetchone()[0]
        print(f"🤖 RL episodes: {e_count}")

        cursor.execute("SELECT COUNT(*) FROM reward_components")
        rc_count = cursor.fetchone()[0]
        print(f"🎯 Reward components: {rc_count}")
        conn.close()
    else:
        print("❌ RL history database not found")


if __name__ == "__main__":
    check_data()
