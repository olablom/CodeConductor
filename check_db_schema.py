#!/usr/bin/env python3
"""Check database schema"""

import sqlite3
from pathlib import Path


def check_database_schema():
    """Check existing database schemas"""

    data_dir = Path("data")

    # Check Q-table database
    qtable_db = data_dir / "qtable.db"
    if qtable_db.exists():
        print("📊 Q-table database schema:")
        conn = sqlite3.connect(qtable_db)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            print(f"\n  Table: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"    {col[1]} ({col[2]})")

        conn.close()
    else:
        print("❌ Q-table database not found")

    # Check metrics database
    metrics_db = data_dir / "metrics.db"
    if metrics_db.exists():
        print("\n📈 Metrics database schema:")
        conn = sqlite3.connect(metrics_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            print(f"\n  Table: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"    {col[1]} ({col[2]})")

        conn.close()
    else:
        print("❌ Metrics database not found")


if __name__ == "__main__":
    check_database_schema()
