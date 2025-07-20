#!/usr/bin/env python3
"""
Debug Dashboard - Simplified version to identify issues
"""

import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

st.set_page_config(page_title="Debug Dashboard", page_icon="🐛", layout="wide")

st.title("🐛 Debug Dashboard")

# Check if data files exist
st.header("📁 File Check")
data_dir = Path("data")

files = ["qtable.db", "metrics.db", "rl_history.db"]
for file in files:
    file_path = data_dir / file
    if file_path.exists():
        st.success(f"✅ {file} exists ({file_path.stat().st_size} bytes)")
    else:
        st.error(f"❌ {file} missing")

# Try to load data
st.header("📊 Data Loading Test")

try:
    # Load Q-table
    if (data_dir / "qtable.db").exists():
        conn = sqlite3.connect(data_dir / "qtable.db")
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        st.write("Tables in qtable.db:", [t[0] for t in tables])

        # Check q_table data
        cursor.execute("SELECT COUNT(*) FROM q_table")
        q_count = cursor.fetchone()[0]
        st.write(f"Q-table entries: {q_count}")

        if q_count > 0:
            cursor.execute("SELECT * FROM q_table LIMIT 3")
            q_data = cursor.fetchall()
            st.write("Sample Q-table data:", q_data)

        conn.close()
    else:
        st.error("qtable.db not found")

except Exception as e:
    st.error(f"Error loading Q-table: {e}")

try:
    # Load metrics
    if (data_dir / "metrics.db").exists():
        conn = sqlite3.connect(data_dir / "metrics.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM metrics")
        m_count = cursor.fetchone()[0]
        st.write(f"Metrics entries: {m_count}")

        if m_count > 0:
            cursor.execute("SELECT * FROM metrics LIMIT 3")
            m_data = cursor.fetchall()
            st.write("Sample metrics data:", m_data)

        conn.close()
    else:
        st.error("metrics.db not found")

except Exception as e:
    st.error(f"Error loading metrics: {e}")

try:
    # Load RL history
    if (data_dir / "rl_history.db").exists():
        conn = sqlite3.connect(data_dir / "rl_history.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM episodes")
        e_count = cursor.fetchone()[0]
        st.write(f"RL episodes: {e_count}")

        if e_count > 0:
            cursor.execute("SELECT * FROM episodes LIMIT 3")
            e_data = cursor.fetchall()
            st.write("Sample episodes data:", e_data)

        conn.close()
    else:
        st.error("rl_history.db not found")

except Exception as e:
    st.error(f"Error loading RL history: {e}")

# Test imports
st.header("📦 Import Test")
try:
    import plotly.graph_objects as go

    st.success("✅ plotly imported successfully")
except Exception as e:
    st.error(f"❌ plotly import failed: {e}")

try:
    import numpy as np

    st.success("✅ numpy imported successfully")
except Exception as e:
    st.error(f"❌ numpy import failed: {e}")

# Simple visualization test
st.header("🎨 Visualization Test")
if st.button("Test Simple Chart"):
    try:
        import plotly.express as px

        # Create simple test data
        test_data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 4, 2, 5, 3]})

        fig = px.line(test_data, x="x", y="y", title="Test Chart")
        st.plotly_chart(fig)
        st.success("✅ Chart created successfully!")

    except Exception as e:
        st.error(f"❌ Chart creation failed: {e}")

st.header("🔧 System Info")
st.write(f"Python version: {st.__version__}")
st.write(f"Streamlit version: {st.__version__}")
st.write(f"Working directory: {Path.cwd()}")
