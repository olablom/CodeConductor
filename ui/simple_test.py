#!/usr/bin/env python3
"""
Simple test dashboard
"""

import streamlit as st

st.title("🧪 Simple Test Dashboard")

st.write("If you can see this, Streamlit is working!")

# Test basic functionality
st.header("Basic Tests")

# Test 1: Simple text
st.write("✅ Text rendering works")

# Test 2: Simple data
import pandas as pd

test_df = pd.DataFrame({"Name": ["Alice", "Bob", "Charlie"], "Age": [25, 30, 35]})
st.write("✅ Pandas works")
st.dataframe(test_df)

# Test 3: Simple chart
try:
    import plotly.express as px

    fig = px.bar(test_df, x="Name", y="Age", title="Test Chart")
    st.plotly_chart(fig)
    st.write("✅ Plotly works")
except Exception as e:
    st.error(f"❌ Plotly failed: {e}")

# Test 4: Database connection
try:
    import sqlite3
    from pathlib import Path

    db_path = Path("data/qtable.db")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM q_table")
        count = cursor.fetchone()[0]
        st.write(f"✅ Database works: {count} entries in q_table")
        conn.close()
    else:
        st.error("❌ Database file not found")
except Exception as e:
    st.error(f"❌ Database failed: {e}")

st.write("🎉 If you see all green checkmarks, everything works!")
