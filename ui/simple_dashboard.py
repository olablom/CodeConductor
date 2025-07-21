#!/usr/bin/env python3
"""
Simple CodeConductor Dashboard - Core functionality only
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from pathlib import Path

st.set_page_config(page_title="CodeConductor Dashboard", page_icon="🎼", layout="wide")

st.title("🎼 CodeConductor Dashboard")


# Simple data loader
def load_data():
    """Load data from databases"""
    data = {}

    # Load Q-table
    try:
        if Path("data/qtable.db").exists():
            conn = sqlite3.connect("data/qtable.db")
            q_df = pd.read_sql_query("SELECT * FROM q_table", conn)
            data["q_table"] = q_df
            conn.close()
        else:
            data["q_table"] = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Q-table: {e}")
        data["q_table"] = pd.DataFrame()

    # Load metrics
    try:
        if Path("data/metrics.db").exists():
            conn = sqlite3.connect("data/metrics.db")
            m_df = pd.read_sql_query("SELECT * FROM metrics", conn)
            data["metrics"] = m_df
            conn.close()
        else:
            data["metrics"] = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        data["metrics"] = pd.DataFrame()

    # Load learning metrics
    try:
        if Path("data/qtable.db").exists():
            conn = sqlite3.connect("data/qtable.db")
            lm_df = pd.read_sql_query("SELECT * FROM learning_metrics", conn)
            data["learning"] = lm_df
            conn.close()
        else:
            data["learning"] = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading learning metrics: {e}")
        data["learning"] = pd.DataFrame()

    return data


# Load data
with st.spinner("Loading data..."):
    data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Overview", "Data Tables", "Charts"])

if page == "Overview":
    st.header("📊 System Overview")

    # Show basic stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not data["metrics"].empty:
            avg_reward = data["metrics"]["reward"].mean()
            st.metric("Average Reward", f"{avg_reward:.3f}")
        else:
            st.metric("Average Reward", "N/A")

    with col2:
        if not data["q_table"].empty:
            total_states = data["q_table"]["state_hash"].nunique()
            st.metric("Total States", total_states)
        else:
            st.metric("Total States", "N/A")

    with col3:
        if not data["learning"].empty:
            total_episodes = data["learning"]["episode_id"].max()
            st.metric("Total Episodes", total_episodes)
        else:
            st.metric("Total Episodes", "N/A")

    with col4:
        if not data["metrics"].empty:
            success_rate = (data["metrics"]["pass_rate"] > 0.5).mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "N/A")

    # Show data info
    st.subheader("📈 Data Summary")
    st.write(f"Q-table entries: {len(data['q_table'])}")
    st.write(f"Metrics entries: {len(data['metrics'])}")
    st.write(f"Learning episodes: {len(data['learning'])}")

elif page == "Data Tables":
    st.header("📋 Data Tables")

    # Q-table
    if not data["q_table"].empty:
        st.subheader("Q-Table Data")
        st.dataframe(data["q_table"].head(10))

    # Metrics
    if not data["metrics"].empty:
        st.subheader("Pipeline Metrics")
        st.dataframe(data["metrics"].head(10))

    # Learning metrics
    if not data["learning"].empty:
        st.subheader("Learning Metrics")
        st.dataframe(data["learning"].head(10))

elif page == "Charts":
    st.header("📊 Charts")

    # Learning curve
    if not data["learning"].empty:
        st.subheader("Learning Curve")
        fig = px.line(
            data["learning"], x="episode_id", y="reward", title="Reward over Episodes"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Metrics over time
    if not data["metrics"].empty:
        st.subheader("Pipeline Performance")
        fig = px.line(
            data["metrics"], x="iteration", y="reward", title="Reward over Iterations"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pass rate
        fig2 = px.line(
            data["metrics"],
            x="iteration",
            y="pass_rate",
            title="Pass Rate over Iterations",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Q-value distribution
    if not data["q_table"].empty:
        st.subheader("Q-Value Distribution")
        fig = px.histogram(data["q_table"], x="q_value", title="Q-Value Distribution")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Status:** ✅ Working")
st.sidebar.markdown("**Data loaded:** " + str(sum(len(df) for df in data.values())))
