import streamlit as st
import asyncio
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import pandas as pd
from collections import deque
import time

# Import pipeline integration
from pipeline_dashboard_integration import (
    DashboardConnector,
    process_events_in_dashboard,
)

# Page config
st.set_page_config(
    page_title="CodeConductor AI Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .agent-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
        animation: fadeIn 0.5s;
    }
    .architect { background-color: #e3f2fd; border-left: 4px solid #2196F3; }
    .codegen { background-color: #e8f5e9; border-left: 4px solid #4CAF50; }
    .reviewer { background-color: #fff3e0; border-left: 4px solid #FF9800; }
    .policy { background-color: #f3e5f5; border-left: 4px solid #9C27B0; }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-active { background-color: #4CAF50; }
    .status-idle { background-color: #FFC107; }
    .status-error { background-color: #F44336; }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = deque(maxlen=50)
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_tasks": 0,
        "successful_tasks": 0,
        "failed_tasks": 0,
        "avg_completion_time": 0,
        "human_interventions": 0,
    }
if "current_task" not in st.session_state:
    st.session_state.current_task = None
if "agent_performance" not in st.session_state:
    st.session_state.agent_performance = {
        "ArchitectAgent": {"tasks": 0, "avg_confidence": 0},
        "CodeGenAgent": {"tasks": 0, "avg_confidence": 0},
        "ReviewAgent": {"tasks": 0, "avg_confidence": 0},
        "PolicyAgent": {"tasks": 0, "avg_confidence": 0},
    }
if "pending_approvals" not in st.session_state:
    st.session_state.pending_approvals = []
if "progress" not in st.session_state:
    st.session_state.progress = {"current": 0, "total": 0}

# Header
st.title("🎯 CodeConductor AI Dashboard")
st.markdown("**Real-time monitoring of your AI coding assistant**")

# Sidebar
with st.sidebar:
    st.header("🎮 Control Panel")

    # Task Input
    st.subheader("New Task")
    task_description = st.text_area(
        "Task Description",
        height=100,
        placeholder="Example: Create a REST API for user authentication with JWT tokens",
    )

    complexity = st.select_slider(
        "Complexity", options=["Simple", "Medium", "Complex", "Expert"], value="Medium"
    )

    if st.button("🚀 Execute Task", type="primary", use_container_width=True):
        if task_description:
            # Initialize connector if not exists
            if "connector" not in st.session_state:
                st.session_state.connector = DashboardConnector()
                st.session_state.connector.initialize_pipeline()

            # Execute task asynchronously
            st.session_state.connector.execute_task_async(task_description)
            st.success("Task submitted!")

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    auto_approve = st.checkbox("Auto-approve agent suggestions", value=False)
    show_confidence = st.checkbox("Show confidence scores", value=True)
    max_steps = st.slider("Max reasoning steps", 2, 10, 6)

    st.divider()

    # System Status
    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Agents", "4", "2")
    with col2:
        st.metric("GPU Usage", "78%", "-5%")

# Process events from pipeline
if "connector" in st.session_state:
    process_events_in_dashboard(st.session_state.connector)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗨️ Live Chat", "📈 Metrics", "🏆 Performance", "📜 History"]
)

with tab1:
    # Live Agent Chat
    st.header("Live Agent Discussion")

    # Current task info
    if st.session_state.current_task:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Task:** {st.session_state.current_task['description'][:50]}...")
        with col2:
            status_color = (
                "🟢" if st.session_state.current_task["status"] == "running" else "🔴"
            )
            st.info(
                f"**Status:** {status_color} {st.session_state.current_task['status'].title()}"
            )
        with col3:
            if st.session_state.current_task["status"] == "running":
                elapsed = (
                    datetime.now() - st.session_state.current_task["start_time"]
                ).seconds
                st.info(f"**Elapsed:** {elapsed}s")

        # Progress bar
        if st.session_state.progress["total"] > 0:
            progress = (
                st.session_state.progress["current"]
                / st.session_state.progress["total"]
            )
            st.progress(
                progress,
                text=f"Step {st.session_state.progress['current']} of {st.session_state.progress['total']}",
            )

    # Pending approvals
    if st.session_state.pending_approvals:
        st.subheader("⏳ Pending Approvals")
        for approval in st.session_state.pending_approvals:
            with st.expander(
                f"Step {approval['step']}: {approval['agent']} needs approval"
            ):
                st.write(approval["suggestion"])
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve", key=f"approve_{approval['step']}"):
                        st.session_state.connector.approve_suggestion(approval["step"])
                        st.success("Approved!")
                        st.rerun()
                with col2:
                    if st.button("❌ Reject", key=f"reject_{approval['step']}"):
                        st.session_state.connector.reject_suggestion(approval["step"])
                        st.error("Rejected!")
                        st.rerun()

    # Chat container
    chat_container = st.container()

    # Display messages
    with chat_container:
        for msg in reversed(list(st.session_state.messages)):
            agent_class = msg["agent"].replace("Agent", "").lower()
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(
                    f"""
                <div class="agent-message {agent_class}">
                    <strong>{msg["agent"]}</strong> ({msg["timestamp"].strftime("%H:%M:%S")})<br>
                    {msg["message"]}
                    {f"<br><small>Confidence: {msg['confidence']:.2%}</small>" if show_confidence else ""}
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                if not auto_approve and msg.get("needs_approval"):
                    if st.button("✅", key=f"approve_{msg['timestamp']}"):
                        st.success("Approved!")
                    if st.button("❌", key=f"reject_{msg['timestamp']}"):
                        st.error("Rejected!")

with tab2:
    # Metrics Dashboard
    st.header("📊 Performance Metrics")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>Total Tasks</h3>
            <h1>{st.session_state.metrics["total_tasks"]}</h1>
            <p>↑ 12% from last week</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        success_rate = (
            st.session_state.metrics["successful_tasks"]
            / max(st.session_state.metrics["total_tasks"], 1)
        ) * 100
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>Success Rate</h3>
            <h1>{success_rate:.0f}%</h1>
            <p>↑ 5% from last week</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>Avg Time</h3>
            <h1>3.2 min</h1>
            <p>↓ 18% improvement</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>Human Input</h3>
            <h1>{st.session_state.metrics["human_interventions"]}</h1>
            <p>↓ 8% less needed</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Charts
    st.subheader("Trends")

    # Success rate over time
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    success_rates = [0.7 + 0.2 * (i / 30) + 0.1 * (i % 5) / 5 for i in range(30)]

    fig_success = go.Figure()
    fig_success.add_trace(
        go.Scatter(
            x=dates,
            y=success_rates,
            mode="lines+markers",
            name="Success Rate",
            line=dict(color="#4CAF50", width=3),
            fill="tozeroy",
            fillcolor="rgba(76, 175, 80, 0.1)",
        )
    )
    fig_success.update_layout(
        title="Success Rate Trend",
        xaxis_title="Date",
        yaxis_title="Success Rate",
        yaxis=dict(tickformat=".0%"),
        height=400,
    )
    st.plotly_chart(fig_success, use_container_width=True)

with tab3:
    # Agent Performance
    st.header("🏆 Agent Performance Analysis")

    # Agent comparison
    agents = ["ArchitectAgent", "CodeGenAgent", "ReviewAgent", "PolicyAgent"]
    tasks_completed = [23, 42, 38, 41]
    avg_confidence = [0.92, 0.88, 0.94, 0.96]

    col1, col2 = st.columns(2)

    with col1:
        fig_tasks = px.bar(
            x=agents,
            y=tasks_completed,
            title="Tasks Completed by Agent",
            color=agents,
            color_discrete_map={
                "ArchitectAgent": "#2196F3",
                "CodeGenAgent": "#4CAF50",
                "ReviewAgent": "#FF9800",
                "PolicyAgent": "#9C27B0",
            },
        )
        st.plotly_chart(fig_tasks, use_container_width=True)

    with col2:
        fig_confidence = px.bar(
            x=agents,
            y=avg_confidence,
            title="Average Confidence by Agent",
            color=agents,
            color_discrete_map={
                "ArchitectAgent": "#2196F3",
                "CodeGenAgent": "#4CAF50",
                "ReviewAgent": "#FF9800",
                "PolicyAgent": "#9C27B0",
            },
        )
        fig_confidence.update_layout(yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig_confidence, use_container_width=True)

    # Agent collaboration matrix
    st.subheader("Agent Collaboration Patterns")
    collaboration_data = {
        "From\\To": agents,
        "ArchitectAgent": [0, 15, 8, 12],
        "CodeGenAgent": [12, 0, 18, 14],
        "ReviewAgent": [8, 20, 0, 16],
        "PolicyAgent": [6, 10, 14, 0],
    }

    df_collab = pd.DataFrame(collaboration_data)
    fig_heatmap = px.imshow(
        df_collab.iloc[:, 1:].values,
        labels=dict(x="To Agent", y="From Agent", color="Interactions"),
        x=agents,
        y=agents,
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab4:
    # Task History
    st.header("📜 Task History")

    # Sample task history
    history_data = {
        "Task": [
            "Create REST API",
            "Implement auth system",
            "Build data pipeline",
            "Optimize database queries",
            "Create ML model API",
        ],
        "Status": ["✅ Success", "✅ Success", "❌ Failed", "✅ Success", "🔄 Running"],
        "Time": ["3.2 min", "5.1 min", "8.3 min", "2.7 min", "1.2 min"],
        "Human Input": ["No", "Yes", "Yes", "No", "No"],
        "Complexity": ["Medium", "Complex", "Expert", "Simple", "Medium"],
    }

    df_history = pd.DataFrame(history_data)
    st.dataframe(
        df_history,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Time": st.column_config.TextColumn("Time", width="small"),
            "Human Input": st.column_config.TextColumn("Human Input", width="small"),
            "Complexity": st.column_config.TextColumn("Complexity", width="small"),
        },
    )

    # Export button
    if st.button("📥 Export History"):
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="codeconductor_history.csv",
            mime="text/csv",
        )

# Footer
st.divider()
st.markdown(
    """
<center>
    <small>CodeConductor AI Dashboard v1.0 | Real-time Multi-Agent Orchestration</small>
</center>
""",
    unsafe_allow_html=True,
)

# Auto-refresh when task is running
current_task = st.session_state.get("current_task", {})
if current_task and current_task.get("status") == "running":
    time.sleep(1)
    st.rerun()
