"""
CodeConductor Dashboard - Real-time Monitoring & Visualization

Features:
- Q-value heatmaps for RL insights
- Learning curves with moving averages
- Agent performance metrics
- Cost control and token usage
- Real-time pipeline monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import time

# Page config
st.set_page_config(
    page_title="CodeConductor Dashboard",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
</style>
""",
    unsafe_allow_html=True,
)


class DashboardDataLoader:
    """Load and process data from multiple databases"""

    def __init__(self):
        self.data_dir = Path("data")
        self.qtable_db = self.data_dir / "qtable.db"
        self.rl_history_db = self.data_dir / "rl_history.db"
        self.metrics_db = self.data_dir / "metrics.db"

    def load_q_table_data(self) -> pd.DataFrame:
        """Load Q-table data for heatmap visualization"""
        try:
            if not self.qtable_db.exists():
                return pd.DataFrame()

            conn = sqlite3.connect(self.qtable_db)

            # Load Q-table with state and action data
            query = """
            SELECT 
                state_hash,
                state_data,
                action_data,
                q_value,
                visit_count,
                last_updated
            FROM q_table
            ORDER BY q_value DESC
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            # Parse state and action data
            if not df.empty:
                df["state_parsed"] = df["state_data"].apply(self._parse_state_hash)
                df["action_parsed"] = df["action_data"].apply(self._parse_action_hash)

            return df

        except Exception as e:
            st.error(f"Error loading Q-table data: {e}")
            return pd.DataFrame()

    def load_learning_metrics(self) -> pd.DataFrame:
        """Load learning metrics for curves"""
        try:
            if not self.qtable_db.exists():
                return pd.DataFrame()

            conn = sqlite3.connect(self.qtable_db)

            # Load learning metrics
            query = """
            SELECT 
                episode_id,
                state_hash,
                action_hash,
                reward,
                timestamp
            FROM learning_metrics
            ORDER BY episode_id
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["episode_id"] = df["episode_id"].astype(int)

            return df

        except Exception as e:
            st.error(f"Error loading learning metrics: {e}")
            return pd.DataFrame()

    def load_rl_history(self) -> pd.DataFrame:
        """Load RL history for detailed analysis"""
        try:
            if not self.rl_history_db.exists():
                return pd.DataFrame()

            conn = sqlite3.connect(self.rl_history_db)

            # Load episodes with reward components
            query = """
            SELECT 
                e.episode_id,
                e.timestamp,
                e.total_reward,
                e.iteration_count,
                e.execution_time,
                e.status,
                rc.component_name,
                rc.reward_value
            FROM episodes e
            LEFT JOIN reward_components rc ON e.episode_id = rc.episode_id
            ORDER BY e.timestamp
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df

        except Exception as e:
            st.error(f"Error loading RL history: {e}")
            return pd.DataFrame()

    def load_pipeline_metrics(self) -> pd.DataFrame:
        """Load pipeline metrics for real-time monitoring"""
        try:
            if not self.metrics_db.exists():
                return pd.DataFrame()

            conn = sqlite3.connect(self.metrics_db)

            query = """
            SELECT 
                iteration,
                reward,
                pass_rate,
                complexity,
                arm_selected,
                timestamp,
                model_source,
                blocked,
                block_reasons,
                optimizer_state,
                optimizer_action
            FROM metrics
            ORDER BY iteration
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df

        except Exception as e:
            st.error(f"Error loading pipeline metrics: {e}")
            return pd.DataFrame()

    def _parse_state_hash(self, state_hash: str) -> Dict[str, Any]:
        """Parse state hash to readable format"""
        try:
            # Try to parse as JSON first
            return json.loads(state_hash)
        except:
            # Fallback to simple parsing
            return {"raw": state_hash[:20] + "..."}

    def _parse_action_hash(self, action_hash: str) -> Dict[str, Any]:
        """Parse action hash to readable format"""
        try:
            return json.loads(action_hash)
        except:
            return {"raw": action_hash[:20] + "..."}


class DashboardVisualizer:
    """Create interactive visualizations"""

    @staticmethod
    def create_q_value_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create Q-value heatmap"""
        if df.empty:
            return go.Figure()

        # Create a simpler heatmap using state_hash and action_data
        # Extract task type from state_data for better visualization
        df_copy = df.copy()

        # Extract task type from state_data
        def extract_task_type(state_data):
            try:
                state_dict = json.loads(state_data)
                return state_dict.get("task_type", "unknown")
            except:
                return "unknown"

        df_copy["task_type"] = df_copy["state_data"].apply(extract_task_type)

        # Extract action type from action_data
        def extract_action_type(action_data):
            try:
                action_dict = json.loads(action_data)
                return action_dict.get("agent_combination", "unknown")
            except:
                return "unknown"

        df_copy["action_type"] = df_copy["action_data"].apply(extract_action_type)

        # Create pivot table for heatmap
        pivot_data = df_copy.pivot_table(
            values="q_value",
            index="task_type",
            columns="action_type",
            fill_value=0,
            aggfunc="mean",
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale="Viridis",
                hoverongaps=False,
                hovertemplate="Task: %{y}<br>Action: %{x}<br>Q-Value: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Q-Value Heatmap (Task Type × Action Type)",
            xaxis_title="Actions",
            yaxis_title="Task Types",
            height=500,
        )

        return fig

    @staticmethod
    def create_learning_curve(df: pd.DataFrame) -> go.Figure:
        """Create learning curve with moving average"""
        if df.empty:
            return go.Figure()

        # Calculate moving average
        window_size = min(10, len(df) // 4)
        if window_size > 0:
            df["reward_ma"] = df["reward"].rolling(window=window_size).mean()

        fig = go.Figure()

        # Raw rewards
        fig.add_trace(
            go.Scatter(
                x=df["episode_id"],
                y=df["reward"],
                mode="lines+markers",
                name="Raw Reward",
                line=dict(color="lightblue", width=1),
                marker=dict(size=4),
            )
        )

        # Moving average
        if "reward_ma" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["episode_id"],
                    y=df["reward_ma"],
                    mode="lines",
                    name=f"Moving Average (window={window_size})",
                    line=dict(color="red", width=3),
                )
            )

        fig.update_layout(
            title="Learning Curve - Reward Progression",
            xaxis_title="Episode",
            yaxis_title="Reward",
            height=400,
            hovermode="x unified",
        )

        return fig

    @staticmethod
    def create_agent_performance_chart(df: pd.DataFrame) -> go.Figure:
        """Create agent performance visualization"""
        if df.empty:
            return go.Figure()

        # Group by component name for reward breakdown
        component_rewards = (
            df.groupby("component_name")["reward_value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=component_rewards["component_name"],
                    y=component_rewards["mean"],
                    error_y=dict(type="data", array=component_rewards["std"]),
                    name="Average Reward",
                    text=component_rewards["count"],
                    texttemplate="%{text} episodes",
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title="Agent Performance by Component",
            xaxis_title="Reward Component",
            yaxis_title="Average Reward",
            height=400,
        )

        return fig

    @staticmethod
    def create_pipeline_monitoring(df: pd.DataFrame) -> go.Figure:
        """Create pipeline monitoring dashboard"""
        if df.empty:
            return go.Figure()

        fig = go.Figure()

        # Add reward line
        fig.add_trace(
            go.Scatter(
                x=df["iteration"],
                y=df["reward"],
                mode="lines+markers",
                name="Reward",
                line=dict(color="blue", width=2),
                yaxis="y",
            )
        )

        # Add pass rate line
        fig.add_trace(
            go.Scatter(
                x=df["iteration"],
                y=df["pass_rate"],
                mode="lines+markers",
                name="Pass Rate",
                line=dict(color="green", width=2),
                yaxis="y2",
            )
        )

        # Add complexity line
        fig.add_trace(
            go.Scatter(
                x=df["iteration"],
                y=df["complexity"],
                mode="lines+markers",
                name="Complexity",
                line=dict(color="orange", width=2),
                yaxis="y3",
            )
        )

        fig.update_layout(
            title="Pipeline Performance Monitoring",
            xaxis_title="Iteration",
            height=500,
            yaxis=dict(title="Reward", side="left"),
            yaxis2=dict(title="Pass Rate", side="right", overlaying="y"),
            yaxis3=dict(title="Complexity", side="right", position=0.95),
            hovermode="x unified",
        )

        return fig


def main():
    """Main dashboard function"""

    # Header
    st.markdown(
        '<h1 class="main-header">🎼 CodeConductor Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Initialize data loader
    data_loader = DashboardDataLoader()
    visualizer = DashboardVisualizer()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "Overview",
            "RL Metrics",
            "Agent Performance",
            "Pipeline Monitoring",
            "Cost Control",
        ],
    )

    # Auto-refresh
    if st.sidebar.checkbox("Auto-refresh (30s)", value=True):
        time.sleep(30)
        st.experimental_rerun()

    # Load data
    with st.spinner("Loading data..."):
        q_table_df = data_loader.load_q_table_data()
        learning_df = data_loader.load_learning_metrics()
        rl_history_df = data_loader.load_rl_history()
        pipeline_df = data_loader.load_pipeline_metrics()

    # Overview page
    if page == "Overview":
        show_overview_page(
            q_table_df, learning_df, rl_history_df, pipeline_df, visualizer
        )

    # RL Metrics page
    elif page == "RL Metrics":
        show_rl_metrics_page(q_table_df, learning_df, visualizer)

    # Agent Performance page
    elif page == "Agent Performance":
        show_agent_performance_page(rl_history_df, visualizer)

    # Pipeline Monitoring page
    elif page == "Pipeline Monitoring":
        show_pipeline_monitoring_page(pipeline_df, visualizer)

    # Cost Control page
    elif page == "Cost Control":
        show_cost_control_page(pipeline_df, rl_history_df)


def show_overview_page(q_table_df, learning_df, rl_history_df, pipeline_df, visualizer):
    """Show overview dashboard"""

    st.header("📊 System Overview")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not pipeline_df.empty:
            avg_reward = pipeline_df["reward"].mean()
            st.metric("Average Reward", f"{avg_reward:.3f}")
        else:
            st.metric("Average Reward", "N/A")

    with col2:
        if not pipeline_df.empty:
            success_rate = (pipeline_df["pass_rate"] > 0.5).mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "N/A")

    with col3:
        if not q_table_df.empty:
            total_states = q_table_df["state_hash"].nunique()
            st.metric("Total States", total_states)
        else:
            st.metric("Total States", "N/A")

    with col4:
        if not learning_df.empty:
            total_episodes = learning_df["episode_id"].max()
            st.metric("Total Episodes", total_episodes)
        else:
            st.metric("Total Episodes", "N/A")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Learning Curve")
        learning_fig = visualizer.create_learning_curve(learning_df)
        st.plotly_chart(learning_fig, use_container_width=True)

    with col2:
        st.subheader("Pipeline Performance")
        pipeline_fig = visualizer.create_pipeline_monitoring(pipeline_df)
        st.plotly_chart(pipeline_fig, use_container_width=True)

    # Q-value heatmap
    st.subheader("Q-Value Heatmap")
    heatmap_fig = visualizer.create_q_value_heatmap(q_table_df)
    st.plotly_chart(heatmap_fig, use_container_width=True)


def show_rl_metrics_page(q_table_df, learning_df, visualizer):
    """Show RL metrics page"""

    st.header("🧠 Reinforcement Learning Metrics")

    # Q-table statistics
    if not q_table_df.empty:
        st.subheader("Q-Table Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_q_value = q_table_df["q_value"].mean()
            st.metric("Average Q-Value", f"{avg_q_value:.3f}")

        with col2:
            max_q_value = q_table_df["q_value"].max()
            st.metric("Max Q-Value", f"{max_q_value:.3f}")

        with col3:
            total_visits = q_table_df["visit_count"].sum()
            st.metric("Total Visits", total_visits)

        # Q-value distribution
        st.subheader("Q-Value Distribution")
        fig = px.histogram(
            q_table_df, x="q_value", nbins=30, title="Q-Value Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Learning curve
    st.subheader("Learning Progress")
    learning_fig = visualizer.create_learning_curve(learning_df)
    st.plotly_chart(learning_fig, use_container_width=True)

    # Top Q-values
    if not q_table_df.empty:
        st.subheader("Top Q-Values")
        top_q_values = q_table_df.nlargest(10, "q_value")[
            ["state_hash", "action_hash", "q_value", "visit_count"]
        ]
        st.dataframe(top_q_values)


def show_agent_performance_page(rl_history_df, visualizer):
    """Show agent performance page"""

    st.header("🤖 Agent Performance")

    if not rl_history_df.empty:
        # Agent performance chart
        st.subheader("Reward Components")
        perf_fig = visualizer.create_agent_performance_chart(rl_history_df)
        st.plotly_chart(perf_fig, use_container_width=True)

        # Execution time analysis
        st.subheader("Execution Time Analysis")
        if "execution_time" in rl_history_df.columns:
            exec_time_df = rl_history_df.groupby("episode_id")["execution_time"].first()
            fig = px.line(exec_time_df, title="Execution Time per Episode")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No agent performance data available")


def show_pipeline_monitoring_page(pipeline_df, visualizer):
    """Show pipeline monitoring page"""

    st.header("🔍 Pipeline Monitoring")

    if not pipeline_df.empty:
        # Real-time monitoring
        st.subheader("Real-time Performance")
        monitoring_fig = visualizer.create_pipeline_monitoring(pipeline_df)
        st.plotly_chart(monitoring_fig, use_container_width=True)

        # Model source distribution
        st.subheader("Model Source Distribution")
        if "model_source" in pipeline_df.columns:
            model_dist = pipeline_df["model_source"].value_counts()
            fig = px.pie(
                values=model_dist.values,
                names=model_dist.index,
                title="Model Source Usage",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Blocked vs Safe
        st.subheader("Safety Analysis")
        if "blocked" in pipeline_df.columns:
            safety_dist = pipeline_df["blocked"].value_counts()
            fig = px.pie(
                values=safety_dist.values,
                names=["Safe", "Blocked"],
                title="Safety Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No pipeline data available")


def show_cost_control_page(pipeline_df, rl_history_df):
    """Show cost control page"""

    st.header("💰 Cost Control")

    st.info("Cost control features coming soon! This will include:")
    st.markdown("""
    - Token usage tracking
    - Cost per iteration
    - Model cost comparison
    - Budget alerts
    - Cost optimization suggestions
    """)


if __name__ == "__main__":
    main()
