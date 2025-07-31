# 📊 **VALIDATION DASHBOARD - Real-Time Monitoring System**

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .logger import ValidationLogger
import numpy as np


class ValidationDashboard:
    """Real-time dashboard for CodeConductor empirical validation"""

    def __init__(self):
        self.logger = ValidationLogger()

    def render_dashboard(self):
        """Main dashboard rendering"""
        st.title("🎯 CodeConductor Validation Dashboard")
        st.markdown("**Real-time monitoring of empirical validation progress**")

        # Load data
        df = self.logger.get_comparison_data()

        if df.empty:
            st.warning("No validation data found. Start logging tasks to see metrics.")
            return

        # Calculate key metrics
        savings = self.logger.calculate_time_savings()
        roi = self.logger.calculate_roi()

        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Time Savings",
                f"{savings['time_savings_percent']:.1f}%",
                help="Percentage time saved with CodeConductor vs Manual",
            )

        with col2:
            st.metric(
                "Hours Saved",
                f"{savings['total_hours_saved']:.1f}h",
                help="Total hours saved across all tasks",
            )

        with col3:
            st.metric(
                "ROI Value",
                f"${roi['value_saved']:.0f}",
                help="Monetary value of time saved",
            )

        with col4:
            avg_manual = savings.get("manual_avg_time", 0) / 60
            avg_cc = savings.get("cc_avg_time", 0) / 60
            st.metric(
                "Avg Task Time",
                f"{avg_manual:.1f}→{avg_cc:.1f} min",
                help="Average task time: Manual → CodeConductor",
            )

        # Charts section
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            self.render_time_comparison_chart(df)

        with col2:
            self.render_quality_metrics_chart(df)

        # Detailed metrics
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            self.render_roi_progress_chart(roi)

        with col2:
            self.render_agent_performance_chart(df)

        # Data table
        st.markdown("---")
        st.subheader("📋 Raw Validation Data")
        st.dataframe(df, use_container_width=True)

    def render_time_comparison_chart(self, df: pd.DataFrame):
        """Time comparison chart - Simplified and professional"""
        st.subheader("⏱️ Time Comparison")

        # Prepare data for comparison
        manual_data = df[df["Mode"] == "Manual"].copy()
        cc_data = df[df["Mode"] == "CodeConductor"].copy()

        if manual_data.empty or cc_data.empty:
            st.info("Need both Manual and CodeConductor data for comparison")
            return

        # Calculate average times
        manual_avg = manual_data["Duration_sec"].astype(float).mean() / 60
        cc_avg = cc_data["Duration_sec"].astype(float).mean() / 60

        # Create simple comparison chart
        fig = go.Figure()

        # Average time comparison
        fig.add_trace(
            go.Bar(
                x=["Manual", "CodeConductor"],
                y=[manual_avg, cc_avg],
                name="Average Time",
                marker_color=["#ff6b6b", "#4ecdc4"],
                text=[f"{manual_avg:.1f} min", f"{cc_avg:.1f} min"],
                textposition="auto",
            )
        )

        fig.update_layout(
            title="Average Task Completion Time",
            xaxis_title="Mode",
            yaxis_title="Time (minutes)",
            showlegend=False,
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Manual Avg",
                f"{manual_avg:.1f} min",
                help="Average time for manual tasks",
            )

        with col2:
            st.metric(
                "CodeConductor Avg",
                f"{cc_avg:.1f} min",
                help="Average time for CodeConductor tasks",
            )

        with col3:
            time_savings = ((manual_avg - cc_avg) / manual_avg) * 100
            st.metric(
                "Time Savings",
                f"{time_savings:.1f}%",
                help="Time saved with CodeConductor",
            )

    def render_quality_metrics_chart(self, df: pd.DataFrame):
        """Quality metrics chart - Simplified and professional"""
        st.subheader("✅ Quality Metrics")

        # Calculate average test pass rates by mode
        manual_data = df[df["Mode"] == "Manual"].copy()
        cc_data = df[df["Mode"] == "CodeConductor"].copy()

        if manual_data.empty or cc_data.empty:
            st.info("Need both Manual and CodeConductor data for comparison")
            return

        # Calculate test pass rates
        def calculate_pass_rate(data):
            if data.empty:
                return 0
            total_passed = data["Tests_Passed"].astype(float).sum()
            total_tests = data["Total_Tests"].astype(float).sum()
            return (total_passed / total_tests * 100) if total_tests > 0 else 0

        manual_pass_rate = calculate_pass_rate(manual_data)
        cc_pass_rate = calculate_pass_rate(cc_data)

        # Create simple comparison chart
        fig = go.Figure()

        # Manual vs CodeConductor comparison
        fig.add_trace(
            go.Bar(
                x=["Manual", "CodeConductor"],
                y=[manual_pass_rate, cc_pass_rate],
                name="Test Pass Rate",
                marker_color=["#ff6b6b", "#4ecdc4"],
                text=[f"{manual_pass_rate:.1f}%", f"{cc_pass_rate:.1f}%"],
                textposition="auto",
            )
        )

        # Add target line
        fig.add_hline(
            y=90,
            line_dash="dash",
            line_color="red",
            annotation_text="Target: 90%",
            annotation_position="top right",
        )

        fig.update_layout(
            title="Average Test Pass Rate Comparison",
            xaxis_title="Mode",
            yaxis_title="Test Pass Rate (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Manual Avg",
                f"{manual_pass_rate:.1f}%",
                help="Average test pass rate for manual tasks",
            )

        with col2:
            st.metric(
                "CodeConductor Avg",
                f"{cc_pass_rate:.1f}%",
                help="Average test pass rate for CodeConductor tasks",
            )

        with col3:
            improvement = cc_pass_rate - manual_pass_rate
            st.metric(
                "Improvement",
                f"+{improvement:.1f}%",
                help="Quality improvement with CodeConductor",
            )

    def render_roi_progress_chart(self, roi: dict):
        """ROI progress chart"""
        st.subheader("💰 ROI Progress")

        # Target ROI from validation plan
        target_roi = 2350  # $2,350 from 11-week plan
        current_roi = roi["value_saved"]
        progress = min(current_roi / target_roi * 100, 100)

        # Create gauge chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_roi,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "ROI Progress ($)"},
                delta={"reference": target_roi},
                gauge={
                    "axis": {"range": [None, target_roi]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, target_roi * 0.5], "color": "lightgray"},
                        {
                            "range": [target_roi * 0.5, target_roi * 0.8],
                            "color": "yellow",
                        },
                        {"range": [target_roi * 0.8, target_roi], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": target_roi,
                    },
                },
            )
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Progress text
        st.markdown(f"**Progress:** {progress:.1f}% of target ${target_roi:,}")

    def render_agent_performance_chart(self, df: pd.DataFrame):
        """Agent performance metrics - Simplified and professional"""
        st.subheader("🤖 Agent Performance")

        cc_data = df[df["Mode"] == "CodeConductor"].copy()

        if cc_data.empty:
            st.info("No CodeConductor data for agent metrics")
            return

        # Calculate average metrics
        avg_confidence = cc_data["Model_Confidence"].astype(float).mean() * 100
        avg_agreement = cc_data["Ensemble_Agreement"].astype(float).mean() * 100
        avg_reward = cc_data["RLHF_Reward"].astype(float).mean() * 100
        avg_cognitive = cc_data["Cognitive_Load"].astype(float).mean()
        avg_satisfaction = cc_data["Satisfaction_Score"].astype(float).mean()

        # Create simple bar chart
        fig = go.Figure()

        metrics = ["Model Confidence", "Ensemble Agreement", "RLHF Reward"]
        values = [avg_confidence, avg_agreement, avg_reward]

        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name="Performance (%)",
                marker_color="#4ecdc4",
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
            )
        )

        fig.update_layout(
            title="CodeConductor Agent Performance",
            xaxis_title="Metric",
            yaxis_title="Performance (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1f}%",
                help="Average model confidence",
            )

        with col2:
            st.metric(
                "Avg Agreement",
                f"{avg_agreement:.1f}%",
                help="Average ensemble agreement",
            )

        with col3:
            st.metric(
                "Avg Satisfaction",
                f"{avg_satisfaction:.1f}/10",
                help="Average user satisfaction",
            )

    def render_weekly_progress(self, df: pd.DataFrame):
        """Weekly progress tracking"""
        st.subheader("📈 Weekly Progress")

        # Add week column
        df["Date"] = pd.to_datetime(df["Date"])
        df["Week"] = df["Date"].dt.isocalendar().week

        # Weekly averages
        weekly_stats = (
            df.groupby(["Week", "Mode"])["Duration_sec"]
            .agg(["mean", "count"])
            .reset_index()
        )

        fig = px.line(
            weekly_stats,
            x="Week",
            y="mean",
            color="Mode",
            title="Average Task Time by Week",
            labels={"mean": "Average Time (seconds)", "Week": "Week Number"},
        )

        st.plotly_chart(fig, use_container_width=True)


# Streamlit app integration
def render_validation_dashboard():
    """Render the validation dashboard in Streamlit"""
    dashboard = ValidationDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    # Test dashboard
    import streamlit as st

    render_validation_dashboard()
