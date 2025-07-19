"""
Advanced Analytics Dashboard for CodeConductor

Provides ML-driven insights, predictions, and trend analysis.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List

from analytics.ml_predictor import create_predictor


def load_metrics_data() -> pd.DataFrame:
    """Load metrics data from database"""
    try:
        conn = sqlite3.connect("data/metrics.db")
        df = pd.read_sql_query("SELECT * FROM metrics", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


def create_quality_prediction_chart(predictor, df: pd.DataFrame):
    """Create quality prediction visualization"""
    st.subheader("🎯 ML Quality Predictions")

    # Get recent predictions
    if not df.empty:
        recent_data = df.tail(20)

        # Create prediction chart
        fig = go.Figure()

        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data["pass_rate"],
                mode="lines+markers",
                name="Actual Pass Rate",
                line=dict(color="blue"),
            )
        )

        # Add trend line
        if len(recent_data) > 1:
            z = np.polyfit(recent_data.index, recent_data["pass_rate"], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=p(recent_data.index),
                    mode="lines",
                    name="Trend",
                    line=dict(color="red", dash="dash"),
                )
            )

        fig.update_layout(
            title="Quality Trends Over Time",
            xaxis_title="Iteration",
            yaxis_title="Pass Rate",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Prediction interface
    st.subheader("🔮 Make New Prediction")

    col1, col2 = st.columns(2)

    with col1:
        prompt = st.text_area("Enter prompt for prediction:", height=100)
        strategy = st.selectbox(
            "Strategy:", ["conservative", "balanced", "exploratory"]
        )

    with col2:
        prev_pass_rate = st.slider("Previous Pass Rate:", 0.0, 1.0, 0.5)
        complexity_avg = st.slider("Average Complexity:", 0.0, 1.0, 0.5)
        model_source = st.selectbox("Model Source:", ["mock", "lm_studio"])

    if st.button("Predict Quality") and prompt:
        context = {
            "strategy": strategy,
            "prev_pass_rate": prev_pass_rate,
            "complexity_avg": complexity_avg,
            "reward_avg": 30.0,
            "iteration_count": 1,
            "model_source": model_source,
        }

        prediction = predictor.predict_quality(prompt, context)

        if "error" not in prediction:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Quality Score", f"{prediction['quality_score']:.2f}")

            with col2:
                st.metric(
                    "Success Probability", f"{prediction['success_probability']:.2f}"
                )

            with col3:
                st.metric("Confidence", f"{prediction['confidence']:.2f}")

            # Show warnings
            if prediction.get("warnings"):
                st.warning("⚠️ Warnings:")
                for warning in prediction["warnings"]:
                    st.write(f"• {warning}")

            # Feature importance
            if prediction.get("feature_importance"):
                st.subheader("📊 Feature Importance")
                importance_df = pd.DataFrame(
                    list(prediction["feature_importance"].items()),
                    columns=["Feature", "Importance"],
                ).sort_values("Importance", ascending=True)

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance for Prediction",
                )
                st.plotly_chart(fig, use_container_width=True)


def create_trends_analysis(predictor):
    """Create trends analysis visualization"""
    st.subheader("📈 Trends Analysis")

    # Get trends
    trends = predictor.get_trends(days=30)

    if "error" not in trends:
        col1, col2, col3 = st.columns(3)

        with col1:
            trend_icon = "📈" if trends["pass_rate_trend"] > 0 else "📉"
            st.metric(
                "Pass Rate Trend",
                f"{trends['pass_rate_trend']:.3f}",
                delta=f"{trends['pass_rate_trend']:.3f}",
                delta_color="normal",
            )

        with col2:
            st.metric("Total Iterations", trends["total_iterations"])

        with col3:
            st.metric("Daily Average", f"{trends['avg_daily_iterations']:.1f}")

        # Improvement rate
        if trends["improvement_rate"] > 0:
            st.success(f"🎉 Improvement rate: {trends['improvement_rate']:.3f} per day")
        else:
            st.warning(f"⚠️ Decline rate: {abs(trends['improvement_rate']):.3f} per day")
    else:
        st.warning("No trend data available")


def create_performance_metrics(df: pd.DataFrame):
    """Create performance metrics dashboard"""
    st.subheader("📊 Performance Metrics")

    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_pass_rate = df["pass_rate"].mean()
            st.metric("Avg Pass Rate", f"{avg_pass_rate:.2f}")

        with col2:
            avg_complexity = df["complexity"].mean()
            st.metric("Avg Complexity", f"{avg_complexity:.2f}")

        with col3:
            avg_reward = df["reward"].mean()
            st.metric("Avg Reward", f"{avg_reward:.1f}")

        with col4:
            total_iterations = len(df)
            st.metric("Total Iterations", total_iterations)

        # Strategy performance
        st.subheader("🎯 Strategy Performance")
        strategy_perf = (
            df.groupby("arm_selected")
            .agg({"pass_rate": "mean", "reward": "mean", "complexity": "mean"})
            .round(3)
        )

        st.dataframe(strategy_perf)

        # Model source performance
        if "model_source" in df.columns:
            st.subheader("🤖 Model Source Performance")
            model_perf = (
                df.groupby("model_source")
                .agg({"pass_rate": "mean", "reward": "mean"})
                .round(3)
            )

            st.dataframe(model_perf)


def create_distribution_charts(df: pd.DataFrame):
    """Create distribution charts"""
    st.subheader("📊 Data Distributions")

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Pass rate distribution
            fig = px.histogram(
                df, x="pass_rate", nbins=20, title="Pass Rate Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Reward distribution
            fig = px.histogram(df, x="reward", nbins=20, title="Reward Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Complexity vs Pass Rate scatter
        fig = px.scatter(
            df,
            x="complexity",
            y="pass_rate",
            color="arm_selected",
            title="Complexity vs Pass Rate by Strategy",
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard function"""
    st.set_page_config(
        page_title="CodeConductor Analytics", page_icon="🎼", layout="wide"
    )

    st.title("🎼 CodeConductor Analytics Dashboard")
    st.markdown("ML-driven insights and predictions for code generation quality")

    # Load data
    df = load_metrics_data()

    # Initialize predictor
    try:
        predictor = create_predictor()
        predictor_available = True
    except Exception as e:
        st.error(f"ML predictor not available: {e}")
        predictor_available = False

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "ML Predictions", "Performance Metrics", "Data Distributions"],
    )

    if page == "Overview":
        st.header("📈 Overview")

        if not df.empty:
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Iterations", len(df))
                st.metric("Avg Pass Rate", f"{df['pass_rate'].mean():.2f}")

            with col2:
                st.metric("Avg Complexity", f"{df['complexity'].mean():.2f}")
                st.metric("Avg Reward", f"{df['reward'].mean():.1f}")

            with col3:
                st.metric(
                    "Best Strategy",
                    df.groupby("arm_selected")["pass_rate"].mean().idxmax(),
                )
                st.metric("Success Rate", f"{(df['pass_rate'] > 0.5).mean():.1%}")

            # Recent activity
            st.subheader("🕒 Recent Activity")
            recent_df = df.tail(10)[
                ["iteration", "arm_selected", "pass_rate", "reward", "complexity"]
            ]
            st.dataframe(recent_df)

            if predictor_available:
                create_trends_analysis(predictor)
        else:
            st.warning("No data available. Run some pipeline iterations first!")

    elif page == "ML Predictions" and predictor_available:
        create_quality_prediction_chart(predictor, df)

    elif page == "Performance Metrics":
        create_performance_metrics(df)

    elif page == "Data Distributions":
        create_distribution_charts(df)

    # Footer
    st.markdown("---")
    st.markdown("*Powered by CodeConductor v2.0 - ML-driven code generation analytics*")


if __name__ == "__main__":
    main()
