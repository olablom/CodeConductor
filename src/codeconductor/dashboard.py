# 📊 **VALIDATION DASHBOARD - Real-Time Monitoring System**

import os
import platform
import re
import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .logger import ValidationLogger


class ValidationDashboard:
    """Real-time dashboard for CodeConductor empirical validation"""

    def __init__(self):
        self.logger = ValidationLogger()

    def render_dashboard(self):
        """Main dashboard rendering"""
        # Sidebar preflight first
        self.render_preflight_sidebar()

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

    def render_preflight_sidebar(self):
        """Render a small preflight panel in the sidebar for Cursor/Ollama status."""
        with st.sidebar:
            cursor_mode = os.environ.get("CURSOR_MODE", "manual")
            badge = "[Manual]" if cursor_mode.lower() != "auto" else "[Auto]"
            st.subheader(f"🔎 Preflight {badge}")

            st.caption("Environment (Cursor API disabled by default)")
            st.code(f"CURSOR_MODE={cursor_mode}")
            # Tooltips for modes
            st.caption(
                "Manual mode: Cursor API skipped; use clipboard flow. Set CURSOR_MODE=auto only if you enable Cursor Local API."
            )

            # Keep only Ollama quick check; drop Cursor API checks in manual mode
            latest_path = Path("artifacts/diagnostics/diagnose_latest.txt")
            ollama_up = None
            diagnostics_ts = None
            if latest_path.exists():
                try:
                    text = latest_path.read_text(encoding="utf-8", errors="ignore")
                    ollama_up = "Port 11434 -> TcpTestSucceeded=True" in text
                    m_ts = re.search(r"^===\s*(.+?)\s*===\s*$", text, re.MULTILINE)
                    if m_ts:
                        diagnostics_ts = m_ts.group(1)
                except Exception:
                    pass
            else:
                st.info("diagnose_latest.txt not found. Run the diagnostics script.")

            # Status badges
            col1, col2 = st.columns(2)
            with col1:
                if ollama_up is True:
                    st.success(
                        "Ollama 11434: OK",
                        help="404 on /health is normal — port is alive.",
                    )
                elif ollama_up is False:
                    st.error("Ollama 11434: DOWN")
                else:
                    st.warning("Ollama 11434: N/A")
            with col2:
                st.info("Cursor API: disabled (manual mode)")

            if diagnostics_ts:
                st.caption(f"Last run: {diagnostics_ts}")

            # Actions
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Open diagnostics folder"):
                    diag_dir = str(
                        (
                            latest_path.parent
                            if latest_path.exists()
                            else Path("artifacts/diagnostics")
                        ).resolve()
                    )
                    try:
                        if platform.system() == "Windows":
                            os.startfile(diag_dir)  # type: ignore[attr-defined]
                        elif platform.system() == "Darwin":
                            subprocess.Popen(["open", diag_dir])
                        else:
                            subprocess.Popen(["xdg-open", diag_dir])
                    except Exception as e:
                        st.warning(f"Could not open folder: {e}")
            with col_b:
                if st.button("Run diagnostics now"):
                    try:
                        if platform.system() == "Windows":
                            cmd = [
                                "powershell",
                                "-NoProfile",
                                "-ExecutionPolicy",
                                "Bypass",
                                "-File",
                                "scripts/diagnose_cursor.ps1",
                                "-Ports",
                                "11434",
                                "3000",
                                "5123",
                                "5173",
                                "8000",
                            ]
                            subprocess.run(cmd, check=False, timeout=5)
                            st.success("Diagnostics executed. Reloading...")
                            st.experimental_rerun()
                        else:
                            st.info("Diagnostics script is PowerShell-based; run it on Windows.")
                    except Exception as e:
                        st.error(f"Diagnostics failed: {e}")

            # Telemetry hint
            if os.environ.get("CC_TELEMETRY", "0") == "1":
                st.caption("preflight logged locally (JSONL)")

            # Selector decision card (if available via artifact/telemetry)
            st.markdown("---")
            st.subheader("🧠 Selector Decision")
            # Try to read latest artifacts/runs/*/selector_decision.json
            try:
                runs_dir = Path("artifacts/runs")
                latest = None
                if runs_dir.exists():
                    run_dirs = sorted(runs_dir.glob("*"), key=lambda p: p.name, reverse=True)
                    latest = run_dirs[0] if run_dirs else None
                decision = None
                if latest:
                    sel_file = latest / "selector_decision.json"
                    if sel_file.exists():
                        import json as _json

                        decision = _json.loads(sel_file.read_text(encoding="utf-8"))

                if decision:
                    pol = decision.get("policy", os.environ.get("SELECTOR_POLICY", "latency"))
                    chosen = decision.get("chosen") or decision.get("selected_model")
                    sampling = decision.get("sampling", {})
                    st.caption(f"Policy: {pol}")
                    st.write(f"Chosen: {chosen}")
                    if sampling:
                        st.code(_json.dumps(sampling, ensure_ascii=False, indent=2))
                    scores = decision.get("scores", {})
                    if scores:
                        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                        st.write("Top candidates:")
                        for mid, sc in top3:
                            st.write(f"- {mid}: {sc:.3f}")
                    if st.button("Copy decision JSON"):
                        try:
                            st.session_state["_selector_json"] = decision
                            st.success("Decision JSON ready to copy")
                        except Exception:
                            pass
                else:
                    st.info("No recent decision. Run a task to populate artifacts.")

                # Consensus mini row (latest)
                try:
                    if latest:
                        cons_file = latest / "consensus.json"
                        if cons_file.exists():
                            cons = _json.loads(cons_file.read_text(encoding="utf-8"))
                            winner = cons.get("winner", {})
                            sc = winner.get("score")
                            model = winner.get("model")
                            if sc is not None:
                                st.caption(
                                    f"Consensus {float(sc):.2f} — {model or 'N/A'}",
                                    help="fast CodeBLEU + heuristik",
                                )
                except Exception:
                    pass
            except Exception as e:
                st.warning(f"Selector decision unavailable: {e}")

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
        cc_data["Cognitive_Load"].astype(float).mean()
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
            df.groupby(["Week", "Mode"])["Duration_sec"].agg(["mean", "count"]).reset_index()
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


if __name__ == "__main__":  # pragma: no cover
    # Test dashboard
    import streamlit as st

    render_validation_dashboard()
