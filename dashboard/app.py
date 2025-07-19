import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

st.set_page_config(page_title="CodeConductor Dashboard", page_icon="🎼", layout="wide")

st.title("🎼 CodeConductor - Learning Dashboard")

# Databas-anslutning
db_path = Path("data/metrics.db")

if not db_path.exists():
    st.warning("No metrics found. Run the pipeline first!")
    st.code("python pipeline.py --prompt prompts/hello_world.md --iters 10")
else:
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Hämta data
    df = pd.read_sql(
        """
        SELECT iteration, reward, pass_rate, complexity, arm_selected, timestamp, 
               model_source, blocked, block_reasons, optimizer_state, optimizer_action
        FROM metrics
        ORDER BY iteration
    """,
        conn,
    )

    if df.empty:
        st.warning("Database is empty. Run some iterations first!")
    else:
        # Översikt
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Iterations", len(df))
        with col2:
            st.metric("Avg Reward", f"{df['reward'].mean():.2f}")
        with col3:
            st.metric("Pass Rate", f"{df['pass_rate'].mean() * 100:.1f}%")
        with col4:
            st.metric("Best Reward", f"{df['reward'].max():.2f}")

        # Kreativitetsbonus sektion
        col5, col6 = st.columns(2)
        with col5:
            # Räkna kreativa lösningar
            creative_count = len(df[df["reward"] > df["reward"].mean() + 5])
            st.metric("Creative Solutions", creative_count)
        with col6:
            # Visa högsta kreativitetsbonus
            max_bonus = df["reward"].max() - 30  # Baserat på max reward
            st.metric("Max Creativity Bonus", f"{max_bonus:.1f}")

        # PolicyAgent sektion
        st.subheader("🔒 PolicyAgent Security")
        col7, col8, col9 = st.columns(3)

        with col7:
            blocked_count = len(df[df["blocked"] == 1])
            st.metric(
                "Blocked Code",
                blocked_count,
                delta=f"{blocked_count / len(df) * 100:.1f}%",
            )
        with col8:
            safe_count = len(df[df["blocked"] == 0])
            st.metric(
                "Safe Code", safe_count, delta=f"{safe_count / len(df) * 100:.1f}%"
            )
        with col9:
            if "model_source" in df.columns:
                lm_studio_count = len(df[df["model_source"] == "lm_studio"])
                st.metric("LM Studio Usage", lm_studio_count)

        # PromptOptimizer sektion
        st.subheader("🤖 PromptOptimizer Agent")
        col10, col11, col12 = st.columns(3)

        with col10:
            if "optimizer_action" in df.columns:
                optimizer_actions = df["optimizer_action"].dropna()
                no_change_count = len(
                    optimizer_actions[optimizer_actions == "no_change"]
                )
                st.metric("No Changes", no_change_count)
        with col11:
            if "optimizer_action" in df.columns:
                mutations_count = len(
                    optimizer_actions[optimizer_actions != "no_change"]
                )
                st.metric("Prompt Mutations", mutations_count)
        with col12:
            if "optimizer_action" in df.columns:
                unique_actions = optimizer_actions.nunique()
                st.metric("Unique Actions", unique_actions)

        # Grafer
        st.subheader("📈 Learning Curves")

        col1, col2 = st.columns(2)

        with col1:
            st.line_chart(
                df.set_index("iteration")[["reward", "pass_rate"]],
                use_container_width=True,
            )

        with col2:
            # Strategy evolution over time
            st.subheader("Strategy Evolution")
            strategy_evolution = (
                df.groupby(["iteration", "arm_selected"]).size().unstack(fill_value=0)
            )
            st.area_chart(strategy_evolution)

        # Tabs för olika vyer
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Metrics",
                "🔒 Policy Violations",
                "🤖 Model Sources",
                "🧠 PromptOptimizer",
            ]
        )

        with tab1:
            st.subheader("📋 Detailed Metrics")
            st.dataframe(
                df[
                    [
                        "iteration",
                        "arm_selected",
                        "reward",
                        "pass_rate",
                        "complexity",
                        "blocked",
                    ]
                ],
                use_container_width=True,
            )

        with tab2:
            st.subheader("🚨 Blocked Code Analysis")
            if blocked_count > 0:
                blocked_df = df[df["blocked"] == 1].copy()
                st.dataframe(
                    blocked_df[
                        ["iteration", "arm_selected", "block_reasons", "reward"]
                    ],
                    use_container_width=True,
                )

                # Analysera block-orsaker
                if "block_reasons" in blocked_df.columns:
                    reasons = []
                    for reasons_str in blocked_df["block_reasons"].dropna():
                        reasons.extend(reasons_str.split("; "))

                    if reasons:
                        st.subheader("Block Reasons Distribution")
                        reason_counts = pd.Series(reasons).value_counts()
                        st.bar_chart(reason_counts)
            else:
                st.success("🎉 No code has been blocked by PolicyAgent!")

        with tab3:
            st.subheader("🤖 Model Source Distribution")
            if "model_source" in df.columns:
                source_counts = df["model_source"].value_counts()
                st.bar_chart(source_counts)

                st.dataframe(
                    df[["iteration", "model_source", "reward", "blocked"]],
                    use_container_width=True,
                )

        with tab4:
            st.subheader("🧠 PromptOptimizer Analysis")
            if "optimizer_action" in df.columns:
                # Action distribution
                action_counts = df["optimizer_action"].value_counts()
                st.subheader("Action Distribution")
                st.bar_chart(action_counts)

                # Action over time
                st.subheader("Actions Over Time")
                action_timeline = df[["iteration", "optimizer_action"]].copy()
                action_timeline = action_timeline[
                    action_timeline["optimizer_action"] != "no_change"
                ]
                if not action_timeline.empty:
                    st.line_chart(action_timeline.set_index("iteration"))
                else:
                    st.info("No prompt mutations yet")

                # Detailed table
                st.subheader("Optimization Details")
                optimizer_df = df[
                    ["iteration", "optimizer_action", "reward", "passed", "blocked"]
                ].copy()
                st.dataframe(optimizer_df, use_container_width=True)
            else:
                st.info("No PromptOptimizer data available")

        # Auto-refresh
        if st.sidebar.checkbox("Auto-refresh", value=False):
            st.rerun()

# Sidebar info
st.sidebar.header("ℹ️ Info")
st.sidebar.markdown("""
**CodeConductor** är ett RL-baserat system för att optimera kodgenerering.

### Komponenter:
- **LinUCB Bandit**: Väljer strategi
- **PolicyAgent**: Säkerhetskontroller
- **PromptOptimizerAgent**: Optimiserar prompts
- **Mock Cursor/LM Studio**: Genererar kod
- **Pytest**: Validerar output
- **Radon**: Mäter komplexitet

### Kör pipeline:
```bash
python pipeline.py \\
  --prompt prompts/hello_world.md \\
  --iters 10 \\
  --online  # Använd LM Studio
```
""")
