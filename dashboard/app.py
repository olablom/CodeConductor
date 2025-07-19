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
        SELECT iteration, reward, pass_rate, complexity, arm_selected, timestamp
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

        # Detaljerad tabell
        st.subheader("📋 Detailed Metrics")
        st.dataframe(
            df[["iteration", "arm_selected", "reward", "pass_rate", "complexity"]],
            use_container_width=True,
        )

        # Auto-refresh
        if st.sidebar.checkbox("Auto-refresh", value=False):
            st.rerun()

# Sidebar info
st.sidebar.header("ℹ️ Info")
st.sidebar.markdown("""
**CodeConductor** är ett RL-baserat system för att optimera kodgenerering.

### Komponenter:
- **LinUCB Bandit**: Väljer strategi
- **Mock Cursor**: Genererar kod
- **Pytest**: Validerar output
- **Radon**: Mäter komplexitet

### Kör pipeline:
```bash
python pipeline.py \\
  --prompt prompts/hello_world.md \\
  --iters 10
```
""")
