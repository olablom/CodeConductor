#!/usr/bin/env python3
"""
Test dashboard data loading without Streamlit
"""

import sys
from pathlib import Path

# Add ui directory to path
sys.path.append("ui")

from ui.dashboard import DashboardDataLoader, DashboardVisualizer


def test_dashboard_data():
    """Test dashboard data loading"""

    print("🧪 Testing Dashboard Data Loading...")

    # Initialize data loader
    data_loader = DashboardDataLoader()

    # Load all data
    print("\n📊 Loading Q-table data...")
    q_table_df = data_loader.load_q_table_data()
    print(f"   Loaded {len(q_table_df)} Q-table entries")

    print("\n📈 Loading learning metrics...")
    learning_df = data_loader.load_learning_metrics()
    print(f"   Loaded {len(learning_df)} learning episodes")

    print("\n🤖 Loading RL history...")
    rl_history_df = data_loader.load_rl_history()
    print(f"   Loaded {len(rl_history_df)} RL history entries")

    print("\n🔍 Loading pipeline metrics...")
    pipeline_df = data_loader.load_pipeline_metrics()
    print(f"   Loaded {len(pipeline_df)} pipeline iterations")

    # Test visualizations
    print("\n🎨 Testing visualizations...")
    visualizer = DashboardVisualizer()

    if not q_table_df.empty:
        heatmap_fig = visualizer.create_q_value_heatmap(q_table_df)
        print("   ✅ Q-value heatmap created")

    if not learning_df.empty:
        learning_fig = visualizer.create_learning_curve(learning_df)
        print("   ✅ Learning curve created")

    if not rl_history_df.empty:
        perf_fig = visualizer.create_agent_performance_chart(rl_history_df)
        print("   ✅ Agent performance chart created")

    if not pipeline_df.empty:
        monitoring_fig = visualizer.create_pipeline_monitoring(pipeline_df)
        print("   ✅ Pipeline monitoring chart created")

    print("\n🎉 Dashboard data loading test completed!")

    # Show some sample data
    if not q_table_df.empty:
        print(f"\n📊 Sample Q-table data:")
        print(f"   Average Q-value: {q_table_df['q_value'].mean():.3f}")
        print(f"   Max Q-value: {q_table_df['q_value'].max():.3f}")
        print(f"   Total visits: {q_table_df['visit_count'].sum()}")

    if not pipeline_df.empty:
        print(f"\n📈 Sample pipeline data:")
        print(f"   Average reward: {pipeline_df['reward'].mean():.3f}")
        print(f"   Average pass rate: {pipeline_df['pass_rate'].mean():.3f}")
        print(f"   Total iterations: {len(pipeline_df)}")


if __name__ == "__main__":
    test_dashboard_data()
