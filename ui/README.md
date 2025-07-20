# 🎼 CodeConductor Dashboard

Real-time monitoring and visualization dashboard for the CodeConductor AI system.

## 🚀 Features

### 📊 Overview Dashboard

- **System Metrics**: Average reward, success rate, total states, total episodes
- **Learning Curve**: Reward progression with moving averages
- **Pipeline Performance**: Real-time monitoring of iterations
- **Q-Value Heatmap**: Visual representation of reinforcement learning state-action values

### 🧠 RL Metrics

- **Q-Table Statistics**: Average, max, min Q-values and visit counts
- **Q-Value Distribution**: Histogram of Q-value distribution
- **Learning Progress**: Detailed learning curves
- **Top Q-Values**: Best performing state-action pairs

### 🤖 Agent Performance

- **Reward Components**: Breakdown by agent component
- **Execution Time Analysis**: Performance over time
- **Agent Interactions**: Detailed interaction metrics

### 🔍 Pipeline Monitoring

- **Real-time Performance**: Live monitoring of pipeline metrics
- **Model Source Distribution**: Usage statistics by model source
- **Safety Analysis**: Blocked vs safe code generation

### 💰 Cost Control (Coming Soon)

- Token usage tracking
- Cost per iteration
- Model cost comparison
- Budget alerts

## 🛠️ Installation

```bash
# Install dependencies
pip install streamlit plotly pandas numpy

# Run the dashboard
streamlit run ui/dashboard.py
```

## 📊 Data Sources

The dashboard connects to multiple databases:

- **`data/qtable.db`**: Q-learning table and learning metrics
- **`data/rl_history.db`**: Reinforcement learning episode history
- **`data/metrics.db`**: Pipeline performance metrics

## 🧪 Testing

```bash
# Generate sample data
python ui/test_dashboard.py

# Test data loading (without Streamlit)
python test_dashboard_data.py
```

## 🎨 Customization

### Adding New Visualizations

1. Add new method to `DashboardVisualizer` class
2. Create corresponding page function
3. Add to navigation in `main()`

### Custom Data Sources

1. Add new method to `DashboardDataLoader` class
2. Update database connection logic
3. Add error handling for missing data

## 📈 Sample Data

The test script generates:

- **18 Q-table entries** (6 states × 3 actions)
- **50 learning episodes** with rewards
- **100 pipeline iterations** with metrics
- **20 RL episodes** with reward components

## 🔧 Configuration

### Auto-refresh

Enable auto-refresh in the sidebar to update data every 30 seconds.

### Data Filtering

Use the sidebar navigation to switch between different views and metrics.

## 🐛 Troubleshooting

### No Data Found

- Run `python ui/test_dashboard.py` to generate sample data
- Check that database files exist in `data/` directory
- Verify database schema matches expected structure

### Import Errors

- Ensure all dependencies are installed: `pip install streamlit plotly pandas numpy`
- Check Python path includes `ui/` directory

### Visualization Errors

- Verify data format matches expected schema
- Check for missing or malformed JSON data
- Ensure numeric columns contain valid values

## 📝 Development

### File Structure

```
ui/
├── dashboard.py          # Main dashboard application
├── test_dashboard.py     # Sample data generator
└── README.md            # This file
```

### Key Classes

- **`DashboardDataLoader`**: Loads data from multiple databases
- **`DashboardVisualizer`**: Creates interactive visualizations
- **`main()`**: Orchestrates the dashboard application

### Adding New Pages

1. Create page function (e.g., `show_new_page()`)
2. Add to navigation selectbox in `main()`
3. Add corresponding data loading logic
4. Create visualizations using `DashboardVisualizer`

## 🎯 Next Steps

- [ ] Cost control implementation
- [ ] Real-time data streaming
- [ ] Export functionality
- [ ] Custom alert system
- [ ] Advanced filtering options
- [ ] Mobile-responsive design
