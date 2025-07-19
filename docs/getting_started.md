# 🎼 CodeConductor v2.0 - Getting Started Guide

**AI-powered code generation with multi-agent discussion, RL optimization, and distributed execution**

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CodeConductor.git
cd CodeConductor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. First Run

```bash
# Run with a simple prompt
python pipeline.py --prompt "Create a simple calculator function" --iters 3 --mock

# Run with distributed execution
python pipeline.py --prompt "Build a REST API" --iters 5 --distributed --mock
```

### 3. Start the Dashboard

```bash
# Main dashboard
streamlit run app.py

# Analytics dashboard
streamlit run analytics/dashboard.py
```

---

## 🎯 Core Features

### **Multi-Agent Discussion**

- **CodeGenAgent**: Generates code based on prompts
- **ArchitectAgent**: Designs system architecture
- **ReviewerAgent**: Reviews and suggests improvements
- **PolicyAgent**: Ensures code safety and compliance

### **Reinforcement Learning**

- **PromptOptimizerAgent**: Continuously improves prompts using Q-learning
- **Bandit algorithms**: Optimizes strategy selection
- **Reward system**: Based on test results, complexity, and human feedback

### **Distributed Execution**

- **Celery + Redis**: Parallel agent execution
- **Plugin architecture**: Extensible with custom plugins
- **GitHub integration**: Automated PR analysis

### **ML Analytics**

- **Quality prediction**: ML models predict code quality
- **Trend analysis**: Track improvements over time
- **Warning system**: Proactive alerts for potential issues

---

## 📁 Project Structure

```
CodeConductor/
├── agents/                 # Multi-agent system
│   ├── base_agent.py      # Abstract base class
│   ├── orchestrator.py    # Agent coordination
│   └── orchestrator_distributed.py  # Distributed version
├── bandits/               # RL algorithms
│   └── q_learning.py      # Q-learning implementation
├── plugins/               # Plugin system
│   ├── base_simple.py     # Plugin base classes
│   ├── security_plugin.py # Security analysis
│   └── formatter_plugin.py # Code formatting
├── integrations/          # External integrations
│   ├── lm_studio.py       # LM Studio integration
│   ├── celery_app.py      # Celery configuration
│   └── github_webhook.py  # GitHub webhook
├── analytics/             # ML analytics
│   ├── ml_predictor.py    # Quality prediction
│   └── dashboard.py       # Analytics dashboard
├── config/                # Configuration
│   └── base.yaml          # Main configuration
├── data/                  # Data storage
│   └── metrics.db         # SQLite database
├── tests/                 # Test suite
├── pipeline.py            # Main pipeline
└── app.py                 # Streamlit dashboard
```

---

## 🎮 Usage Examples

### Basic Code Generation

```bash
# Generate a simple function
python pipeline.py --prompt "Create a function to calculate fibonacci numbers" --iters 1

# Generate with multiple iterations
python pipeline.py --prompt "Build a web scraper" --iters 5 --mock
```

### Advanced Features

```bash
# Use distributed execution
python pipeline.py --prompt "Create a machine learning pipeline" --distributed --iters 3

# Enable plugins
python pipeline.py --prompt "Build a secure authentication system" --iters 2

# Use specific model
python pipeline.py --prompt "Generate a React component" --model lm_studio --iters 1
```

### GitHub Integration

```bash
# Start webhook server
python integrations/github_webhook.py

# Configure in GitHub:
# 1. Go to repository settings
# 2. Add webhook: http://your-server:5000/webhook
# 3. Select events: pull_request, push
```

---

## 🔧 Configuration

### Main Configuration (`config/base.yaml`)

```yaml
# Agent settings
agents:
  codegen:
    temperature: 0.7
    max_tokens: 2000
  architect:
    temperature: 0.5
    max_tokens: 1500

# RL settings
bandits:
  epsilon: 0.1
  learning_rate: 0.1
  strategies:
    - conservative
    - balanced
    - exploratory

# Distributed execution
distributed:
  enabled: true
  broker_url: "redis://localhost:6379/0"
  concurrency: 4

# GitHub integration
github_webhook:
  enabled: false
  secret: ""
  token: ""
```

### Environment Variables

```bash
# LM Studio settings
export LM_STUDIO_URL="http://localhost:1234/v1"
export LM_STUDIO_API_KEY=""

# GitHub settings
export GITHUB_TOKEN="your_github_token"
export GITHUB_WEBHOOK_SECRET="your_webhook_secret"
```

---

## 📊 Monitoring & Analytics

### Dashboard Access

1. **Main Dashboard**: `http://localhost:8501`

   - Real-time pipeline monitoring
   - Agent discussion visualization
   - RL learning curves

2. **Analytics Dashboard**: `http://localhost:8502`
   - ML quality predictions
   - Trend analysis
   - Performance metrics

### Key Metrics

- **Pass Rate**: Percentage of generated code that passes tests
- **Complexity Score**: Code complexity measurement
- **Reward**: RL reward based on quality and efficiency
- **Strategy Performance**: Success rates by strategy

---

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_bandits.py
pytest tests/test_plugins.py
```

### Test Specific Features

```bash
# Test plugin system
python test_plugin_integration.py

# Test distributed execution
python test_distributed.py

# Test GitHub integration
python test_github_integration.py

# Test analytics
python test_analytics.py
```

---

## 🔌 Plugin Development

### Creating a Custom Plugin

```python
# plugins/my_plugin.py
from plugins.base_simple import BaseAgentPlugin

class MyCustomPlugin(BaseAgentPlugin):
    def analyze(self, prompt: str, context: dict) -> dict:
        # Your analysis logic here
        return {
            "suggestions": ["Use async/await for better performance"],
            "score": 0.8,
            "confidence": 0.9
        }

    def act(self, prompt: str, context: dict) -> str:
        # Your action logic here
        return "Modified prompt with optimizations"
```

### Plugin Registration

Plugins are automatically discovered and loaded from the `plugins/` directory.

---

## 🚀 Advanced Usage

### Custom Agent Development

```python
# agents/custom_agent.py
from agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def analyze(self, prompt: str, context: dict) -> dict:
        # Custom analysis logic
        pass

    def act(self, prompt: str, context: dict) -> str:
        # Custom action logic
        pass

    def observe(self, result: dict) -> None:
        # Custom observation logic
        pass
```

### RL Strategy Customization

```python
# bandits/custom_bandit.py
from bandits.q_learning import QLearningBandit

class CustomBandit(QLearningBandit):
    def calculate_reward(self, result: dict) -> float:
        # Custom reward calculation
        base_reward = result.get("reward", 0)
        complexity_penalty = result.get("complexity", 0) * 0.1
        return base_reward - complexity_penalty
```

---

## 🐛 Troubleshooting

### Common Issues

1. **LM Studio Connection Failed**

   ```bash
   # Check if LM Studio is running
   curl http://localhost:1234/v1/models

   # Verify configuration
   cat config/base.yaml | grep lm_studio
   ```

2. **Redis Connection Failed**

   ```bash
   # Start Redis server
   redis-server

   # Test connection
   redis-cli ping
   ```

3. **Plugin Loading Issues**
   ```bash
   # Check plugin discovery
   python -c "from plugins.base_simple import discover_plugins; print(discover_plugins())"
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python pipeline.py --prompt "test" --iters 1 --mock
```

---

## 📚 Learning Resources

### Documentation

- [Agent Architecture](docs/agents.md)
- [RL Implementation](docs/bandits.md)
- [Plugin System](docs/plugins.md)
- [API Reference](docs/api.md)

### Examples

- [Basic Examples](examples/basic/)
- [Advanced Examples](examples/advanced/)
- [Plugin Examples](examples/plugins/)

### Community

- [GitHub Issues](https://github.com/yourusername/CodeConductor/issues)
- [Discussions](https://github.com/yourusername/CodeConductor/discussions)
- [Wiki](https://github.com/yourusername/CodeConductor/wiki)

---

## 🎯 Next Steps

1. **Run your first pipeline**: Try the quick start examples
2. **Explore the dashboard**: Monitor real-time metrics
3. **Experiment with prompts**: Test different strategies
4. **Develop plugins**: Extend functionality
5. **Contribute**: Submit issues and pull requests

---

**Happy coding with CodeConductor! 🎼✨**
