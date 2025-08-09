# CodeConductor Test Suite

This directory contains the comprehensive test suite for CodeConductor, ensuring all components work correctly and perform within expected parameters.

## 🧪 **Quick Start**

```bash
# Run the master test suite
python test_master_simple.py --self_reflection

# Run specific component tests
python tests/test_single_agent.py
python tests/test_rag_system.py
python tests/test_performance.py
```

## 📊 **Benchmark Suite**

### **Master Test (`test_master_simple.py`)**
- **Purpose**: End-to-end validation of all core components
- **Duration**: ~37 seconds
- **Components Tested**:
  - Single Agent Performance (100% success rate)
  - RAG Functionality (100% success rate)
  - Performance Benchmarks (TTFT < 6.0s)
  - RLHF Learning (Active weight updates)

### **Single Agent Tests**
- **Fibonacci Functions**: Tests recursive/iterative implementations
- **Binary Search**: Tests algorithm correctness and edge cases
- **REST APIs**: Tests Flask endpoint generation

### **Performance Benchmarks**
- **TTFT (Time To First Token)**: Target < 6.0s
- **Tokens/Second**: Target > 40 tokens/s
- **Memory Usage**: Target < 40% VRAM
- **CPU Usage**: Target < 15%

## 🔧 **Test Configuration**

### **Environment Variables**
```bash
export CODECONDUCTOR_LOG_LEVEL=INFO
export CODECONDUCTOR_MODEL_PATH=/path/to/models
export CODECONDUCTOR_MAX_TOKENS=2048
```

### **Command Line Options**
```bash
python test_master_simple.py \
  --self_reflection \     # Enable self-reflection loop
  --agent_count 2 \       # Number of agents to use
  --quick \              # Quick mode (fewer iterations)
  --verbose              # Detailed logging
```

## 📈 **Expected Results**

| Test Category | Success Rate | Median Latency | 99th Percentile |
|---------------|--------------|----------------|-----------------|
| Single Agent  | 100%         | 5.2s           | 6.1s            |
| RAG System    | 100%         | 0.0s           | 0.1s            |
| Performance   | 100%         | 4.8s           | 12.0s           |
| RLHF Learning | 100%         | 0.5s           | 1.2s            |

## 🐛 **Troubleshooting**

### **Common Issues**

1. **VRAM Out of Memory**
   ```bash
   # Reduce model count
   export CODECONDUCTOR_MAX_MODELS=1
   ```

2. **Model Loading Failures**
   ```bash
   # Check LM Studio is running
   lms ps
   ```

3. **Test Timeouts**
   ```bash
   # Increase timeout
   export CODECONDUCTOR_TIMEOUT=60
   ```

### **Debug Mode**
```bash
# Enable debug logging
export CODECONDUCTOR_LOG_LEVEL=DEBUG
python test_master_simple.py --verbose
```

## 📝 **Adding New Tests**

1. **Create test file**: `tests/test_new_feature.py`
2. **Follow naming convention**: `test_*_feature.py`
3. **Include in master suite**: Add to `test_master_simple.py`
4. **Document expected results**: Update this README

## 🚀 **CI/CD Integration**

The test suite is designed to run in CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run CodeConductor Tests
  run: |
    python test_master_simple.py --quick
    python tests/run_benchmarks.py --agent-count=2
```

## 📊 **Performance Monitoring**

Tests automatically collect metrics:
- **Latency**: Request/response times
- **Throughput**: Tokens per second
- **Resource Usage**: CPU, memory, VRAM
- **Success Rate**: Pass/fail ratios

Results are saved to `simple_master_test_results_*.json` for analysis. 