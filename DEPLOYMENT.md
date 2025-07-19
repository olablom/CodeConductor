# 🚀 CodeConductor v2.0 Deployment Guide

## Quick Start Options

### Option 1: Docker Deployment (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor

# 2. Start with Docker
docker-compose up --build

# 3. Access the system
# 📊 API: http://localhost:8000
# 🎨 GUI: http://localhost:8501
```

### Option 2: Local Development

```bash
# 1. Clone the repository
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor

# 2. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start with startup script
python start_codeconductor.py
```

### Option 3: Manual Start

```bash
# Terminal 1: Start API
uvicorn generated_api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start GUI
streamlit run app.py --server.port 8501 --server.headless true
```

## System Architecture

```
CodeConductor v2.0
├── 📊 FastAPI Server (Port 8000)
│   ├── Authentication endpoints
│   ├── Code generation API
│   └── Health monitoring
├── 🎨 Streamlit GUI (Port 8501)
│   ├── Multi-agent discussion
│   ├── Human approval interface
│   ├── Learning metrics
│   ├── Generated code viewer
│   └── Project history
└── 📁 Data Storage
    ├── Generated projects
    ├── RL learning data
    └── Human feedback
```

## Features

### ✅ Multi-Agent Collaboration

- **CodeGenAgent**: Implementation strategy
- **ArchitectAgent**: Design patterns
- **ReviewerAgent**: Code quality & security
- **AgentOrchestrator**: Consensus coordination

### ✅ Reinforcement Learning

- **PromptOptimizerAgent**: Q-learning optimization
- **PolicyAgent**: Security validation
- **RewardAgent**: Multi-factor rewards

### ✅ Human-in-the-Loop

- **HumanGate**: Approval interface
- **Feedback System**: Thumbs up/down with comments
- **Decision Logging**: Complete audit trail

### ✅ Multi-File Projects

- **Complete Project Generation**: Multiple files with structure
- **Automated Testing**: pytest integration
- **ZIP Download**: Full project export
- **Project Tree Visualization**: File structure display

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Authentication

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}'
```

## Pipeline Usage

### Single File Generation

```bash
python pipeline.py --prompt prompts/hello_world.md --iters 10 --mock
```

### Multi-File Project Generation

```bash
python pipeline.py --prompt prompts/hello_world.md --multi-file --iters 5 --mock
```

### With LM Studio (if available)

```bash
python pipeline.py --prompt prompts/calculator.md --online --iters 10
```

## Configuration

### Environment Variables

```bash
JWT_SECRET=your_secret_key
CONFIG_PATH=/app/config/base.yaml
PYTHONPATH=/app
```

### Config Files

- `config/base.yaml`: Main configuration
- `prompts/`: Input prompt templates
- `data/`: Generated projects and metrics

## Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   # Check what's using the port
   netstat -ano | findstr :8000
   # Kill the process or change ports
   ```

2. **Dependencies missing**

   ```bash
   pip install -r requirements.txt
   ```

3. **Permission issues (Linux/Mac)**
   ```bash
   chmod +x start_codeconductor.py
   ```

### Logs

- API logs: Check terminal output
- GUI logs: Check browser console
- Pipeline logs: Check terminal output

## Development

### Running Tests

```bash
pytest tests/ -v
python test_system.py
```

### Code Quality

```bash
black .
pylint agents/ integrations/ storage/
mypy .
```

## Production Deployment

### Docker Production

```bash
# Build production image
docker build -t codeconductor:v2.0 .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -p 8501:8501 \
  -e JWT_SECRET=your_production_secret \
  codeconductor:v2.0
```

### Kubernetes (Optional)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codeconductor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: codeconductor
  template:
    metadata:
      labels:
        app: codeconductor
    spec:
      containers:
        - name: codeconductor
          image: codeconductor:v2.0
          ports:
            - containerPort: 8000
            - containerPort: 8501
```

## Support

For issues and questions:

- Check the logs for error messages
- Verify all dependencies are installed
- Ensure ports are available
- Check firewall settings

**CodeConductor v2.0** - Multi-Agent AI Code Generation System
