# 🚀 CodeConductor v2.0 Quick Start

**Get up and running in 5 minutes!**

---

## ⚡ Super Quick Start

```bash
# 1. Clone and start
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor
docker-compose up --build

# 2. Open in browser
# GUI: http://localhost:8501
# API: http://localhost:8000
```

---

## 🎯 What You'll See

### 1. **Multi-Agent Discussion** (Tab 1)

- Watch 3 AI agents collaborate on code generation
- See consensus-building in real-time
- Understand different perspectives on implementation

### 2. **Human Approval** (Tab 2)

- Review generated code
- Provide thumbs up/down feedback
- Add ratings and comments
- Approve, reject, or request changes

### 3. **Learning Metrics** (Tab 3)

- View reinforcement learning progress
- See reward curves and performance
- Analyze strategy effectiveness

### 4. **Generated Code** (Tab 4)

- View latest generated code
- Download single files or complete projects
- See project structure visualization

### 5. **Project History** (Tab 5)

- Browse past generations
- Compare different approaches
- Track system evolution

### 6. **Feedback Analytics** (Tab 6)

- View human feedback statistics
- See approval rates and trends
- Analyze comment patterns

---

## 🧪 Try These Examples

### Example 1: Simple Function

```
Prompt: "Create a function that validates email addresses"
```

### Example 2: REST API

```
Prompt: "Create a simple REST API with user authentication"
```

### Example 3: Multi-File Project

```bash
# In terminal, run:
python pipeline.py --prompt prompts/hello_world.md --multi-file --iters 1 --mock
```

---

## 🔧 Key Features to Explore

### **Multi-Agent Collaboration**

- **CodeGenAgent**: Implementation strategy
- **ArchitectAgent**: Design patterns
- **ReviewerAgent**: Code quality & security
- **Orchestrator**: Consensus coordination

### **Human-in-the-Loop**

- Review and approve generated code
- Provide feedback with ratings
- Request modifications
- Track decision history

### **Reinforcement Learning**

- Q-learning optimization
- Strategy selection
- Performance tracking
- Learning curves

### **Multi-File Projects**

- Complete project generation
- Automated testing
- ZIP download
- Project structure visualization

---

## 📊 API Access

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

### API Documentation

Visit: http://localhost:8000/docs

---

## 🎓 Learning Path

### **Beginner** (30 min)

1. Start the system
2. Try a simple prompt
3. Watch the multi-agent discussion
4. Approve/reject some code
5. View the learning metrics

### **Intermediate** (1 hour)

1. Run the pipeline multiple times
2. Analyze learning curves
3. Generate multi-file projects
4. Test the generated code
5. Explore API endpoints

### **Advanced** (2+ hours)

1. Modify configuration parameters
2. Add custom policy rules
3. Experiment with different prompts
4. Study the codebase
5. Extend the system

---

## 🔍 Troubleshooting

### **Port Already in Use**

```bash
# Check what's using the port
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Mac/Linux

# Kill the process or change ports
```

### **Docker Issues**

```bash
# Rebuild containers
docker-compose down
docker-compose up --build
```

### **Local Setup Issues**

```bash
# Install dependencies
pip install -r requirements.txt

# Start manually
python start_codeconductor.py
```

---

## 📚 Next Steps

1. **Read the Full Tutorial**: `tutorials/CodeConductor_Tutorial.ipynb`
2. **Follow the Workshop**: `tutorials/Workshop_Guide.md`
3. **Check Documentation**: `DEPLOYMENT.md`
4. **Explore the Codebase**: Study `agents/`, `integrations/`, `storage/`

---

## 🎉 You're Ready!

You now have access to a cutting-edge multi-agent AI code generation system with:

- ✅ Human-in-the-loop approval
- ✅ Reinforcement learning optimization
- ✅ Multi-file project generation
- ✅ REST API access
- ✅ Complete documentation

**Happy coding! 🎼**
