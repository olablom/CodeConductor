# 🎼 CodeConductor v2.0 Workshop Guide

**Multi-Agent AI Code Generation with Reinforcement Learning**

---

## 📋 Workshop Overview

**Duration**: 2-3 hours  
**Prerequisites**: Basic Python knowledge, Git, Docker (optional)  
**Learning Outcomes**: Understand multi-agent AI systems, human-in-the-loop workflows, and RL optimization

---

## 🎯 Learning Objectives

By the end of this workshop, you will:

- ✅ **Understand Multi-Agent Collaboration** - How AI agents work together to generate code
- ✅ **Experience Human-in-the-Loop** - The importance of human oversight in AI systems
- ✅ **See RL in Action** - How reinforcement learning optimizes code generation
- ✅ **Generate Complete Projects** - Create multi-file applications with tests
- ✅ **Analyze Performance** - Study learning curves and system metrics
- ✅ **Integrate via API** - Use the REST API for programmatic access

---

## 🚀 Setup Instructions

### Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/olablom/CodeConductor.git
cd CodeConductor

# 2. Start with Docker (recommended)
docker-compose up --build

# 3. Or start locally
python start_codeconductor.py
```

### Verify Installation

Once running, you should have access to:

- **📊 API**: http://localhost:8000
- **🎨 GUI**: http://localhost:8501
- **📚 API Docs**: http://localhost:8000/docs

---

## 🧪 Workshop Activities

### Activity 1: Understanding the Multi-Agent System (30 min)

**Objective**: Explore how three AI agents collaborate to generate code

#### Step 1: Agent Introduction

CodeConductor uses three specialized agents:

1. **🤖 CodeGenAgent** - Implementation Strategy Expert

   - Analyzes prompts and suggests approaches
   - Considers code structure and patterns
   - Provides strategic guidance

2. **🏗️ ArchitectAgent** - Software Architecture Expert

   - Evaluates design patterns
   - Ensures scalability and maintainability
   - Suggests modular approaches

3. **🔍 ReviewerAgent** - Code Quality & Security Expert
   - Reviews code for quality and security
   - Identifies potential issues
   - Ensures production standards

#### Step 2: Watch Agents in Action

1. Open the GUI at http://localhost:8501
2. Go to the "Agent Discussion" tab
3. Enter this prompt: `"Create a simple REST API with user authentication"`
4. Click "Start Discussion" and observe:
   - How each agent analyzes the prompt
   - The consensus-building process
   - The final approach selection

#### Step 3: Discussion Questions

- What different perspectives did each agent bring?
- How did the agents reach consensus?
- What would happen if agents disagreed?

---

### Activity 2: Human-in-the-Loop Approval (20 min)

**Objective**: Experience the human oversight workflow

#### Step 1: Generate Code

1. In the GUI, go to "Human Approval" tab
2. Enter a prompt: `"Create a password validation function"`
3. Click "Generate Code" and wait for the multi-agent discussion

#### Step 2: Review and Approve

1. **Review the generated code** - Check for:

   - Functionality (does it do what was asked?)
   - Security (are there vulnerabilities?)
   - Quality (is it well-written?)

2. **Provide feedback**:

   - Use the thumbs up/down buttons
   - Add a rating (-5 to +5)
   - Write comments explaining your decision

3. **Make a decision**:
   - **Approve**: Code is ready to use
   - **Reject**: Code needs significant changes
   - **Request Changes**: Code needs minor modifications

#### Step 3: Discussion Questions

- What factors influenced your approval decision?
- How did the agent discussion help you evaluate the code?
- What would you change about the generated code?

---

### Activity 3: Reinforcement Learning Exploration (25 min)

**Objective**: Understand how the system learns and improves

#### Step 1: Run Multiple Iterations

```bash
# Run the pipeline multiple times
python pipeline.py --prompt prompts/calculator.md --iters 10 --mock
```

#### Step 2: Analyze Learning Metrics

1. In the GUI, go to "Learning Metrics" tab
2. Observe:
   - **Reward Curve**: How rewards change over time
   - **Pass Rate**: Success rate of generated code
   - **Strategy Distribution**: Which approaches work best
   - **Q-Learning Progress**: How the system learns

#### Step 3: Experiment with Parameters

1. Check the configuration file: `config/base.yaml`
2. Try modifying:

   - `bandit.alpha`: Exploration vs exploitation balance
   - `rewards.test_pass`: Weight of test success
   - `rewards.complexity`: Weight of code complexity

3. Run the pipeline again and compare results

#### Step 4: Discussion Questions

- How did the system's performance change over iterations?
- Which strategies were most successful?
- How did parameter changes affect learning?

---

### Activity 4: Multi-File Project Generation (30 min)

**Objective**: Generate complete, production-ready projects

#### Step 1: Generate a Multi-File Project

```bash
# Generate a complete project
python pipeline.py --prompt prompts/hello_world.md --multi-file --iters 1 --mock
```

#### Step 2: Explore the Generated Project

1. Navigate to `data/generated/iter_0_project/`
2. Examine the project structure:
   ```
   project/
   ├── main.py              # Main application
   ├── utils.py             # Utility functions
   ├── config.py            # Configuration
   ├── requirements.txt     # Dependencies
   ├── README.md           # Documentation
   ├── tests/              # Test suite
   │   ├── test_main.py
   │   └── test_utils.py
   └── .gitignore          # Git ignore
   ```

#### Step 3: Test the Generated Code

```bash
# Navigate to the project directory
cd data/generated/iter_0_project

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run the application
python main.py
```

#### Step 4: Download and Share

1. In the GUI, go to "Generated Code" tab
2. Use the "Download Full Project (ZIP)" button
3. Extract and examine the project structure

#### Step 5: Discussion Questions

- How complete was the generated project?
- What would you add or modify?
- How does this compare to manually creating a project?

---

### Activity 5: API Integration (20 min)

**Objective**: Use the REST API for programmatic access

#### Step 1: Explore API Documentation

1. Visit http://localhost:8000/docs
2. Examine the available endpoints:
   - `/health` - System health check
   - `/auth/register` - User registration
   - `/auth/login` - User authentication
   - `/generate` - Code generation (if available)

#### Step 2: Test API Endpoints

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(f"Status: {response.json()}")

# Register a user
auth_data = {"username": "workshop_user", "password": "workshop_pass"}
response = requests.post("http://localhost:8000/auth/register", json=auth_data)
print(f"Registration: {response.status_code}")

# Login
response = requests.post("http://localhost:8000/auth/login", json=auth_data)
if response.status_code == 200:
    token = response.json()["access_token"]
    print(f"Login successful, token: {token[:20]}...")
```

#### Step 3: Build a Simple Client

Create a simple Python script that:

1. Authenticates with the API
2. Sends a code generation request
3. Downloads the generated code

#### Step 4: Discussion Questions

- What are the advantages of API access?
- How could you integrate this into your own applications?
- What additional endpoints would be useful?

---

## 🔬 Advanced Experiments

### Experiment 1: Custom Policy Rules

1. Examine `agents/policy_agent.py`
2. Add a custom security rule
3. Test with potentially dangerous code
4. Observe how the system blocks unsafe code

### Experiment 2: Prompt Optimization

1. Create different versions of the same prompt
2. Run each through the pipeline
3. Compare results and learning curves
4. Identify what makes a good prompt

### Experiment 3: Agent Customization

1. Study the agent implementations in `agents/`
2. Modify an agent's behavior
3. Test the impact on code generation
4. Observe changes in consensus patterns

---

## 📊 Assessment and Reflection

### Self-Assessment Questions

1. **Multi-Agent Understanding**:

   - Can you explain how the three agents work together?
   - What happens when agents disagree?

2. **Human-in-the-Loop**:

   - When would you approve vs reject generated code?
   - How does human feedback improve the system?

3. **Reinforcement Learning**:

   - How does the system learn from experience?
   - What metrics indicate successful learning?

4. **Project Generation**:
   - What makes a generated project production-ready?
   - How complete was the generated code?

### Group Discussion Topics

1. **Ethics and Safety**: How do we ensure AI-generated code is safe and ethical?
2. **Human Oversight**: What level of human involvement is appropriate?
3. **Learning and Improvement**: How can we measure and improve AI system performance?
4. **Real-World Applications**: Where could this technology be most valuable?

---

## 🎓 Next Steps

### Further Learning

1. **Study the Codebase**: Explore the implementation details
2. **Read the Documentation**: Check DEPLOYMENT.md and API docs
3. **Experiment**: Try different prompts and configurations
4. **Extend the System**: Add new agents or features

### Resources

- **GitHub Repository**: https://github.com/olablom/CodeConductor
- **API Documentation**: http://localhost:8000/docs
- **Configuration Guide**: See config/base.yaml
- **Deployment Guide**: See DEPLOYMENT.md

### Community

- **Questions**: Open an issue on GitHub
- **Contributions**: Submit pull requests
- **Feedback**: Share your workshop experience

---

## 🎉 Workshop Completion

Congratulations! You've successfully:

✅ **Explored Multi-Agent AI Systems**  
✅ **Experienced Human-in-the-Loop Workflows**  
✅ **Witnessed Reinforcement Learning in Action**  
✅ **Generated Complete Projects**  
✅ **Analyzed System Performance**  
✅ **Integrated via REST API**

You now have hands-on experience with cutting-edge AI code generation technology!

---

**Happy coding! 🎼**
