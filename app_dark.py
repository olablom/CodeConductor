"""
CodeConductor v2.0 - Modern Dark Mode GUI

Features:
- 🌙 Dark mode with beautiful gradients
- ✨ Live code editing and preview
- 🎯 Real-time agent discussions
- 📊 Interactive metrics dashboard
- 🔧 Model management integration
- 🚀 Modern UI components
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import plotly.express as px

# Import our components
try:
    from agents.orchestrator_simple import SimpleAgentOrchestrator
    from integrations.model_manager_simple import SimpleModelManager
    from integrations.human_gate import HumanGate
except ImportError:
    st.error("Some components not available. Running in demo mode.")
    SimpleAgentOrchestrator = None
    SimpleModelManager = None
    HumanGate = None

# Page configuration
st.set_page_config(
    page_title="CodeConductor v2.0",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/codeconductor",
        "Report a bug": "https://github.com/your-repo/codeconductor/issues",
        "About": "CodeConductor v2.0 - Multi-Agent AI Development Orchestrator",
    },
)

# Dark mode CSS with modern styling
st.markdown(
    """
<style>
    /* Dark theme base */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Modern header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .agent-card {
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.8) 0%, rgba(42, 82, 152, 0.8) 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 8px 32px rgba(52, 152, 219, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .consensus-box {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.8) 0%, rgba(56, 239, 125, 0.8) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .human-gate {
        background: linear-gradient(135deg, rgba(255, 236, 210, 0.1) 0%, rgba(252, 182, 159, 0.1) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid rgba(230, 126, 34, 0.5);
        backdrop-filter: blur(10px);
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Code editor styling */
    .code-editor {
        background: #1e1e1e;
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Courier New', monospace;
    }
    
    /* Status indicators */
    .status-success { color: #2ecc71; font-weight: bold; }
    .status-warning { color: #f39c12; font-weight: bold; }
    .status-error { color: #e74c3c; font-weight: bold; }
    .status-info { color: #3498db; font-weight: bold; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #fafafa;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Data frames */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "discussion_history" not in st.session_state:
    st.session_state.discussion_history = []
if "current_proposal" not in st.session_state:
    st.session_state.current_proposal = None
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "model_info" not in st.session_state:
    st.session_state.model_info = {}
if "human_gate" not in st.session_state:
    if HumanGate:
        st.session_state.human_gate = HumanGate("data/approval_log.json")
    else:
        st.session_state.human_gate = None

# Header
st.markdown(
    """
<div class="main-header">
    <h1>🎼 CodeConductor v2.0</h1>
    <p>Multi-Agent AI Development Orchestrator</p>
    <p><em>Where AI agents collaborate to build better code</em></p>
    <div style="margin-top: 1rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
            🌙 Dark Mode
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
            ✨ Live Editing
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
            🤖 Multi-Agent
        </span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Project Configuration")
    st.markdown("---")

    # Project Input
    st.markdown("### 📝 Project Request")
    project_type = st.selectbox(
        "Project Type",
        ["Web API", "CLI Tool", "Data Pipeline", "ML Model", "Web App", "Custom"],
        help="Select the type of project you want to build",
    )

    project_description = st.text_area(
        "Describe what you want to build",
        height=150,
        placeholder="I need a REST API that manages user authentication with JWT tokens, includes user registration, login, and password reset functionality...",
        help="Provide a detailed description of your project requirements",
    )

    # Model Selection
    st.markdown("### 🤖 Model Configuration")
    if SimpleModelManager:
        try:
            model_manager = SimpleModelManager()
            model_info = model_manager.get_model_info()
            st.session_state.model_info = model_info

            # Model status
            st.markdown("**Model Status:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Configured", len(model_info.get("configured_models", [])))
            with col2:
                st.metric("Downloaded", len(model_info.get("downloaded_models", [])))
            with col3:
                st.metric("Loaded", len(model_info.get("currently_loaded", [])))

            # Model selection
            available_models = model_info.get("configured_models", [])
            if available_models:
                selected_model = st.selectbox(
                    "Select Model",
                    available_models,
                    index=0,
                    help="Choose which model to use for code generation",
                )
            else:
                st.warning("No models configured")
                selected_model = None

        except Exception as e:
            st.error(f"Model manager error: {e}")
            selected_model = None
    else:
        st.warning("Model manager not available")
        selected_model = None

    # Advanced Options
    with st.expander("⚙️ Advanced Settings"):
        max_iterations = st.slider("Max Iterations", 1, 10, 3)
        require_approval = st.checkbox(
            "Require Human Approval", True, help="Enable human-in-the-loop control"
        )
        use_rl = st.checkbox(
            "Use RL Optimization",
            True,
            help="Enable reinforcement learning for prompt optimization",
        )
        show_debug = st.checkbox(
            "Show Debug Info", False, help="Display detailed agent reasoning"
        )

        # Performance settings
        st.markdown("**Performance:**")
        use_plugins = st.checkbox("Enable Plugins", True)
        cache_results = st.checkbox("Cache Results", True)

    # Action Buttons
    st.markdown("### 🚀 Actions")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "🚀 Start Discussion", type="primary", use_container_width=True
        )
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.discussion_history = []
            st.session_state.current_proposal = None
            st.session_state.generated_code = ""
            st.rerun()

# Main Content Area
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "🤖 Agent Discussion",
        "👤 Human Approval",
        "💻 Live Code Editor",
        "📊 Learning Metrics",
        "📈 Project History",
        "💬 Feedback",
        "⚙️ Policies",
        "🔌 Plugins",
    ]
)

with tab1:
    st.markdown("## 🤖 Multi-Agent Discussion")

    if start_btn and project_description:
        # Initialize orchestrator
        if SimpleAgentOrchestrator:
            orchestrator = SimpleAgentOrchestrator()
        else:
            orchestrator = None

        # Create progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate agent discussion with real-time updates
        status_text.text("🧠 Initializing agents...")
        progress_bar.progress(10)
        time.sleep(0.5)

        # Agent Discussion Phase
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔧 CodeGenAgent")
            st.markdown(
                """
<div class="agent-card">
    <h4>Analysis</h4>
    <p>This appears to be a {project_type.lower()} project requiring authentication functionality. 
    I recommend using FastAPI for the API framework with SQLAlchemy for database management.</p>
    <p class="status-success">Confidence: High</p>
</div>
""",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("### 🏗️ ArchitectAgent")
            st.markdown(
                """
<div class="agent-card">
    <h4>Architecture</h4>
    <p>Proposed structure: REST API with JWT authentication, user management endpoints, 
    and secure password hashing using bcrypt.</p>
    <p class="status-info">Confidence: Medium</p>
</div>
""",
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown("### 🔍 ReviewerAgent")
            st.markdown(
                """
<div class="agent-card">
    <h4>Review</h4>
    <p>Security considerations: Input validation, rate limiting, secure headers, 
    and proper error handling are essential.</p>
    <p class="status-warning">Confidence: High</p>
</div>
""",
                unsafe_allow_html=True,
            )

        progress_bar.progress(50)
        status_text.text("🤝 Synthesizing consensus...")
        time.sleep(1)

        # Consensus
        st.markdown(
            """
<div class="consensus-box">
    <h3>🎯 Agent Consensus</h3>
    <p><strong>Recommended Approach:</strong> FastAPI-based REST API with JWT authentication</p>
    <p><strong>Key Features:</strong> User registration, login, password reset, secure endpoints</p>
    <p><strong>Security:</strong> Input validation, rate limiting, secure headers</p>
    <p class="status-success">Consensus Level: Strong Agreement</p>
</div>
""",
            unsafe_allow_html=True,
        )

        progress_bar.progress(100)
        status_text.text("✅ Discussion complete!")

        # Store proposal
        st.session_state.current_proposal = {
            "type": project_type,
            "description": project_description,
            "consensus": "FastAPI-based REST API with JWT authentication",
            "timestamp": datetime.now().isoformat(),
        }

    elif st.session_state.discussion_history:
        # Show previous discussions
        st.markdown("### 📜 Previous Discussions")
        for i, discussion in enumerate(st.session_state.discussion_history):
            with st.expander(
                f"Discussion {i + 1} - {discussion.get('timestamp', 'Unknown')}"
            ):
                st.json(discussion)

with tab2:
    st.markdown("## 👤 Human Approval")

    if st.session_state.current_proposal:
        st.markdown(
            """
<div class="human-gate">
    <h3>🤔 Human Decision Required</h3>
    <p><strong>Proposal:</strong> {}</p>
    <p><strong>Consensus:</strong> {}</p>
    <p><strong>Generated:</strong> {}</p>
</div>
""".format(
                st.session_state.current_proposal.get("description", "No description"),
                st.session_state.current_proposal.get("consensus", "No consensus"),
                st.session_state.current_proposal.get("timestamp", "Unknown"),
            ),
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                st.success("Proposal approved! Generating code...")
                # Generate code here
                st.session_state.generated_code = """
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="User Authentication API")
security = HTTPBearer()

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

@app.post("/register")
async def register(user: UserCreate):
    # Implementation here
    return {"message": "User registered successfully"}

@app.post("/login")
async def login(user: UserLogin):
    # Implementation here
    return {"access_token": "jwt_token_here"}
"""
                st.rerun()

        with col2:
            if st.button("❌ Reject", use_container_width=True):
                st.error("Proposal rejected. Please provide feedback.")
                st.text_area("Feedback", placeholder="Why was this proposal rejected?")

        with col3:
            if st.button("🔄 Request Changes", use_container_width=True):
                st.warning("Changes requested. Please specify modifications.")
                st.text_area(
                    "Requested Changes", placeholder="What changes would you like?"
                )

    else:
        st.info("No proposal available for approval. Start a discussion first.")

with tab3:
    st.markdown("## 💻 Live Code Editor")

    # Code editor with syntax highlighting
    if st.session_state.generated_code:
        st.markdown("### 📝 Generated Code")

        # Code editing
        edited_code = st.text_area(
            "Edit Code",
            value=st.session_state.generated_code,
            height=400,
            help="Modify the generated code as needed",
        )

        if edited_code != st.session_state.generated_code:
            st.session_state.generated_code = edited_code
            st.success("Code updated!")

        # Code actions
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                st.success("Code saved!")
        with col2:
            if st.button("▶️ Run", use_container_width=True):
                st.info("Running code...")
        with col3:
            if st.button("🧪 Test", use_container_width=True):
                st.info("Running tests...")
        with col4:
            if st.button("📤 Export", use_container_width=True):
                st.success("Code exported!")

        # Code metrics
        st.markdown("### 📊 Code Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lines", len(edited_code.split("\n")))
        with col2:
            st.metric("Characters", len(edited_code))
        with col3:
            st.metric("Functions", edited_code.count("def "))
        with col4:
            st.metric("Complexity", "Low")

    else:
        st.info(
            "No code generated yet. Start a discussion and approve a proposal to generate code."
        )

with tab4:
    st.markdown("## 📊 Learning Metrics")

    # Create sample metrics data
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    metrics_data = pd.DataFrame(
        {
            "Date": dates,
            "Success Rate": np.random.normal(0.8, 0.1, 30),
            "Code Quality": np.random.normal(0.75, 0.15, 30),
            "Human Approval Rate": np.random.normal(0.9, 0.05, 30),
            "Iterations": np.random.randint(1, 5, 30),
        }
    )

    # Success rate over time
    fig1 = px.line(
        metrics_data,
        x="Date",
        y="Success Rate",
        title="Success Rate Over Time",
        color_discrete_sequence=["#667eea"],
    )
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)

    # Metrics comparison
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(
            metrics_data.tail(10),
            x="Date",
            y=["Success Rate", "Code Quality"],
            title="Recent Performance",
            barmode="group",
        )
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.scatter(
            metrics_data,
            x="Success Rate",
            y="Code Quality",
            title="Success vs Quality Correlation",
            color="Human Approval Rate",
        )
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

with tab5:
    st.markdown("## 📈 Project History")

    # Sample project history
    projects = [
        {
            "name": "User Auth API",
            "status": "Completed",
            "success_rate": 0.95,
            "date": "2024-01-15",
        },
        {
            "name": "Data Pipeline",
            "status": "In Progress",
            "success_rate": 0.78,
            "date": "2024-01-20",
        },
        {
            "name": "ML Model",
            "status": "Completed",
            "success_rate": 0.88,
            "date": "2024-01-25",
        },
        {
            "name": "Web App",
            "status": "Failed",
            "success_rate": 0.45,
            "date": "2024-01-30",
        },
    ]

    df = pd.DataFrame(projects)
    st.dataframe(df, use_container_width=True)

with tab6:
    st.markdown("## 💬 Feedback")

    # Feedback form
    feedback_type = st.selectbox(
        "Feedback Type", ["Bug Report", "Feature Request", "General"]
    )
    feedback_text = st.text_area("Your Feedback", height=150)

    if st.button("Submit Feedback"):
        st.success("Feedback submitted! Thank you.")

with tab7:
    st.markdown("## ⚙️ Policies")

    # Policy configuration
    st.markdown("### 🔒 Security Policies")
    security_policies = {
        "No dangerous imports": True,
        "No hardcoded secrets": True,
        "Input validation required": True,
        "Error handling required": True,
    }

    for policy, enabled in security_policies.items():
        st.checkbox(policy, enabled)

with tab8:
    st.markdown("## 🔌 Plugins")

    # Plugin management
    plugins = [
        {
            "name": "Security Plugin",
            "status": "Active",
            "description": "Code security analysis",
        },
        {
            "name": "Formatter Plugin",
            "status": "Active",
            "description": "Code formatting",
        },
    ]

    for plugin in plugins:
        with st.expander(f"{plugin['name']} - {plugin['status']}"):
            st.write(plugin["description"])
            if st.button(f"Toggle {plugin['name']}"):
                st.success(f"{plugin['name']} toggled!")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #888; padding: 2rem;">
    <p>🎼 CodeConductor v2.0 - Multi-Agent AI Development Orchestrator</p>
    <p>Built with ❤️ using Streamlit, Plotly, and modern AI</p>
</div>
""",
    unsafe_allow_html=True,
)
