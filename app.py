"""
CodeConductor GUI - Multi-Agent Development Dashboard

Beautiful Streamlit interface showcasing:
- Real-time multi-agent discussions
- Human-in-the-loop approval
- RL learning metrics
- Generated code preview
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Import our components
from agents.orchestrator import AgentOrchestrator
from integrations.human_gate import HumanGate

# Page configuration
st.set_page_config(
    page_title="CodeConductor",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful styling
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-high { color: #2ecc71; font-weight: bold; }
    .confidence-med { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    
    .consensus-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    
    .human-gate {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #e67e22;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "discussion_history" not in st.session_state:
    st.session_state.discussion_history = []
if "current_proposal" not in st.session_state:
    st.session_state.current_proposal = None
if "human_gate" not in st.session_state:
    st.session_state.human_gate = HumanGate("data/approval_log.json")

# Header
st.markdown(
    """
<div class="main-header">
    <h1>🎼 CodeConductor v2.0</h1>
    <p>Multi-Agent AI Development Orchestrator</p>
    <p><em>Where AI agents collaborate to build better code</em></p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("🎯 Project Configuration")
    st.markdown("---")

    # Project Input
    st.subheader("📝 Project Request")
    project_type = st.selectbox(
        "Project Type",
        ["Web API", "CLI Tool", "Data Pipeline", "ML Model", "Web App", "Custom"],
    )

    project_description = st.text_area(
        "Describe what you want to build",
        height=150,
        placeholder="I need a REST API that manages user authentication with JWT tokens, includes user registration, login, and password reset functionality...",
    )

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

    # Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "🚀 Start Discussion", type="primary", use_container_width=True
        )
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.discussion_history = []
            st.session_state.current_proposal = None
            st.rerun()

# Main Content Area
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "🤖 Agent Discussion",
        "👤 Human Approval",
        "📊 Learning Metrics",
        "💻 Generated Code",
        "📈 Project History",
        "💬 Feedback",
        "⚙️ Policies",
        "🔌 Plugins",
    ]
)

with tab1:
    st.header("🤖 Multi-Agent Discussion")

    if start_btn and project_description:
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()

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
            with st.container():
                status_text.text("🔧 CodeGenAgent analyzing implementation approach...")
                progress_bar.progress(30)
                time.sleep(1)

                # Real agent analysis
                result = orchestrator.codegen_agent.analyze(project_description, {})

                st.markdown(
                    """
                <div class="agent-card">
                    <h4>Implementation Strategy</h4>
                    <p>{}</p>
                    <p><strong>Confidence:</strong> <span class="confidence-{}">{:.0%}</span></p>
                </div>
                """.format(
                        result["approach"][:100] + "..."
                        if len(result["approach"]) > 100
                        else result["approach"],
                        "high"
                        if result["confidence"] > 0.8
                        else "med"
                        if result["confidence"] > 0.5
                        else "low",
                        result["confidence"],
                    ),
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown("### 🏗️ ArchitectAgent")
            with st.container():
                status_text.text("🏗️ ArchitectAgent analyzing design patterns...")
                progress_bar.progress(50)
                time.sleep(1)

                result = orchestrator.architect_agent.analyze(project_description, {})

                st.markdown(
                    """
                <div class="agent-card">
                    <h4>Architecture Analysis</h4>
                    <p><strong>Patterns:</strong> {}</p>
                    <p><strong>Structure:</strong> {}</p>
                    <p><strong>Scalability:</strong> {}</p>
                </div>
                """.format(
                        ", ".join(result["patterns"]),
                        result["structure"][:80] + "..."
                        if len(result["structure"]) > 80
                        else result["structure"],
                        result["scalability"].title(),
                    ),
                    unsafe_allow_html=True,
                )

        with col3:
            st.markdown("### 🔍 ReviewerAgent")
            with st.container():
                status_text.text("🔍 ReviewerAgent identifying potential issues...")
                progress_bar.progress(70)
                time.sleep(1)

                result = orchestrator.reviewer_agent.analyze(project_description, {})

                st.markdown(
                    """
                <div class="agent-card">
                    <h4>Quality Review</h4>
                    <p><strong>Security Risks:</strong> {}</p>
                    <p><strong>Quality Issues:</strong> {}</p>
                    <p><strong>Testing Needs:</strong> {}</p>
                    <p><strong>Maintainability:</strong> {}</p>
                </div>
                """.format(
                        ", ".join(result["security_risks"]),
                        ", ".join(result["quality_issues"]),
                        ", ".join(result["testing_needs"]),
                        result["maintainability"].title(),
                    ),
                    unsafe_allow_html=True,
                )

        # Consensus Building
        status_text.text("🤝 Building consensus and applying RL optimization...")
        progress_bar.progress(90)
        time.sleep(1.5)

        # Get final consensus
        consensus = orchestrator.facilitate_discussion(project_description)

        # Display consensus
        progress_bar.progress(100)
        status_text.text("✅ Consensus reached!")

        st.markdown(
            """
        <div class="consensus-box">
            <h2>🎯 Multi-Agent Consensus Reached!</h2>
            <p><strong>Proposal ID:</strong> {}</p>
            <p><strong>Combined Confidence:</strong> {:.0%}</p>
            <p><strong>RL Optimization Score:</strong> {:.2f}</p>
            <p><strong>Optimization Applied:</strong> {}</p>
        </div>
        """.format(
                consensus["proposal_id"],
                consensus["confidence"],
                consensus.get("rl_score", 0.85),
                consensus.get("optimization", "add_examples"),
            ),
            unsafe_allow_html=True,
        )

        # Store for human approval
        st.session_state.current_proposal = consensus

        # Show detailed consensus
        with st.expander("📋 Detailed Consensus Analysis"):
            st.json(consensus)

    elif not project_description:
        st.info(
            "💡 Enter a project description in the sidebar to start the multi-agent discussion!"
        )

with tab2:
    st.header("👤 Human-in-the-Loop Approval")

    if st.session_state.current_proposal:
        proposal = st.session_state.current_proposal

        st.markdown(
            """
        <div class="human-gate">
            <h3>🤖 AI Consensus Proposal</h3>
            <p>Review the multi-agent consensus before implementation</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Display proposal details
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📝 Proposed Approach")
            st.text_area(
                "Synthesized Approach",
                proposal.get("approach", "No approach specified"),
                height=200,
                disabled=True,
            )

            st.subheader("🔍 Agent Analyses")
            agent_analyses = proposal.get("agent_analyses", {})
            for agent_name, analysis in agent_analyses.items():
                with st.expander(f"🤖 {agent_name.title()} Analysis"):
                    st.json(analysis)

        with col2:
            st.subheader("📊 Metrics")
            st.metric("Confidence", f"{proposal.get('confidence', 0):.0%}")
            st.metric("RL Score", f"{proposal.get('rl_score', 0):.2f}")
            st.metric("Patterns", len(proposal.get("patterns", [])))
            st.metric("Risks", len(proposal.get("risks", [])))

            st.subheader("🎯 Decision")

            # Approval buttons with feedback
            col_approve, col_reject, col_feedback = st.columns(3)

            with col_approve:
                if st.button("✅ Approve", type="primary", use_container_width=True):
                    st.success("🎉 Proposal approved! Moving to implementation...")
                    st.balloons()

                    # Log approval
                    st.session_state.human_gate._log_decision(
                        proposal,
                        {
                            "approved": True,
                            "timestamp": datetime.now().isoformat(),
                            "reason": "human_approved",
                        },
                    )

            with col_reject:
                if st.button("❌ Reject", use_container_width=True):
                    feedback = st.text_input("Reason for rejection:")
                    if feedback:
                        st.error("❌ Proposal rejected. Agents will revise.")

                        # Log rejection
                        st.session_state.human_gate._log_decision(
                            proposal,
                            {
                                "approved": False,
                                "timestamp": datetime.now().isoformat(),
                                "reason": "human_rejected",
                                "feedback": feedback,
                            },
                        )

            with col_feedback:
                if st.button("🔄 Request Changes", use_container_width=True):
                    st.warning("🔄 Changes requested. Please provide feedback below.")

            # Enhanced feedback section
            st.subheader("💬 Detailed Feedback")

            feedback_score = st.slider(
                "Rate this proposal:",
                -5,
                5,
                0,
                help="Negative = Poor, Positive = Excellent",
            )

            feedback_comment = st.text_area(
                "Additional comments (optional):",
                placeholder="Share your thoughts about this proposal...",
                help="Detailed feedback helps improve the system!",
            )

            if st.button("📤 Submit Feedback", use_container_width=True):
                # Store feedback in database
                if hasattr(st.session_state, "db"):
                    feedback_data = {
                        "approved": feedback_score >= 0,
                        "feedback_score": feedback_score,
                        "comment": feedback_comment,
                        "feedback_text": f"Score: {feedback_score}, Comment: {feedback_comment}",
                    }

                    # Store in database
                    episode_id = proposal.get(
                        "proposal_id", f"feedback_{datetime.now().timestamp()}"
                    )
                    st.session_state.db.store_human_feedback(
                        episode_id=episode_id,
                        approved=feedback_data["approved"],
                        feedback_text=feedback_data["feedback_text"],
                        feedback_score=feedback_data["feedback_score"],
                        comment=feedback_data["comment"],
                    )

                    st.success(f"📤 Feedback submitted! Score: {feedback_score}")

                    # Update reward calculation
                    if hasattr(st.session_state, "reward_agent"):
                        st.session_state.reward_agent.calculate_reward(
                            test_results={},
                            human_feedback=feedback_data,
                            iteration_count=1,
                        )

            # Edit option
            if st.button("✏️ Edit Proposal", use_container_width=True):
                new_approach = st.text_area(
                    "Edit the approach:", proposal.get("approach", "")
                )
                if st.button("💾 Save Changes", use_container_width=True):
                    proposal["approach"] = new_approach
                    proposal["edited_by_human"] = True
                    st.success("✅ Proposal edited and approved!")

                    # Log edit
                    st.session_state.human_gate._log_decision(
                        proposal,
                        {
                            "approved": True,
                            "timestamp": datetime.now().isoformat(),
                            "reason": "human_edited",
                            "new_approach": new_approach,
                        },
                    )
    else:
        st.info("🤖 No proposal available. Start a multi-agent discussion first!")

with tab3:
    st.header("📊 Learning Metrics")

    # RL Performance Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Avg Iterations</h3>
            <h2>2.3</h2>
            <p>↓ -0.7</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Success Rate</h3>
            <h2>94%</h2>
            <p>↑ +12%</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Avg Quality</h3>
            <h2>8.7/10</h2>
            <p>↑ +1.2</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Total Episodes</h3>
            <h2>147</h2>
            <p>↑ +23</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Learning Curves
    st.subheader("📈 Learning Progress")

    # Mock data for visualization
    episodes = list(range(1, 148))
    rewards = [
        40 + 30 * (1 - np.exp(-i / 50)) + np.random.normal(0, 5) for i in episodes
    ]
    iterations = [
        5 - 3 * (1 - np.exp(-i / 30)) + np.random.normal(0, 0.5) for i in episodes
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Reward Learning Curve")
        chart_data = pd.DataFrame({"Episode": episodes, "Reward": rewards})
        st.line_chart(chart_data.set_index("Episode"))

    with col2:
        st.subheader("🔄 Iteration Reduction")
        chart_data = pd.DataFrame({"Episode": episodes, "Iterations": iterations})
        st.line_chart(chart_data.set_index("Episode"))

    # Agent Performance
    st.subheader("🤖 Agent Performance Over Time")

    agent_data = pd.DataFrame(
        {
            "Agent": ["CodeGen", "Architect", "Reviewer"] * 10,
            "Episode": list(range(1, 31)),
            "Confidence": [0.7 + 0.2 * np.random.random() for _ in range(30)],
            "Success_Rate": [0.8 + 0.15 * np.random.random() for _ in range(30)],
        }
    )

    st.line_chart(
        agent_data.pivot(index="Episode", columns="Agent", values="Confidence")
    )

with tab4:
    st.header("💻 Generated Code")

    if st.session_state.current_proposal:
        st.subheader("🎯 Implementation Plan")

        # Mock generated code based on proposal
        mock_code = f'''
# Generated by CodeConductor v2.0
# Proposal ID: {st.session_state.current_proposal.get("proposal_id", "unknown")}
# Confidence: {st.session_state.current_proposal.get("confidence", 0):.0%}

import fastapi
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta

app = FastAPI(title="User Authentication API")
security = HTTPBearer()

class User(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

# Database simulation
users_db = {{}}

@app.post("/register")
async def register_user(user: User):
    """Register a new user"""
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # In production, hash the password
    users_db[user.username] = {{
        "email": user.email,
        "password": user.password,
        "created_at": datetime.now()
    }}
    
    return {{"message": "User registered successfully"}}

@app.post("/login")
async def login(login_req: LoginRequest):
    """Authenticate user and return JWT token"""
    if login_req.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = users_db[login_req.username]
    if user["password"] != login_req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate JWT token
    token = jwt.encode(
        {{"username": login_req.username, "exp": datetime.utcnow() + timedelta(hours=24)}},
        "secret_key",
        algorithm="HS256"
    )
    
    return {{"access_token": token, "token_type": "bearer"}}

@app.get("/users/me")
async def get_current_user(token: str = Depends(security)):
    """Get current user information"""
    try:
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        username = payload.get("username")
        if username not in users_db:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {{"username": username, "email": users_db[username]["email"]}}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

        st.code(mock_code, language="python")

        # Project structure visualization
        st.subheader("📁 Project Structure")

        project_dir = Path("data/generated/iter_0_project")
        if project_dir.exists():

            def get_file_tree(path: Path, prefix: str = "") -> str:
                tree = []
                for item in sorted(path.iterdir()):
                    if item.is_file():
                        tree.append(f"{prefix}📄 {item.name}")
                    elif item.is_dir():
                        tree.append(f"{prefix}📁 {item.name}/")
                        tree.append(get_file_tree(item, prefix + "  "))
                return "\n".join(tree)

            file_tree = get_file_tree(project_dir)
            st.code(file_tree, language="text")
        else:
            st.info("📁 No multi-file project structure available")

        # Code metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lines of Code", "67")
        with col2:
            st.metric("Complexity Score", "2.1")
        with col3:
            st.metric("Test Coverage", "96%")
        with col4:
            st.metric("Security Score", "A+")

        # Download buttons
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="⬇️ Download Single File",
                data=mock_code,
                file_name="generated_api.py",
                mime="text/python",
            )

        with col2:
            # Check if multi-file project exists
            project_dir = Path("data/generated/iter_0_project")
            if project_dir.exists():
                import zipfile
                import io

                # Create ZIP file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for file_path in project_dir.rglob("*"):
                        if file_path.is_file():
                            zip_file.write(
                                file_path, file_path.relative_to(project_dir)
                            )

                zip_buffer.seek(0)
                st.download_button(
                    label="📦 Download Full Project (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="generated_project.zip",
                    mime="application/zip",
                )
            else:
                st.info("📦 No multi-file project available")

        # Implementation status
        st.success("✅ Code generated successfully! Ready for deployment.")

    else:
        st.info(
            "💻 No code generated yet. Complete the multi-agent discussion and human approval first!"
        )

with tab5:
    st.header("📈 Project History")

    # Get approval stats
    stats = st.session_state.human_gate.get_approval_stats()

    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Decisions", stats["total_decisions"])
    with col2:
        st.metric("Approved", stats["approved"])
    with col3:
        st.metric("Rejected", stats["rejected"])
    with col4:
        st.metric("Approval Rate", f"{stats['approval_rate']:.0%}")

    # History table
    st.subheader("📋 Recent Decisions")

    if st.session_state.human_gate.approval_history:
        history_data = []
        for entry in st.session_state.human_gate.approval_history[-10:]:  # Last 10
            history_data.append(
                {
                    "Timestamp": entry["decision"]["timestamp"],
                    "Proposal ID": entry["proposal"].get("proposal_id", "unknown"),
                    "Approved": "✅" if entry["decision"]["approved"] else "❌",
                    "Reason": entry["decision"]["reason"],
                    "Confidence": f"{entry['proposal'].get('confidence', 0):.0%}",
                }
            )

        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("📊 No decision history yet. Start making decisions!")

    # Learning trends
    st.subheader("📈 Learning Trends")

    # Mock learning data
    learning_data = pd.DataFrame(
        {
            "Date": pd.date_range(start="2024-01-01", periods=30, freq="D"),
            "Success_Rate": [0.7 + 0.2 * np.random.random() for _ in range(30)],
            "Avg_Confidence": [0.6 + 0.3 * np.random.random() for _ in range(30)],
            "Iterations_Needed": [4 - 2 * np.random.random() for _ in range(30)],
        }
    )

    st.line_chart(learning_data.set_index("Date"))

with tab6:
    st.header("💬 Human Feedback Analytics")

    # Initialize database if not exists
    if not hasattr(st.session_state, "db"):
        from storage.rl_database import RLDatabase

        st.session_state.db = RLDatabase()

    # Get feedback statistics
    try:
        feedback_stats = st.session_state.db.get_feedback_statistics()

        # Display feedback metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Feedback", feedback_stats["total_feedback"])
        with col2:
            st.metric("👍 Positive", feedback_stats["positive_feedback"])
        with col3:
            st.metric("👎 Negative", feedback_stats["negative_feedback"])
        with col4:
            st.metric("Approval Rate", f"{feedback_stats['approval_rate']:.0%}")

        # Feedback distribution
        st.subheader("📊 Feedback Distribution")

        if feedback_stats["total_feedback"] > 0:
            feedback_data = pd.DataFrame(
                {
                    "Type": ["Positive", "Negative"],
                    "Count": [
                        feedback_stats["positive_feedback"],
                        feedback_stats["negative_feedback"],
                    ],
                }
            )

            st.bar_chart(feedback_data.set_index("Type"))

            # Average comment length
            st.metric(
                "Avg Comment Length",
                f"{feedback_stats['avg_comment_length']:.1f} chars",
            )

            # Recent feedback
            st.subheader("📝 Recent Feedback")

            if feedback_stats["recent_feedback"]:
                recent_df = pd.DataFrame(feedback_stats["recent_feedback"])
                recent_df["Status"] = recent_df["approved"].map(
                    {True: "👍", False: "👎"}
                )
                recent_df["Date"] = pd.to_datetime(recent_df["timestamp"]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                )

                st.dataframe(
                    recent_df[["Date", "Status", "feedback_text"]].head(10),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("📝 No feedback submitted yet. Try rating a proposal!")
        else:
            st.info("📊 No feedback data yet. Submit your first feedback!")

    except Exception as e:
        st.error(f"Error loading feedback data: {e}")
        st.info("📊 Feedback system ready - submit your first feedback!")

with tab7:
    st.header("⚙️ Policy Configuration")

    try:
        from integrations.policy_loader import PolicyLoader
        from pathlib import Path

        # Initialize policy loader
        policy_loader = PolicyLoader()

        # Display current policy summary
        st.subheader("📊 Current Policy Summary")
        summary = policy_loader.get_policy_summary()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Enforcement Mode", summary["enforcement_mode"].title())
        with col2:
            st.metric("Blocked Imports", summary["blocked_imports_count"])
        with col3:
            st.metric("Blocked Patterns", summary["blocked_patterns_count"])
        with col4:
            st.metric("Forbidden Functions", summary["forbidden_functions_count"])

        # Policy editor
        st.subheader("📝 Edit Policies")

        # Load current YAML content
        policy_path = Path("config/policies.yaml")
        if policy_path.exists():
            current_yaml = policy_path.read_text(encoding="utf-8")

            # YAML editor
            edited_yaml = st.text_area(
                "Policy Configuration (YAML)",
                value=current_yaml,
                height=400,
                help="Edit the policy configuration. Changes will be saved when you click 'Save Policies'.",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Policies", type="primary"):
                    try:
                        policy_loader.save_policies(edited_yaml)
                        st.success("✅ Policies saved successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to save policies: {e}")

            with col2:
                if st.button("🔄 Reload Policies"):
                    try:
                        policy_loader.reload()
                        st.success("✅ Policies reloaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to reload policies: {e}")

            # Policy testing
            st.subheader("🧪 Test Policies")

            test_code = st.text_area(
                "Test Code",
                value="""import os
import subprocess

def test_function():
    os.system("rm -rf /")
    print("Hello World")
    x = 123
    return x""",
                height=150,
                help="Enter code to test against current policies",
            )

            if st.button("🔍 Analyze Code"):
                try:
                    result = policy_loader.analyze_code(test_code)

                    # Display results
                    st.subheader("📊 Analysis Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Total Violations", result["summary"]["total_violations"]
                        )
                    with col2:
                        st.metric("Critical", result["summary"]["critical"])
                    with col3:
                        st.metric("High", result["summary"]["high"])
                    with col4:
                        st.metric(
                            "Should Block", "Yes" if result["should_block"] else "No"
                        )

                    # Show violations
                    if result["violations"]:
                        st.subheader("🚨 Policy Violations")
                        for violation in result["violations"]:
                            with st.expander(
                                f"Line {violation.line_number}: {violation.description}"
                            ):
                                st.code(violation.code_snippet, language="python")
                                st.info(f"💡 Suggestion: {violation.suggestion}")

                    # Show recommendations
                    if result["recommendations"]:
                        st.subheader("💡 Recommendations")
                        for rec in result["recommendations"]:
                            st.info(rec)

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

        else:
            st.error("❌ Policy file not found!")

    except Exception as e:
        st.error(f"❌ Failed to load policy system: {e}")
        st.info("Make sure config/policies.yaml exists and is valid YAML.")

with tab8:
    st.header("🔌 Plugin Management")

    try:
        from plugins.base import PluginManager, PluginType
        from agents.orchestrator import AgentOrchestrator

        # Initialize plugin manager
        plugin_manager = PluginManager()

        # Display plugin summary
        st.subheader("📊 Plugin Summary")

        discovered_plugins = plugin_manager.discover_plugins()
        active_plugins = plugin_manager.list_plugins()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Discovered Plugins", len(discovered_plugins))
        with col2:
            st.metric("Active Plugins", len(active_plugins))
        with col3:
            st.metric(
                "Agent Plugins",
                len(
                    [
                        p
                        for p in discovered_plugins
                        if p.metadata.plugin_type == PluginType.AGENT
                    ]
                ),
            )
        with col4:
            st.metric(
                "Tool Plugins",
                len(
                    [
                        p
                        for p in discovered_plugins
                        if p.metadata.plugin_type == PluginType.TOOL
                    ]
                ),
            )

        # Plugin discovery
        st.subheader("🔍 Plugin Discovery")

        if discovered_plugins:
            plugin_data = []
            for plugin in discovered_plugins:
                plugin_data.append(
                    {
                        "Name": plugin.metadata.name,
                        "Type": plugin.metadata.plugin_type.value,
                        "Version": plugin.metadata.version,
                        "Author": plugin.metadata.author,
                        "Status": plugin.status.value,
                        "Enabled": "✅" if plugin.is_enabled else "❌",
                    }
                )

            df = pd.DataFrame(plugin_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "🔍 No plugins discovered. Check plugin directories and configuration."
            )

        # Plugin management
        st.subheader("⚙️ Plugin Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Plugins", type="primary"):
                try:
                    discovered_plugins = plugin_manager.discover_plugins()
                    st.success(f"✅ Refreshed! Found {len(discovered_plugins)} plugins")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to refresh plugins: {e}")

        with col2:
            if st.button("💾 Save Plugin Config"):
                try:
                    plugin_manager.save_plugin_config()
                    st.success("✅ Plugin configuration saved!")
                except Exception as e:
                    st.error(f"❌ Failed to save plugin config: {e}")

        # Plugin details
        if discovered_plugins:
            st.subheader("📋 Plugin Details")

            selected_plugin = st.selectbox(
                "Select Plugin",
                options=[p.metadata.name for p in discovered_plugins],
                format_func=lambda x: f"{x} ({next(p.metadata.plugin_type.value for p in discovered_plugins if p.metadata.name == x)})",
            )

            if selected_plugin:
                plugin = next(
                    p for p in discovered_plugins if p.metadata.name == selected_plugin
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Name:** {plugin.metadata.name}")
                    st.markdown(f"**Version:** {plugin.metadata.version}")
                    st.markdown(f"**Author:** {plugin.metadata.author}")
                    st.markdown(f"**Type:** {plugin.metadata.plugin_type.value}")
                    st.markdown(f"**Status:** {plugin.status.value}")

                    if plugin.metadata.homepage:
                        st.markdown(
                            f"**Homepage:** [{plugin.metadata.homepage}]({plugin.metadata.homepage})"
                        )

                    if plugin.metadata.license:
                        st.markdown(f"**License:** {plugin.metadata.license}")

                with col2:
                    st.markdown(f"**Description:** {plugin.metadata.description}")

                    if plugin.metadata.tags:
                        st.markdown(
                            "**Tags:** "
                            + ", ".join([f"`{tag}`" for tag in plugin.metadata.tags])
                        )

                    if plugin.metadata.dependencies:
                        st.markdown(
                            "**Dependencies:** "
                            + ", ".join(
                                [f"`{dep}`" for dep in plugin.metadata.dependencies]
                            )
                        )

                # Plugin configuration
                if plugin.metadata.config_schema:
                    st.subheader("⚙️ Plugin Configuration")

                    config_editor = {}
                    for key, schema in plugin.metadata.config_schema.items():
                        if schema.get("type") == "boolean":
                            config_editor[key] = st.checkbox(
                                key.replace("_", " ").title(),
                                value=plugin.config.get(
                                    key, schema.get("default", False)
                                ),
                                help=schema.get("description", ""),
                            )
                        elif schema.get("type") == "integer":
                            config_editor[key] = st.number_input(
                                key.replace("_", " ").title(),
                                value=plugin.config.get(key, schema.get("default", 0)),
                                help=schema.get("description", ""),
                            )
                        else:
                            config_editor[key] = st.text_input(
                                key.replace("_", " ").title(),
                                value=plugin.config.get(key, schema.get("default", "")),
                                help=schema.get("description", ""),
                            )

                    if st.button("💾 Update Plugin Config"):
                        plugin.config.update(config_editor)
                        st.success("✅ Plugin configuration updated!")

                # Plugin actions
                st.subheader("🎯 Plugin Actions")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if plugin.status.value == "inactive":
                        if st.button("▶️ Load Plugin"):
                            try:
                                plugin_instance = plugin_manager.load_plugin(plugin)
                                if plugin_instance:
                                    st.success(
                                        f"✅ Loaded plugin: {plugin.metadata.name}"
                                    )
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to load plugin")
                            except Exception as e:
                                st.error(f"❌ Failed to load plugin: {e}")
                    else:
                        if st.button("⏹️ Unload Plugin"):
                            try:
                                if plugin_manager.unload_plugin(plugin.metadata.name):
                                    st.success(
                                        f"✅ Unloaded plugin: {plugin.metadata.name}"
                                    )
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to unload plugin")
                            except Exception as e:
                                st.error(f"❌ Failed to unload plugin: {e}")

                with col2:
                    if plugin.is_enabled:
                        if st.button("🚫 Disable Plugin"):
                            plugin.is_enabled = False
                            st.success(f"✅ Disabled plugin: {plugin.metadata.name}")
                            st.rerun()
                    else:
                        if st.button("✅ Enable Plugin"):
                            plugin.is_enabled = True
                            st.success(f"✅ Enabled plugin: {plugin.metadata.name}")
                            st.rerun()

                with col3:
                    if plugin.error_message:
                        st.error(f"❌ Error: {plugin.error_message}")

        # Test plugin system
        st.subheader("🧪 Test Plugin System")

        if st.button("🚀 Test Agent Orchestrator with Plugins"):
            try:
                orchestrator = AgentOrchestrator(enable_plugins=True)
                plugin_info = orchestrator.get_plugin_info()

                if plugin_info["plugins_enabled"]:
                    st.success("✅ Plugin system is working!")
                    st.json(plugin_info)
                else:
                    st.warning("⚠️ Plugin system is disabled")

            except Exception as e:
                st.error(f"❌ Plugin system test failed: {e}")

    except Exception as e:
        st.error(f"❌ Failed to load plugin system: {e}")
        st.info("Make sure plugins are properly configured and installed.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <h3>🎼 CodeConductor v2.0</h3>
        <p>Multi-Agent AI Development Orchestrator</p>
        <p><em>Built with ❤️ using Reinforcement Learning</em></p>
        <p>🤖 3 AI Agents | 👤 Human Control | 🧠 RL Optimization</p>
    </div>
    """,
    unsafe_allow_html=True,
)
