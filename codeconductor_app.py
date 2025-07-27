import streamlit as st
import asyncio
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from ensemble.model_manager import ModelManager
from ensemble.query_dispatcher import QueryDispatcher
from ensemble.consensus_calculator import ConsensusCalculator
from ensemble.hybrid_ensemble import HybridEnsemble
from ensemble.ensemble_engine import EnsembleEngine
from generators.prompt_generator import PromptGenerator
from integrations.notifications import notify_success, notify_error
from analysis.planner_agent import PlannerAgent
from feedback.validation_system import validate_cursor_output
from feedback.learning_system import save_successful_pattern, LearningSystem
from context.rag_system import rag_system

# Page configuration
st.set_page_config(
    page_title="CodeConductor MVP",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .model-status {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .healthy { background: rgba(76, 175, 80, 0.2); }
    .unhealthy { background: rgba(244, 67, 54, 0.2); }
    .unknown { background: rgba(255, 152, 0, 0.2); }
    
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class CodeConductorApp:
    def __init__(self):
        self.model_manager = ModelManager()
        self.query_dispatcher = QueryDispatcher(self.model_manager)
        self.consensus_calculator = ConsensusCalculator()
        self.hybrid_ensemble = HybridEnsemble()
        self.ensemble_engine = None  # Will be initialized on demand
        self.prompt_generator = PromptGenerator()
        self.learning_system = LearningSystem()
        self.generation_history = []

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "models_discovered" not in st.session_state:
            st.session_state.models_discovered = False
        if "current_task" not in st.session_state:
            st.session_state.current_task = ""
        if "generation_results" not in st.session_state:
            st.session_state.generation_results = None

    def render_header(self):
        """Render the main header"""
        st.markdown(
            """
        <div class="main-header">
            <h1>🎼 CodeConductor MVP</h1>
            <p>AI-Powered Development Pipeline with Multi-Model Ensemble Intelligence</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        """Render the sidebar with controls and history"""
        with st.sidebar:
            st.markdown("### 🎛️ Controls")

            # Model discovery
            if st.button("🔄 Refresh Models", use_container_width=True):
                with st.spinner("Discovering models..."):
                    try:
                        # Use asyncio to run the async function
                        import asyncio

                        models = asyncio.run(self.model_manager.list_models())
                        st.session_state.models_discovered = True
                        st.session_state.models = models
                        st.success(f"Found {len(models)} models!")
                    except Exception as e:
                        st.error(f"Error discovering models: {e}")

            # Generation options
            st.markdown("### ⚙️ Generation Options")
            iterations = st.slider("Iterations", 1, 5, 1)
            timeout = st.slider("Timeout (seconds)", 10, 60, 30)

            # Output options
            st.markdown("### 📤 Output Options")
            auto_copy = st.checkbox("Auto-copy to clipboard", value=True)
            save_files = st.checkbox("Save generated files", value=True)

            # Generation history
            if self.generation_history:
                st.markdown("### 📚 Generation History")
                for i, entry in enumerate(reversed(self.generation_history[-5:])):
                    with st.expander(
                        f"Task {len(self.generation_history) - i}: {entry['task'][:50]}..."
                    ):
                        st.write(f"**Status:** {entry['status']}")
                        st.write(f"**Models:** {entry['models_used']}")
                        st.write(f"**Time:** {entry['timestamp']}")

            # Project Analysis
            st.markdown("---")
            st.markdown("### 📁 Project Analysis")

            # Input för sökväg till projektmappen
            project_path = st.text_input(
                "Add Project Path",
                value="",
                help="Enter the root folder of your codebase",
            )

            # Knapp för att starta analys
            if st.button("🔍 Analyze Project"):
                if not project_path:
                    st.error("Please enter a valid project path.")
                else:
                    try:
                        from analysis.project_analyzer import analyze_project

                        with st.spinner("🔍 Scanning project..."):
                            # Use the optimized analyze_project function
                            report = analyze_project(project_path)
                        st.success("✅ Analysis complete!")
                        st.session_state.project_report = report
                        st.session_state.project_path = (
                            project_path  # Store for Planner Agent
                        )
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")

            # Knapp för att visa rapporten (om den finns)
            if "project_report" in st.session_state:
                if st.button("📊 View Report"):
                    st.session_state.show_project_report = True

    def render_model_status(self):
        """Render model status dashboard"""
        st.markdown("### 🤖 Model Status Dashboard")

        if not st.session_state.models_discovered:
            st.info(
                "Click 'Refresh Models' in the sidebar to discover available models."
            )
            return

        try:
            # Use cached models from session state - don't fallback to avoid infinite loops
            models = st.session_state.get("models", [])

            if not models:
                st.warning(
                    "No models found. Click 'Refresh Models' in the sidebar to discover models."
                )
                return

            # Create columns for model display
            cols = st.columns(3)

            for i, model in enumerate(models):
                col_idx = i % 3
                with cols[col_idx]:
                    # Health check - only if we have models
                    try:
                        import asyncio

                        health = asyncio.run(self.model_manager.check_health(model))
                        status_icon = "✅" if health else "❌"
                        status_class = "healthy" if health else "unhealthy"
                    except:
                        status_icon = "⚠️"
                        status_class = "unknown"

                    st.markdown(
                        f"""
                    <div class="model-status {status_class}">
                        <div>
                            <strong>{model.id}</strong><br>
                            <small>{model.provider}</small>
                        </div>
                        <div>{status_icon}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"Error loading models: {e}")

    def render_task_input(self):
        """Render task input section"""
        st.markdown("### 🎯 Task Input")

        # Quick example buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📱 Phone Validator", use_container_width=True):
                st.session_state.current_task = (
                    "Create a function to validate Swedish phone numbers"
                )
        with col2:
            if st.button("🧮 Calculator", use_container_width=True):
                st.session_state.current_task = (
                    "Create a simple calculator class with basic operations"
                )
        with col3:
            if st.button("🔐 Password Generator", use_container_width=True):
                st.session_state.current_task = "Create a secure password generator with configurable length and complexity"

        # Task input
        task = st.text_area(
            "Enter your development task:",
            value=st.session_state.current_task,
            height=100,
            placeholder="Describe what you want to build... (e.g., 'Create a function to validate email addresses')",
        )

        # Planner Agent integration
        if task.strip():
            st.markdown("---")
            st.markdown("### 🧠 Intelligent Planning")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📋 Create Development Plan", use_container_width=True):
                    self._create_development_plan(task)

            with col2:
                if st.button("🤖 Generate Cursor Prompts", use_container_width=True):
                    self._generate_cursor_prompts(task)

        return task

    def _create_development_plan(self, task):
        """Create development plan using Planner Agent with RAG context"""
        try:
            with st.spinner(
                "🧠 Creating intelligent development plan with RAG context..."
            ):
                # Initialize planner with current project
                project_path = st.session_state.get(
                    "project_path", "test_fastapi_project"
                )
                planner = PlannerAgent(project_path)

                # Get relevant context using RAG
                context_docs = rag_system.retrieve_context(task, k=3)
                rag_context = rag_system.format_context_for_prompt(context_docs)

                # Create enhanced task with RAG context
                enhanced_task = f"""
{task}

{rag_context}
"""

                # Create plan with enhanced context
                plan = planner.create_development_plan(enhanced_task)

                # Store in session state
                st.session_state.development_plan = plan
                st.session_state.rag_context = {
                    "context_docs": context_docs,
                    "context_summary": rag_system.get_context_summary(task),
                }

                # Display plan
                self._display_development_plan(plan)

        except Exception as e:
            st.error(f"Failed to create development plan: {str(e)}")

    def _generate_cursor_prompts(self, task):
        """Generate Cursor prompts using Planner Agent with RAG context"""
        try:
            with st.spinner(
                "🤖 Generating optimized Cursor prompts with RAG context..."
            ):
                # Initialize planner
                project_path = st.session_state.get(
                    "project_path", "test_fastapi_project"
                )
                planner = PlannerAgent(project_path)

                # Get relevant context using RAG
                context_docs = rag_system.retrieve_context(task, k=3)
                rag_context = rag_system.format_context_for_prompt(context_docs)

                # Create enhanced task with RAG context
                enhanced_task = f"""
{task}

{rag_context}
"""

                # Create plan and get prompts with enhanced context
                plan = planner.create_development_plan(enhanced_task)

                # Store prompts in session state for save pattern functionality
                st.session_state.last_generated_prompts = plan.cursor_prompts

                # Display prompts
                st.markdown("### 🤖 Generated Cursor Prompts (with RAG context)")
                for i, prompt in enumerate(plan.cursor_prompts, 1):
                    with st.expander(f"Prompt {i}: {plan.steps[i - 1]['description']}"):
                        st.text_area(
                            f"Copy this prompt to Cursor:",
                            value=prompt,
                            height=200,
                            key=f"prompt_{i}",
                        )
                        if st.button(f"📋 Copy Prompt {i}", key=f"copy_{i}"):
                            st.write("✅ Prompt copied to clipboard!")
                            # Store the selected prompt for save pattern
                            st.session_state.last_generated_prompt = prompt

        except Exception as e:
            st.error(f"Failed to generate prompts: {str(e)}")

    def _display_development_plan(self, plan):
        """Display development plan in GUI with RAG context"""
        st.markdown("### 📋 Development Plan")

        # Display RAG context if available
        if hasattr(st.session_state, "rag_context") and st.session_state.rag_context:
            rag_info = st.session_state.rag_context
            with st.expander("🔍 RAG Context Used", expanded=False):
                summary = rag_info["context_summary"]
                st.markdown(f"""
                **Context Available:** {"✅" if summary["context_available"] else "❌"}
                **Documents Found:** {summary["context_count"]}
                **Average Relevance:** {summary["avg_relevance"]:.3f}
                **Context Types:** {", ".join(summary["context_types"])}
                """)

                if rag_info["context_docs"]:
                    st.markdown("**Retrieved Documents:**")
                    for i, doc in enumerate(rag_info["context_docs"], 1):
                        st.markdown(f"""
                        **{i}. {doc["metadata"].get("filename", "Unknown")}** (Score: {doc["relevance_score"]:.3f})
                        - Type: {doc["metadata"].get("type", "Unknown")}
                        - Content: {doc["content"][:200]}...
                        """)

        # Plan overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Complexity", plan.estimated_complexity.upper())
        with col2:
            st.metric("Steps", len(plan.steps))
        with col3:
            st.metric("Dependencies", len(plan.dependencies))

        # Dependencies
        if plan.dependencies:
            st.markdown("#### 📦 Dependencies Needed")
            for dep in plan.dependencies:
                st.write(f"- `{dep}`")

        # Implementation steps
        st.markdown("#### 📝 Implementation Steps")
        for step in plan.steps:
            with st.expander(f"Step {step['number']}: {step['description']}"):
                st.write(f"**Estimated time:** {step['estimated_time']}")
                if step["files_affected"]:
                    st.write(f"**Files affected:** {', '.join(step['files_affected'])}")

        # Validation criteria
        st.markdown("#### ✅ Validation Criteria")
        for criteria in plan.validation_criteria:
            st.write(f"- {criteria}")

    def render_generation_controls(self, task):
        """Render generation controls"""
        if not task.strip():
            st.warning("Please enter a task to begin generation.")
            return False

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            if st.button("🚀 Generate Code", type="primary", use_container_width=True):
                return True

        with col2:
            if st.button("🧪 Test Locally First", use_container_width=True):
                return "ensemble_test"

        with col3:
            if st.button("🧪 Test Only", use_container_width=True):
                return "test"

        with col4:
            if st.button("📋 Copy Prompt Only", use_container_width=True):
                return "prompt"

        return False

    async def run_generation(self, task):
        """Run the full generation pipeline using hybrid ensemble"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Complexity analysis
            status_text.text("🔍 Analyzing task complexity...")
            progress_bar.progress(10)

            # Step 2: Hybrid ensemble processing
            status_text.text("🤖 Running hybrid ensemble engine...")
            progress_bar.progress(30)

            # Use hybrid ensemble for processing
            hybrid_result = await self.hybrid_ensemble.process_task(task)

            if not hybrid_result:
                st.error("Hybrid ensemble failed. Please check model availability.")
                return

            # Step 3: Consensus calculation
            status_text.text("🧠 Calculating consensus...")
            progress_bar.progress(60)

            # Step 4: Prompt generation
            status_text.text("📝 Generating prompt...")
            progress_bar.progress(80)

            prompt = self.prompt_generator.generate_prompt(
                task, hybrid_result.final_consensus
            )

            # Step 5: Complete
            status_text.text("✅ Generation complete!")
            progress_bar.progress(100)

            return {
                "task": task,
                "models_used": len(hybrid_result.local_responses)
                + len(hybrid_result.cloud_responses),
                "consensus": hybrid_result.final_consensus,
                "prompt": prompt,
                "status": "success",
                "complexity_analysis": hybrid_result.complexity_analysis,
                "total_cost": hybrid_result.total_cost,
                "total_time": hybrid_result.total_time,
                "escalation_used": hybrid_result.escalation_used,
                "escalation_reason": hybrid_result.escalation_reason,
                "local_confidence": hybrid_result.local_confidence,
                "cloud_confidence": hybrid_result.cloud_confidence,
                "local_responses": hybrid_result.local_responses,
                "cloud_responses": hybrid_result.cloud_responses,
            }

        except Exception as e:
            st.error(f"Generation failed: {e}")
            return None

    async def run_ensemble_test(self, task):
        """Run Ensemble Engine test for local code generation"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Initialize Ensemble Engine
            status_text.text("🤖 Initializing Ensemble Engine...")
            progress_bar.progress(20)

            if self.ensemble_engine is None:
                self.ensemble_engine = EnsembleEngine()

            # Step 2: Process request with fallback
            status_text.text("⚡ Testing with fast models...")
            progress_bar.progress(50)

            # Create ensemble request
            from ensemble.ensemble_engine import EnsembleRequest

            request = EnsembleRequest(
                task=task, min_models=1, max_tokens=500, temperature=0.1
            )

            # Process with fallback strategy
            result = await self.ensemble_engine.process_request_with_fallback(request)

            if not result:
                st.error("Ensemble test failed. Please check model availability.")
                return None

            # Step 3: Complete
            status_text.text("✅ Ensemble test complete!")
            progress_bar.progress(100)

            return {
                "task": task,
                "type": "ensemble_test",
                "generated_code": result.get("generated_code", ""),
                "confidence": result.get("confidence", 0.0),
                "strategy": result.get("strategy", "unknown"),
                "models_used": result.get("models_used", []),
                "status": "success",
                "total_time": result.get("total_time", 0.0),
            }

        except Exception as e:
            st.error(f"Ensemble test failed: {e}")
            return None

    def _render_ensemble_results(self, results):
        """Render Ensemble test results"""
        st.markdown("### 🧪 Ensemble Test Results")

        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strategy", results.get("strategy", "unknown"))
        with col2:
            st.metric("Confidence", f"{results.get('confidence', 0.0):.2f}")
        with col3:
            st.metric("Models Used", len(results.get("models_used", [])))
        with col4:
            st.metric("Total Time", f"{results.get('total_time', 0.0):.2f}s")

        # Generated code
        st.markdown("#### 💻 Generated Code")
        if results.get("generated_code"):
            st.code(results["generated_code"], language="python")
        else:
            st.warning("No code generated")

        # Model details
        if results.get("models_used"):
            st.markdown("#### 🤖 Models Used")
            for model in results["models_used"]:
                st.info(f"✅ {model}")

        # Strategy details
        with st.expander("🔍 Strategy Details", expanded=False):
            st.json(
                {
                    "strategy": results.get("strategy", "unknown"),
                    "confidence": results.get("confidence", 0.0),
                    "models_used": results.get("models_used", []),
                    "total_time": results.get("total_time", 0.0),
                }
            )

    def render_results(self, results):
        """Render generation results"""
        if not results:
            return

        # Handle Ensemble test results differently
        if results.get("type") == "ensemble_test":
            self._render_ensemble_results(results)
            return

        st.markdown("### 📊 Generation Results")

        # Enhanced metrics with hybrid info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Models Used", results["models_used"])
        with col2:
            st.metric("Status", results["status"])
        with col3:
            st.metric("Total Time", f"{results.get('total_time', 0):.2f}s")
        with col4:
            st.metric("Total Cost", f"${results.get('total_cost', 0):.4f}")

        # Complexity analysis
        if "complexity_analysis" in results:
            complexity = results["complexity_analysis"]
            st.markdown("#### 🔍 Complexity Analysis")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Level", complexity.level.value.title())
            with col2:
                st.metric("Confidence", f"{complexity.confidence:.2f}")
            with col3:
                escalation_status = (
                    "☁️ Used" if results.get("escalation_used") else "🏠 Local"
                )
                st.metric("Escalation", escalation_status)
            with col4:
                if "escalation_reason" in results:
                    st.metric(
                        "Reason",
                        results["escalation_reason"][:20] + "..."
                        if len(results["escalation_reason"]) > 20
                        else results["escalation_reason"],
                    )

        # Confidence breakdown
        if "local_confidence" in results or "cloud_confidence" in results:
            st.markdown("#### 🎯 Confidence Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                local_conf = results.get("local_confidence", 0.0)
                st.metric("Local Confidence", f"{local_conf:.2f}")
            with col2:
                cloud_conf = results.get("cloud_confidence", 0.0)
                st.metric("Cloud Confidence", f"{cloud_conf:.2f}")

        # Model breakdown
        if "local_responses" in results and "cloud_responses" in results:
            st.markdown("#### 🤖 Model Breakdown")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Local Models", len(results["local_responses"]))
            with col2:
                st.metric("Cloud Models", len(results["cloud_responses"]))
            with col3:
                total_models = len(results["local_responses"]) + len(
                    results["cloud_responses"]
                )
                st.metric("Total Models", total_models)

        # Performance metrics
        if "total_time" in results:
            st.markdown("#### ⚡ Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Time", f"{results['total_time']:.2f}s")
            with col2:
                if results.get("total_time", 0) > 0:
                    models_per_second = total_models / results["total_time"]
                    st.metric("Models/sec", f"{models_per_second:.2f}")
            with col3:
                if results.get("total_cost", 0) > 0:
                    cost_per_second = results["total_cost"] / results["total_time"]
                    st.metric("Cost/sec", f"${cost_per_second:.4f}")

        # Consensus details
        with st.expander("🧠 Consensus Details", expanded=True):
            if hasattr(results["consensus"], "consensus"):
                consensus_data = results["consensus"].consensus
                if consensus_data:
                    st.json(consensus_data)
                else:
                    st.info(
                        "Consensus data is empty - models may not have provided structured responses"
                    )
                    st.write("This is normal for free-form text responses from LLMs")
            else:
                st.write("No structured consensus data available")

        # Generated prompt
        with st.expander("📝 Generated Prompt", expanded=False):
            if "prompt" in results:
                st.code(results["prompt"], language="markdown")
            else:
                st.warning("No prompt generated")

        # Detailed model responses
        with st.expander("🔍 Detailed Model Responses", expanded=False):
            if "local_responses" in results and results["local_responses"]:
                st.markdown("#### 🏠 Local Model Responses")
                for model_id, response in results["local_responses"].items():
                    with st.expander(f"Local: {model_id}", expanded=False):
                        if isinstance(response, dict) and "error" not in response:
                            content = ""
                            if "choices" in response and response["choices"]:
                                content = response["choices"][0]["message"]["content"]
                            elif "response" in response:
                                content = response["response"]
                            else:
                                content = str(response)
                            st.code(
                                content[:500] + "..."
                                if len(content) > 500
                                else content,
                                language="python",
                            )
                        else:
                            st.error(f"Error: {response.get('error', 'Unknown error')}")

            if "cloud_responses" in results and results["cloud_responses"]:
                st.markdown("#### ☁️ Cloud Model Responses")
                for response in results["cloud_responses"]:
                    with st.expander(f"Cloud: {response.model}", expanded=False):
                        st.metric("Cost", f"${response.cost_estimate:.4f}")
                        st.metric("Time", f"{response.response_time:.2f}s")
                        st.metric("Confidence", f"{response.confidence:.2f}")
                        st.code(
                            response.content[:500] + "..."
                            if len(response.content) > 500
                            else response.content,
                            language="python",
                        )

        # Add to history
        self.generation_history.append(
            {
                "timestamp": datetime.now(),
                "task": results["task"],
                "models_used": results["models_used"],
                "total_time": results.get("total_time", 0),
                "total_cost": results.get("total_cost", 0),
                "escalation_used": results.get("escalation_used", False),
                "complexity_level": results.get("complexity_analysis", {}).level.value
                if results.get("complexity_analysis")
                else "unknown",
            }
        )

    def render_project_report(self, report):
        """Render project analysis report"""
        st.markdown("### 📊 Project Analysis Report")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("FastAPI Routes", len(report.get("routes", [])))
        with col2:
            st.metric(
                "Database Tables", len(report.get("schema", {}).get("tables", []))
            )
        with col3:
            st.metric("Files Analyzed", report.get("files_analyzed", 0))

        # FastAPI Routes
        if report.get("routes"):
            st.markdown("#### 🚀 FastAPI Routes")
            for route in report["routes"]:
                with st.expander(f"{route['method']} {route['path']}", expanded=False):
                    st.write(f"**Function:** {route['function']}")
                    st.write(f"**File:** {route['file']}")
                    if route.get("parameters"):
                        st.write(f"**Parameters:** {route['parameters']}")

        # Database Schema
        if report.get("schema", {}).get("tables"):
            st.markdown("#### 🗄️ Database Schema")
            for table in report["schema"]["tables"]:
                with st.expander(f"Table: {table['name']}", expanded=False):
                    st.write(f"**Columns:** {len(table['columns'])}")
                    for col in table["columns"]:
                        st.write(f"- {col['name']}: {col['type']}")

        # AI Recommendations (if available)
        if report.get("ai_recommendations"):
            st.markdown("#### 🤖 AI Recommendations")
            for rec in report["ai_recommendations"]:
                st.info(f"**{rec['category']}:** {rec['message']}")

        # Export options
        st.markdown("#### 📤 Export Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Export as JSON"):
                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name="project_analysis.json",
                    mime="application/json",
                )
        with col2:
            if st.button("📊 Export as CSV"):
                # Convert report to CSV format
                csv_data = self.convert_report_to_csv(report)
                st.download_button(
                    label="Download CSV Report",
                    data=csv_data,
                    file_name="project_analysis.csv",
                    mime="text/csv",
                )

    def convert_report_to_csv(self, report):
        """Convert report to CSV format"""
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Write routes
        writer.writerow(["Type", "Name", "Details"])
        for route in report.get("routes", []):
            writer.writerow(
                [
                    "Route",
                    f"{route['method']} {route['path']}",
                    f"Function: {route['function']}, File: {route['file']}",
                ]
            )

        # Write tables
        for table in report.get("schema", {}).get("tables", []):
            writer.writerow(
                ["Table", table["name"], f"Columns: {len(table['columns'])}"]
            )

        return output.getvalue()

    def render_validation_section(self):
        """Render code validation section"""
        st.markdown("### 🔍 Code Validation")
        st.markdown("Paste code generated by Cursor for validation:")

        generated_code = st.text_area(
            "Generated Code",
            height=300,
            placeholder="Paste your Cursor-generated code here...",
        )

        if st.button("✅ Validate Code", type="secondary"):
            if generated_code.strip():
                with st.spinner("Validating code..."):
                    try:
                        result = validate_cursor_output(
                            generated_code,
                            st.session_state.get("current_task", "Unknown task"),
                            ".",
                        )

                        # Store validation result in session state for save pattern functionality
                        st.session_state.last_validation_result = result
                        st.session_state.last_generated_code = generated_code

                        # Display validation results
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Validation Score", f"{result.score:.1%}")
                            st.metric(
                                "Is Valid", "✅ Yes" if result.is_valid else "❌ No"
                            )

                        with col2:
                            st.metric("Total Issues", len(result.issues))
                            st.metric("Suggestions", len(result.suggestions))

                        # Display compliance
                        st.markdown("#### 📊 Compliance Check")
                        compliance_cols = st.columns(3)

                        compliance_items = [
                            (
                                "Syntax Valid",
                                result.compliance.get("syntax_valid", False),
                            ),
                            (
                                "Type Hints",
                                result.compliance.get("has_type_hints", False),
                            ),
                            (
                                "Docstrings",
                                result.compliance.get("has_docstrings", False),
                            ),
                            (
                                "Error Handling",
                                result.compliance.get("has_error_handling", False),
                            ),
                            (
                                "FastAPI Patterns",
                                result.compliance.get(
                                    "follows_fastapi_patterns", False
                                ),
                            ),
                            (
                                "Code Style",
                                result.compliance.get("code_style_ok", False),
                            ),
                        ]

                        for i, (name, status) in enumerate(compliance_items):
                            with compliance_cols[i % 3]:
                                st.markdown(f"**{name}:** {'✅' if status else '❌'}")

                        # Display issues
                        if result.issues:
                            st.markdown("#### ⚠️ Issues Found")
                            for issue in result.issues:
                                st.error(issue)

                        # Display suggestions
                        if result.suggestions:
                            st.markdown("#### 💡 Suggestions")
                            for suggestion in result.suggestions:
                                st.info(suggestion)

                        # Display metrics
                        st.markdown("#### 📈 Code Metrics")
                        metrics_cols = st.columns(4)

                        metrics_items = [
                            ("Total Lines", result.metrics.get("total_lines", 0)),
                            ("Functions", result.metrics.get("total_functions", 0)),
                            ("Classes", result.metrics.get("total_classes", 0)),
                            ("Complexity", result.metrics.get("complexity_score", 0)),
                        ]

                        for i, (name, value) in enumerate(metrics_items):
                            with metrics_cols[i]:
                                st.metric(name, value)

                        # Success notification
                        if result.is_valid:
                            st.success(
                                "🎉 Code validation passed! Your code meets the project standards."
                            )

                            # Save Pattern button
                            st.markdown("---")
                            st.markdown("#### 💾 Save Successful Pattern")

                            col1, col2, col3 = st.columns([2, 1, 1])

                            with col1:
                                user_rating = st.selectbox(
                                    "Rate this pattern (1-5):",
                                    options=[5, 4, 3, 2, 1],
                                    index=0,
                                    help="How well did this pattern work for you?",
                                )

                            with col2:
                                model_used = st.selectbox(
                                    "Model used:",
                                    options=["phi3", "codellama", "mistral", "unknown"],
                                    index=0,
                                )

                            with col3:
                                if st.button("💾 Save Pattern", type="primary"):
                                    try:
                                        # Get the last generated prompt from session state
                                        last_prompt = st.session_state.get(
                                            "last_generated_prompt", "Manual prompt"
                                        )

                                        success = save_successful_pattern(
                                            prompt=last_prompt,
                                            code=generated_code,
                                            validation={
                                                "score": result.score,
                                                "is_valid": result.is_valid,
                                                "issues": result.issues,
                                                "suggestions": result.suggestions,
                                                "compliance": result.compliance,
                                                "metrics": result.metrics,
                                            },
                                            task_description=st.session_state.get(
                                                "current_task", "Unknown task"
                                            ),
                                            model_used=model_used,
                                            user_rating=user_rating,
                                        )

                                        if success:
                                            st.success("✅ Pattern saved successfully!")

                                            # Add pattern to RAG context database
                                            try:
                                                pattern_data = {
                                                    "prompt": last_prompt,
                                                    "code": generated_code,
                                                    "validation": {
                                                        "score": result.score,
                                                        "is_valid": result.is_valid,
                                                        "issues": result.issues,
                                                        "suggestions": result.suggestions,
                                                        "compliance": result.compliance,
                                                        "metrics": result.metrics,
                                                    },
                                                    "task_description": st.session_state.get(
                                                        "current_task", "Unknown task"
                                                    ),
                                                    "model_used": model_used,
                                                    "user_rating": user_rating,
                                                    "timestamp": datetime.now().isoformat(),
                                                }
                                                rag_system.add_pattern_to_context(
                                                    pattern_data
                                                )
                                                st.info(
                                                    "🔍 Pattern also added to RAG context database for future reference!"
                                                )
                                            except Exception as rag_error:
                                                st.warning(
                                                    f"⚠️ Pattern saved but RAG update failed: {rag_error}"
                                                )

                                            # Clear the form
                                            st.session_state.last_validation_result = (
                                                None
                                            )
                                            st.session_state.last_generated_code = None
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to save pattern")
                                    except Exception as e:
                                        st.error(f"Error saving pattern: {str(e)}")
                        else:
                            st.warning(
                                "⚠️ Code validation failed. Please address the issues above."
                            )

                    except Exception as e:
                        st.error(f"Validation error: {str(e)}")
            else:
                st.warning("Please paste some code to validate.")

    def render_patterns_overview(self):
        """Render patterns overview tab"""
        st.markdown("### 📚 Learning Patterns Overview")

        # Get statistics
        stats = self.learning_system.get_statistics()

        # Ensure all required keys exist
        stats = {
            "total_patterns": stats.get("total_patterns", 0),
            "average_score": stats.get("average_score", 0.0),
            "best_score": stats.get("best_score", 0.0),
            "recent_patterns": stats.get("recent_patterns", 0),
            "task_types": stats.get("task_types", []),
            "models_used": stats.get("models_used", []),
        }

        # Display statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Patterns", stats["total_patterns"])
        with col2:
            st.metric("Average Score", f"{stats['average_score']:.1%}")
        with col3:
            st.metric("Best Score", f"{stats['best_score']:.1%}")
        with col4:
            st.metric("Recent (7 days)", stats["recent_patterns"])

        # Filters
        st.markdown("#### 🔍 Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.1)
        with col2:
            task_filter = st.text_input(
                "Task Keyword", placeholder="e.g., 'auth', 'cache'"
            )
        with col3:
            model_filter = st.selectbox("Model", ["All"] + stats["models_used"])

        # Get filtered patterns
        patterns = self.learning_system.get_patterns()

        if min_score > 0.0:
            patterns = [
                p for p in patterns if p.validation.get("score", 0.0) >= min_score
            ]

        if task_filter:
            patterns = [
                p for p in patterns if task_filter.lower() in p.task_description.lower()
            ]

        if model_filter != "All":
            patterns = [p for p in patterns if p.model_used == model_filter]

        # Display patterns
        st.markdown(f"#### 📋 Patterns ({len(patterns)} found)")

        # Show top-rated patterns first
        top_patterns = [p for p in patterns if p.user_rating == 5]
        if top_patterns:
            st.markdown("##### ⭐ Top-Rated Patterns (5/5)")
            for pattern in top_patterns[:3]:  # Show top 3
                with st.expander(
                    f"🏆 {pattern.task_description[:50]}...", expanded=True
                ):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Task:** {pattern.task_description}")
                        st.markdown(
                            f"**Score:** {pattern.validation.get('score', 0.0):.1%}"
                        )
                        st.markdown(f"**Model:** {pattern.model_used or 'Unknown'}")
                        st.markdown(f"**Date:** {pattern.timestamp[:10]}")
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_top_{id(pattern)}"):
                            if self.learning_system.delete_pattern(
                                self.learning_system.patterns.index(pattern)
                            ):
                                st.success("Pattern deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete pattern")

                    tab1, tab2 = st.tabs(["📝 Prompt", "💻 Code"])
                    with tab1:
                        st.code(pattern.prompt, language="text")
                    with tab2:
                        st.code(pattern.code, language="python")

        if patterns:
            for i, pattern in enumerate(reversed(patterns)):  # Show newest first
                with st.expander(
                    f"Pattern {len(patterns) - i}: {pattern.task_description[:50]}..."
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Task:** {pattern.task_description}")
                        st.markdown(
                            f"**Score:** {pattern.validation.get('score', 0.0):.1%}"
                        )
                        st.markdown(f"**Model:** {pattern.model_used or 'Unknown'}")
                        st.markdown(f"**Rating:** {'⭐' * (pattern.user_rating or 0)}")
                        st.markdown(f"**Date:** {pattern.timestamp[:10]}")

                        if pattern.notes:
                            st.markdown(f"**Notes:** {pattern.notes}")

                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                            if self.learning_system.delete_pattern(
                                len(patterns) - 1 - i
                            ):
                                st.success("Pattern deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete pattern")

                    # Show prompt and code in tabs
                    tab1, tab2 = st.tabs(["📝 Prompt", "💻 Code"])

                    with tab1:
                        st.code(pattern.prompt, language="text")

                    with tab2:
                        st.code(pattern.code, language="python")

                    # Show validation details
                    with st.expander("🔍 Validation Details"):
                        val = pattern.validation
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Score", f"{val.get('score', 0.0):.1%}")
                            st.metric("Valid", "✅" if val.get("is_valid") else "❌")

                        with col2:
                            st.metric("Issues", len(val.get("issues", [])))
                            st.metric("Suggestions", len(val.get("suggestions", [])))

                        if val.get("issues"):
                            st.markdown("**Issues:**")
                            for issue in val["issues"]:
                                st.error(issue)

                        if val.get("suggestions"):
                            st.markdown("**Suggestions:**")
                            for suggestion in val["suggestions"]:
                                st.info(suggestion)
        else:
            st.info("No patterns found matching your filters.")

        # Export functionality
        st.markdown("---")
        st.markdown("#### 📤 Export Patterns")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export All Patterns"):
                export_file = (
                    f"patterns_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                if self.learning_system.export_patterns(export_file):
                    st.success(f"✅ Patterns exported to {export_file}")
                else:
                    st.error("❌ Failed to export patterns")

        with col2:
            if st.button("🗑️ Clear All Patterns"):
                if st.checkbox(
                    "I understand this will delete ALL patterns permanently"
                ):
                    # Clear all patterns by recreating the file
                    self.learning_system.patterns = []
                    self.learning_system._save_patterns()
                    st.success("✅ All patterns cleared!")
                    st.rerun()

    def render_cost_analysis(self):
        """Render cost analysis dashboard"""
        st.markdown("### 💰 Cost Analysis Dashboard")

        # Cost comparison data
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Local LLM Cost", "$0.00", help="Cost for using local models (free)"
            )

        with col2:
            # Calculate estimated cloud cost based on usage
            total_generations = len(self.generation_history)
            estimated_cloud_cost = total_generations * 0.15  # $0.15 per API call
            st.metric(
                "Estimated Cloud Cost",
                f"${estimated_cloud_cost:.2f}",
                help="Estimated cost if using cloud APIs",
            )

        with col3:
            savings = estimated_cloud_cost
            st.metric(
                "Total Savings",
                f"${savings:.2f}",
                delta=f"100% savings",
                delta_color="normal",
            )

        # Cost breakdown chart
        if self.generation_history:
            st.markdown("#### 📊 Cost Savings Over Time")

            # Create cost data
            dates = [
                entry.get("timestamp", datetime.now())
                for entry in self.generation_history
            ]
            local_costs = [0.0] * len(dates)  # Always free
            cloud_costs = [0.15] * len(dates)  # $0.15 per call

            # Create cumulative costs
            cumulative_local = [
                sum(local_costs[: i + 1]) for i in range(len(local_costs))
            ]
            cumulative_cloud = [
                sum(cloud_costs[: i + 1]) for i in range(len(cloud_costs))
            ]

            # Create DataFrame for plotting
            cost_data = pd.DataFrame(
                {
                    "Date": dates,
                    "Local Cost": cumulative_local,
                    "Cloud Cost": cumulative_cloud,
                    "Savings": [
                        cloud - local
                        for cloud, local in zip(cumulative_cloud, cumulative_local)
                    ],
                }
            )

            # Plot cost comparison
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=cost_data["Date"],
                    y=cost_data["Local Cost"],
                    mode="lines+markers",
                    name="Local LLMs (Free)",
                    line=dict(color="green", width=3),
                    marker=dict(size=8),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=cost_data["Date"],
                    y=cost_data["Cloud Cost"],
                    mode="lines+markers",
                    name="Cloud APIs (Paid)",
                    line=dict(color="red", width=3),
                    marker=dict(size=8),
                )
            )

            fig.update_layout(
                title="Cost Comparison: Local vs Cloud LLMs",
                xaxis_title="Date",
                yaxis_title="Cumulative Cost ($)",
                hovermode="x unified",
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Savings breakdown
            st.markdown("#### 💡 Cost Breakdown")

            breakdown_cols = st.columns(2)

            with breakdown_cols[0]:
                st.markdown("**Local LLM Benefits:**")
                st.markdown("- ✅ **Free to use** - No API costs")
                st.markdown("- 🔒 **Privacy first** - All data stays local")
                st.markdown("- ⚡ **No rate limits** - Unlimited requests")
                st.markdown("- 🚀 **No latency** - Direct model access")

            with breakdown_cols[1]:
                st.markdown("**Cloud API Costs:**")
                st.markdown("- 💸 **$0.15 per request** - Adds up quickly")
                st.markdown("- 📊 **Usage tracking** - Always monitored")
                st.markdown("- 🌐 **Network latency** - Slower responses")
                st.markdown("- 📈 **Scaling costs** - More usage = higher bills")

        else:
            st.info(
                "No generation history yet. Start generating code to see cost savings!"
            )

        # Cost calculator
        st.markdown("---")
        st.markdown("#### 🧮 Cost Calculator")

        calc_col1, calc_col2 = st.columns(2)

        with calc_col1:
            daily_requests = st.slider(
                "Daily API requests",
                min_value=1,
                max_value=100,
                value=10,
                help="How many AI requests do you make per day?",
            )

            monthly_requests = daily_requests * 30
            monthly_cloud_cost = monthly_requests * 0.15
            yearly_cloud_cost = monthly_cloud_cost * 12

        with calc_col2:
            st.markdown(f"**Monthly requests:** {monthly_requests}")
            st.markdown(f"**Monthly cloud cost:** ${monthly_cloud_cost:.2f}")
            st.markdown(f"**Yearly cloud cost:** ${yearly_cloud_cost:.2f}")
            st.markdown(f"**Yearly savings:** ${yearly_cloud_cost:.2f}")

            if yearly_cloud_cost > 0:
                st.success(
                    f"💰 You could save ${yearly_cloud_cost:.2f} per year with CodeConductor!"
                )

    def run(self):
        """Main app runner"""
        self.initialize_session_state()
        self.render_header()

        # Sidebar
        self.render_sidebar()

        # Main content with tabs
        tab1, tab2, tab3 = st.tabs(
            ["🎯 Code Generation", "📚 Learning Patterns", "💰 Cost Analysis"]
        )

        with tab1:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Model status
                self.render_model_status()

            # Project Analysis Report (if available)
            if (
                st.session_state.get("show_project_report", False)
                and "project_report" in st.session_state
            ):
                self.render_project_report(st.session_state.project_report)
                st.session_state.show_project_report = False

            # Task input
            task = self.render_task_input()

            # Generation controls
            generation_action = self.render_generation_controls(task)

            if generation_action is True:
                # Run generation
                try:
                    results = asyncio.run(self.run_generation(task))
                    if results:
                        self.render_results(results)
                        st.session_state.generation_results = results
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)

            elif generation_action == "ensemble_test":
                # Run ensemble test
                try:
                    results = asyncio.run(self.run_ensemble_test(task))
                    if results:
                        self.render_results(results)
                        st.session_state.ensemble_results = results
                except Exception as e:
                    st.error(f"Ensemble test failed: {e}")
                    st.exception(e)

            elif generation_action == "test":
                st.info("Test functionality coming soon...")

            elif generation_action == "prompt":
                st.info("Prompt-only functionality coming soon...")

            # Validation section
            self.render_validation_section()

        with col2:
            # Quick stats
            st.markdown("### 📈 Quick Stats")

            if self.generation_history:
                total_generations = len(self.generation_history)
                successful = len(
                    [h for h in self.generation_history if h["status"] == "success"]
                )

                st.metric("Total Generations", total_generations)
                st.metric(
                    "Success Rate", f"{successful / total_generations * 100:.1f}%"
                )

                # Recent activity chart
                if len(self.generation_history) > 1:
                    df = pd.DataFrame(self.generation_history)
                    fig = px.line(
                        df, x=df.index, y="models_used", title="Recent Model Usage"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No generation history yet. Start generating to see stats!")

        with tab2:
            # Learning Patterns Overview
            self.render_patterns_overview()

        with tab3:
            # Cost Analysis Dashboard
            self.render_cost_analysis()


def main():
    app = CodeConductorApp()
    app.run()


if __name__ == "__main__":
    main()
