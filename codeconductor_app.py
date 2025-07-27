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
from generators.prompt_generator import PromptGenerator
from integrations.notifications import notify_success, notify_error

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
        self.prompt_generator = PromptGenerator()
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
                help="Enter the root folder of your codebase"
            )

            # Knapp för att starta analys
            if st.button("🔍 Analyze Project"):
                if not project_path:
                    st.error("Please enter a valid project path.")
                else:
                    try:
                        from analysis.project_analyzer import ProjectAnalyzer
                        analyzer = ProjectAnalyzer()
                        with st.spinner("🔍 Scanning project..."):
                            # Fas 1: basic scanning
                            routes = analyzer.scan_fastapi_routes(project_path)
                            schema = analyzer.introspect_postgresql()  # Optional DB analysis
                            report = analyzer.generate_report(routes, schema)
                        st.success("✅ Analysis complete!")
                        st.session_state.project_report = report
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
            # Use cached models from session state
            models = st.session_state.get("models", [])

            if not models:
                # Fallback: try to get models directly
                import asyncio

                models = asyncio.run(self.model_manager.list_models())
                st.session_state.models = models

            # Create columns for model display
            cols = st.columns(3)

            for i, model in enumerate(models):
                col_idx = i % 3
                with cols[col_idx]:
                    # Health check
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

        return task

    def render_generation_controls(self, task):
        """Render generation controls"""
        if not task.strip():
            st.warning("Please enter a task to begin generation.")
            return False

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.button("🚀 Generate Code", type="primary", use_container_width=True):
                return True

        with col2:
            if st.button("🧪 Test Only", use_container_width=True):
                return "test"

        with col3:
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

    def render_results(self, results):
        """Render generation results"""
        if not results:
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
            st.metric("Database Tables", len(report.get("schema", {}).get("tables", [])))
        with col3:
            st.metric("Files Analyzed", report.get("files_analyzed", 0))

        # FastAPI Routes
        if report.get("routes"):
            st.markdown("#### 🚀 FastAPI Routes")
            for route in report["routes"]:
                with st.expander(f"{route['method']} {route['path']}", expanded=False):
                    st.write(f"**Function:** {route['function']}")
                    st.write(f"**File:** {route['file']}")
                    if route.get('parameters'):
                        st.write(f"**Parameters:** {route['parameters']}")

        # Database Schema
        if report.get("schema", {}).get("tables"):
            st.markdown("#### 🗄️ Database Schema")
            for table in report["schema"]["tables"]:
                with st.expander(f"Table: {table['name']}", expanded=False):
                    st.write(f"**Columns:** {len(table['columns'])}")
                    for col in table['columns']:
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
                    mime="application/json"
                )
        with col2:
            if st.button("📊 Export as CSV"):
                # Convert report to CSV format
                csv_data = self.convert_report_to_csv(report)
                st.download_button(
                    label="Download CSV Report",
                    data=csv_data,
                    file_name="project_analysis.csv",
                    mime="text/csv"
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
            writer.writerow([
                "Route", 
                f"{route['method']} {route['path']}", 
                f"Function: {route['function']}, File: {route['file']}"
            ])
        
        # Write tables
        for table in report.get("schema", {}).get("tables", []):
            writer.writerow([
                "Table", 
                table['name'], 
                f"Columns: {len(table['columns'])}"
            ])
        
        return output.getvalue()

    def run(self):
        """Main app runner"""
        self.initialize_session_state()
        self.render_header()

        # Sidebar
        self.render_sidebar()

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            # Model status
            self.render_model_status()

            # Project Analysis Report (if available)
            if st.session_state.get("show_project_report", False) and "project_report" in st.session_state:
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

            elif generation_action == "test":
                st.info("Test functionality coming soon...")

            elif generation_action == "prompt":
                st.info("Prompt-only functionality coming soon...")

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


def main():
    app = CodeConductorApp()
    app.run()


if __name__ == "__main__":
    main()
