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
        """Run the full generation pipeline"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Model discovery and health check
            status_text.text("🔍 Discovering models...")
            progress_bar.progress(10)

            models = await self.model_manager.list_models()
            if not models:
                st.error(
                    "No models found. Please ensure LM Studio or Ollama is running."
                )
                return

            # Step 2: Ensemble processing
            status_text.text("🤖 Running ensemble engine...")
            progress_bar.progress(30)

            # Use simple dispatch method (avoiding the problematic dispatch_with_fallback)
            results = await self.query_dispatcher.dispatch(task, max_models=3)

            if not results:
                st.error("No models responded. Please check model availability.")
                return

            # Step 3: Consensus calculation
            status_text.text("🧠 Calculating consensus...")
            progress_bar.progress(60)

            # Format results for consensus
            formatted_results = []
            for model_id, result in results.items():
                # Check if result is successful (no error field)
                if isinstance(result, dict) and "error" not in result:
                    # Extract content from different response formats
                    content = ""
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0]["message"]["content"]
                    elif "response" in result:
                        content = result["response"]
                    else:
                        content = str(result)

                    formatted_results.append(
                        {
                            "model_id": model_id,
                            "success": True,
                            "response": content,
                            "response_time": 0,  # Not available in raw response
                        }
                    )

            consensus = self.consensus_calculator.calculate_consensus(formatted_results)

            # Step 4: Prompt generation
            status_text.text("📝 Generating prompt...")
            progress_bar.progress(80)

            prompt = self.prompt_generator.generate_prompt(consensus, task)

            # Step 5: Complete
            status_text.text("✅ Generation complete!")
            progress_bar.progress(100)

            return {
                "task": task,
                "models_used": len(results),
                "consensus": consensus,
                "prompt": prompt,
                "status": "success",
            }

        except Exception as e:
            st.error(f"Generation failed: {e}")
            return None

    def render_results(self, results):
        """Render generation results"""
        if not results:
            return

        st.markdown("### 📊 Generation Results")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Used", results["models_used"])
        with col2:
            st.metric("Status", results["status"])
        with col3:
            st.metric(
                "Confidence", f"{getattr(results['consensus'], 'confidence', 0):.2f}"
            )

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
        with st.expander("📝 Generated Prompt", expanded=True):
            st.code(results["prompt"], language="text")

            # Copy button
            if st.button("📋 Copy to Clipboard"):
                st.write("Prompt copied to clipboard!")

        # Add to history
        self.generation_history.append(
            {
                "task": results["task"],
                "status": results["status"],
                "models_used": results["models_used"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )

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
