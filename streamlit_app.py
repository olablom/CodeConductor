#!/usr/bin/env python3
"""
CodeConductor Streamlit App
Minimal UI for running the robust pipeline
"""

import streamlit as st
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="CodeConductor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_pipeline(task_description):
    """Run the CodeConductor pipeline with real-time logging."""
    if not task_description.strip():
        st.error("Please enter a task description!")
        return False
    
    # Prepare command (use Windows-compatible version)
    pipeline_script = "ci/test_pipeline_auto_win.sh" if os.name == 'nt' else "ci/test_pipeline_auto.sh"
    cmd = ["bash", pipeline_script, task_description]
    
    try:
        # Start pipeline process with proper encoding
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Real-time logging with better formatting
        logs = []
        log_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Show spinner while running
        with progress_placeholder.container():
            with st.spinner("🚀 Running CodeConductor pipeline..."):
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logs.append(output.strip())
                        # Show last 30 lines with better formatting
                        display_logs = logs[-30:] if len(logs) > 30 else logs
                        log_text = "\n".join(display_logs)
                        log_placeholder.text_area(
                            "📋 Pipeline Logs (Real-time)",
                            value=log_text,
                            height=300,
                            disabled=True,
                            key="realtime_logs"
                        )
                        time.sleep(0.05)  # Faster updates for better responsiveness
        
        # Clear spinner
        progress_placeholder.empty()
        
        # Wait for completion
        return_code = process.poll()
        
        if return_code == 0:
            st.success("✅ Pipeline completed successfully!")
            return True, logs
        else:
            st.error(f"❌ Pipeline failed with exit code {return_code}")
            return False, logs
            
    except Exception as e:
        st.error(f"❌ Error running pipeline: {str(e)}")
        st.error("💡 This might be an encoding issue. Try running the pipeline manually to see the full error.")
        return False, []

def show_results(logs=None):
    """Display pipeline results and generated code."""
    st.subheader("📋 Pipeline Results")
    
    # Show diff if available
    if os.path.exists("patch.diff"):
        with st.expander("🔧 Code Changes (Patch)", expanded=True):
            try:
                with open("patch.diff", 'r', encoding='utf-8') as f:
                    diff_content = f.read()
                if diff_content.strip():
                    st.code(diff_content, language="diff")
                else:
                    st.info("Patch file is empty")
            except Exception as e:
                st.error(f"Error reading patch file: {e}")
    
    # Check for generated files
    result_files = {
        "Generated Code": "test_project/",
        "Last Error": "last_error.txt",
        "Last Prompt": "last_prompt.txt",
        "Last Feedback": "last_feedback.txt"
    }
    
    for title, filepath in result_files.items():
        if os.path.exists(filepath):
            with st.expander(f"📄 {title}"):
                if os.path.isdir(filepath):
                    # Show directory contents
                    files = list(Path(filepath).glob("*"))
                    if files:
                        for file in files:
                            if file.is_file():
                                with open(file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                st.code(content, language="python")
                    else:
                        st.info("Directory is empty")
                else:
                    # Show file content
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if filepath.endswith('.diff'):
                            st.code(content, language="diff")
                        else:
                            st.text(content)
                    except UnicodeDecodeError:
                        st.warning("File contains binary data or encoding issues")
        else:
            st.info(f"📄 {title}: File not found")
    
    # Show recent logs if provided
    if logs:
        with st.expander("📋 Recent Pipeline Logs"):
            log_text = "\n".join(logs[-50:]) if len(logs) > 50 else "\n".join(logs)
            st.text_area("Pipeline Output", value=log_text, height=200, disabled=True, key="recent_logs")

def main():
    """Main Streamlit app."""
    # Header
    st.title("🚀 CodeConductor")
    st.markdown("**Local-first, multi-agent code generation pipeline**")
    
    # Initialize session state
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'pipeline_logs' not in st.session_state:
        st.session_state.pipeline_logs = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pipeline Info")
        st.info("""
        **How it works:**
        1. Enter your task description
        2. Click 'Run Pipeline'
        3. Watch real-time logs
        4. Review generated code
        
        **Features:**
        - Multi-model ensemble
        - Automatic testing
        - Error handling
        - Real-time feedback
        """)
        
        # Pipeline status with better error display
        if os.path.exists("last_error.txt"):
            with open("last_error.txt", 'r', encoding='utf-8') as f:
                last_error = f.read().strip()
            if last_error:
                st.error("⚠️ Last run had errors")
                # Show error summary
                error_lines = last_error.split('\n')
                if error_lines:
                    error_summary = error_lines[0][:100] + "..." if len(error_lines[0]) > 100 else error_lines[0]
                    st.caption(f"Error: {error_summary}")
            else:
                st.success("✅ Last run successful")
        
        # Theme toggle (removed - not supported in runtime)
        st.markdown("---")
        st.subheader("🎨 Theme")
        st.info("💡 Use ☰ → Settings → Theme to change appearance")
    
    # Main content
    st.header("🎯 Task Input")
    
    # Task input
    task_description = st.text_area(
        "Describe what you want to build or modify:",
        height=150,
        placeholder="Example: Add a --version flag to the CLI tool that displays version 1.0.0"
    )
    
    # Run button with disabled state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Run Pipeline", 
            type="primary", 
            use_container_width=True,
            disabled=st.session_state.pipeline_running
        ):
            if task_description.strip():
                # Set running state
                st.session_state.pipeline_running = True
                st.session_state.pipeline_success = None
                
                # Run pipeline
                success, logs = run_pipeline(task_description)
                st.session_state.pipeline_success = success
                st.session_state.pipeline_logs = logs
                st.session_state.pipeline_running = False
                
                # Show results
                if success:
                    show_results(logs)
                else:
                    show_results(logs)
            else:
                st.error("Please enter a task description!")
    
    # Show results if available
    if hasattr(st.session_state, 'pipeline_success') and st.session_state.pipeline_success is not None:
        show_results(st.session_state.pipeline_logs)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>CodeConductor - Local-first AI code generation</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 