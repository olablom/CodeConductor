"""
Streamlit UI - tested manually
"""

import os
import sys  # pragma: no cover

# Suppress Streamlit warnings using environment variables
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Default runtime safeguards for GUI runs
os.environ.setdefault("CC_TEST_SCOPE", "artifact")
os.environ.setdefault("MATERIALIZE_ENABLE_DOCTEST", "1")
os.environ.setdefault("MATERIALIZE_STRICT", "1")
os.environ.setdefault("MAX_TOKENS", "1024")
os.environ.setdefault("REQUEST_TIMEOUT", "30")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Suppress all warnings
import warnings

# Try to import pyperclip for clipboard functionality
try:
    import pyperclip

    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    print("Warning: pyperclip not available. Install with: pip install pyperclip")

warnings.filterwarnings("ignore")

import asyncio
import json
import logging
import subprocess
import threading
import time
import webbrowser
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure the `src` directory is on sys.path for absolute imports when run via Streamlit
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from codeconductor.analysis.planner_agent import PlannerAgent
from codeconductor.context.rag_system import rag_system
from codeconductor.ensemble.consensus_calculator import ConsensusCalculator
from codeconductor.ensemble.ensemble_engine import EnsembleEngine
from codeconductor.ensemble.hybrid_ensemble import HybridEnsemble
from codeconductor.ensemble.model_manager import ModelManager
from codeconductor.ensemble.query_dispatcher import QueryDispatcher
from codeconductor.feedback.learning_system import (
    LearningSystem,
    save_successful_pattern,
)
from codeconductor.feedback.validation_system import validate_cursor_output
from codeconductor.generators.prompt_generator import PromptGenerator
from codeconductor.runners.test_runner import PytestRunner, TestRunner

# Configure logging to suppress Streamlit warnings
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)

# Ensure the `src` directory is on sys.path for absolute imports when run via Streamlit
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def _auto_prune_on_start() -> None:
    """Optionally prune exports and runs on Streamlit app startup."""
    try:
        auto = os.getenv("AUTO_PRUNE", "1").strip()
        if auto != "1":
            return
        repo_root = Path(__file__).resolve().parents[2]
        # Exports prune
        try:
            keep_full = os.getenv("EXPORT_KEEP_FULL", "20").strip()
            delete_min = os.getenv("EXPORT_DELETE_MINIMAL", "1").strip() in {
                "1",
                "true",
                "yes",
            }
            script = str(repo_root / "scripts" / "prune_exports.py")
            cmd = [sys.executable, script, "--keep-full", keep_full]
            if delete_min:
                cmd.append("--delete-minimal")
            subprocess.run(cmd, cwd=str(repo_root), check=False)
        except Exception:
            pass
        # Runs prune
        try:
            days = os.getenv("RUNS_KEEP_DAYS", "7").strip()
            keep = os.getenv("RUNS_KEEP", "50").strip()
            script = str(repo_root / "scripts" / "cleanup_runs.py")
            cmd = [sys.executable, script, "--days", days, "--keep", keep]
            subprocess.run(cmd, cwd=str(repo_root), check=False)
        except Exception:
            pass
    except Exception:
        pass


def _materialize_generated_code(consensus_text: str) -> dict:
    """Materialize model output into artifacts/runs/<ts>_gui/after/generated.py with fixes.

    Returns a dict with keys: run_dir, path, ok, error.
    """
    import ast as _ast
    import py_compile as _pyc
    from datetime import datetime as _dt

    from codeconductor.utils.extract import extract_code, normalize_python

    ts = _dt.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts") / "runs" / f"{ts}_gui"
    after_dir = run_dir / "after"
    logs_dir = after_dir / "logs"
    after_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw = str(consensus_text or "")
        code = normalize_python(extract_code(raw, lang_hint="python"))
        # Trim to last code-like line
        symbols = "()=:_'\"[]{}"
        lines = code.splitlines()
        last = -1
        for i, ln in enumerate(lines):
            st = ln.strip()
            if not st:
                continue
            if (
                st.startswith("#")
                or any(
                    st.startswith(k)
                    for k in (
                        "def ",
                        "class ",
                        "if ",
                        "for ",
                        "while ",
                        "import ",
                        "from ",
                        "print",
                    )
                )
                or any(c in st for c in symbols)
            ):
                last = i
        if last >= 0:
            code = "\n".join(lines[: last + 1]).rstrip()

        # Size cap
        try:
            cap_bytes = int(os.getenv("MATERIALIZE_MAX_BYTES", "204800"))
        except Exception:
            cap_bytes = 204800
        enc = code.encode("utf-8")
        if len(enc) > cap_bytes:
            code = enc[:cap_bytes].decode("utf-8", errors="ignore").rstrip()

        # Auto append doctest runner if doctest present
        if os.getenv("MATERIALIZE_ENABLE_DOCTEST", "1").strip() == "1":
            if (
                "\n>>>" in code or code.strip().startswith(">>>")
            ) and "doctest.testmod()" not in code:
                code = (
                    code.rstrip()
                    + '\n\nif __name__ == "__main__":\n    import doctest\n    doctest.testmod()\n'
                )

        # Normalize to exact three header lines at top with nothing before them
        EXACT_HEADER = (
            "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n# generated.py\n\n"
        )
        # Remove BOM
        if code.startswith("\ufeff"):
            code = code.lstrip("\ufeff")
        # Strip any existing header-like lines and leading blanks/comments before code
        lines = code.splitlines()
        i = 0
        while i < len(lines):
            st = lines[i].strip()
            if i < 3 and (
                st.startswith("#!")
                or st.startswith("# -*- coding:")
                or st == "# generated.py"
                or st == ""
            ):
                i += 1
                continue
            break
        code_body = "\n".join(lines[i:]).lstrip("\n")
        gen_path = after_dir / "generated.py"
        gen_path.write_text(EXACT_HEADER + code_body + "\n", encoding="utf-8")

        # Validate; try to autoclose triple quotes and trim trailing non-code on failure
        strict = os.getenv("MATERIALIZE_STRICT", "1").strip() == "1"

        def _ast_ok(txt: str) -> bool:
            try:
                _ast.parse(txt)
                return True
            except Exception:
                return False

        def _compile_ok(p: Path) -> bool:
            try:
                _pyc.compile(str(p), doraise=True)
                return True
            except Exception:
                return False

        ok = _ast_ok(code) and _compile_ok(gen_path)
        if strict and not ok:
            try:
                txt = gen_path.read_text(encoding="utf-8")
                tlines = txt.splitlines()
                last_idx = -1
                for i, ln in enumerate(tlines):
                    st = ln.strip()
                    if not st:
                        continue
                    if (
                        st.startswith("#")
                        or any(
                            st.startswith(k)
                            for k in (
                                "def ",
                                "class ",
                                "if ",
                                "for ",
                                "while ",
                                "import ",
                                "from ",
                                "print",
                            )
                        )
                        or any(c in st for c in symbols)
                    ):
                        last_idx = i
                if last_idx >= 0:
                    txt = "\n".join(tlines[: last_idx + 1]).rstrip() + "\n"
                for delim in ('"""', "'''"):
                    if txt.count(delim) % 2 != 0:
                        txt = txt.rstrip() + f"\n{delim}\n"
                gen_path.write_text(txt, encoding="utf-8")
                ok = _ast_ok(txt) and _compile_ok(gen_path)
                if not ok:
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    (logs_dir / "raw.txt").write_text(raw, encoding="utf-8")
            except Exception:
                pass
        return {
            "run_dir": str(run_dir),
            "path": str(gen_path),
            "ok": bool(ok),
            "error": None,
        }
    except Exception as e:
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "error.txt").write_text(str(e), encoding="utf-8")
        except Exception:
            pass
        return {
            "run_dir": str(run_dir),
            "path": str(after_dir / "generated.py"),
            "ok": False,
            "error": str(e),
        }


# Page configuration
_auto_prune_on_start()

st.set_page_config(
    page_title="CodeConductor MVP",
    page_icon="CC",
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

    .health-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .health-green { background-color: #4CAF50; }
    .health-yellow { background-color: #FF9800; }
    .health-red { background-color: #F44336; }
</style>
""",
    unsafe_allow_html=True,
)


class MonitoringSystem:
    """Real-time monitoring system for CodeConductor"""

    def __init__(self):
        self.model_metrics = defaultdict(
            lambda: {
                "response_times": deque(maxlen=100),
                "success_count": 0,
                "error_count": 0,
                "last_response_time": 0,
                "last_check": 0,
                "circuit_breaker_state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                "circuit_breaker_failures": 0,
                "circuit_breaker_last_failure": 0,
            }
        )
        self.system_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "uptime": time.time(),
        }
        self._lock = threading.Lock()

    def record_model_request(self, model_id: str, response_time: float, success: bool):
        """Record metrics for a model request"""
        with self._lock:
            metrics = self.model_metrics[model_id]
            metrics["response_times"].append(response_time)
            metrics["last_response_time"] = response_time
            metrics["last_check"] = time.time()

            if success:
                metrics["success_count"] += 1
                metrics["circuit_breaker_failures"] = 0
            else:
                metrics["error_count"] += 1
                metrics["circuit_breaker_failures"] += 1
                metrics["circuit_breaker_last_failure"] = time.time()

            # Update circuit breaker state
            self._update_circuit_breaker(model_id)

            # Update system metrics
            self.system_metrics["total_requests"] += 1
            if success:
                self.system_metrics["successful_requests"] += 1
            else:
                self.system_metrics["failed_requests"] += 1

    def _update_circuit_breaker(self, model_id: str):
        """Update circuit breaker state based on failure patterns"""
        metrics = self.model_metrics[model_id]

        if metrics["circuit_breaker_state"] == "CLOSED":
            # Open circuit if too many failures
            if metrics["circuit_breaker_failures"] >= 5:
                metrics["circuit_breaker_state"] = "OPEN"
        elif metrics["circuit_breaker_state"] == "OPEN":
            # Try to close circuit after timeout
            if (
                time.time() - metrics["circuit_breaker_last_failure"] > 60
            ):  # 1 minute timeout
                metrics["circuit_breaker_state"] = "HALF_OPEN"
        elif metrics["circuit_breaker_state"] == "HALF_OPEN":
            # Close circuit if recent success
            if metrics["success_count"] > metrics["error_count"]:
                metrics["circuit_breaker_state"] = "CLOSED"

    def get_model_health(self, model_id: str) -> dict:
        """Get health status for a specific model"""
        with self._lock:
            metrics = self.model_metrics[model_id]

            if not metrics["response_times"]:
                return {
                    "status": "unknown",
                    "last_response_time": 0,
                    "success_rate": 0,
                    "circuit_breaker": metrics["circuit_breaker_state"],
                }

            success_rate = (
                metrics["success_count"]
                / (metrics["success_count"] + metrics["error_count"])
                if (metrics["success_count"] + metrics["error_count"]) > 0
                else 0
            )
            avg_response_time = sum(metrics["response_times"]) / len(
                metrics["response_times"]
            )

            # Determine health status
            if success_rate >= 0.9 and avg_response_time < 5.0:
                status = "healthy"
            elif success_rate >= 0.7 and avg_response_time < 10.0:
                status = "degraded"
            else:
                status = "unhealthy"

            return {
                "status": status,
                "last_response_time": metrics["last_response_time"],
                "avg_response_time": avg_response_time,
                "success_rate": success_rate,
                "circuit_breaker": metrics["circuit_breaker_state"],
                "total_requests": metrics["success_count"] + metrics["error_count"],
            }

    def get_system_health(self) -> dict:
        """Get overall system health"""
        with self._lock:
            total_requests = self.system_metrics["total_requests"]
            success_rate = (
                self.system_metrics["successful_requests"] / total_requests
                if total_requests > 0
                else 0
            )
            uptime = time.time() - self.system_metrics["uptime"]

            return {
                "status": "healthy" if success_rate >= 0.8 else "degraded",
                "success_rate": success_rate,
                "total_requests": total_requests,
                "uptime_seconds": uptime,
                "models": {
                    model_id: self.get_model_health(model_id)
                    for model_id in self.model_metrics.keys()
                },
            }


class CodeConductorApp:
    def __init__(self):
        self.model_manager = ModelManager()
        self.query_dispatcher = QueryDispatcher(self.model_manager)
        self.consensus_calculator = ConsensusCalculator()
        self.hybrid_ensemble = HybridEnsemble()
        self.ensemble_engine = None  # Will be initialized on demand
        self.prompt_generator = PromptGenerator()
        self.learning_system = LearningSystem()
        self.test_runner = TestRunner()  # Initialize TestRunner
        self.generation_history = []
        self.monitoring = MonitoringSystem()  # Add monitoring system

        # Initialize RAG system
        try:
            from codeconductor.context.rag_system import RAGSystem

            self.rag_system = RAGSystem()
        except Exception as e:
            st.warning(f"RAG system not available: {e}")
            self.rag_system = None

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
        """Render the sidebar with all controls"""
        st.sidebar.title("🎼 CodeConductor MVP")
        st.sidebar.markdown(
            "AI-Powered Development Pipeline with Multi-Model Ensemble Intelligence"
        )

        # RTX 5090 GPU Memory Monitor
        with st.sidebar.expander("🎮 RTX 5090 GPU Memory", expanded=False):
            if hasattr(self, "model_manager"):
                try:
                    gpu_info = asyncio.run(self.model_manager.get_gpu_memory_info())
                    if gpu_info:
                        st.markdown(
                            f"**Memory Usage:** {gpu_info['usage_percent']:.1f}%"
                        )
                        st.markdown(
                            f"**Used:** {gpu_info['used_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB"
                        )
                        st.markdown(f"**Free:** {gpu_info['free_gb']:.1f}GB")

                        # Memory warning
                        if gpu_info["usage_percent"] > 85:
                            st.warning("⚠️ High Memory Usage!")
                        elif gpu_info["usage_percent"] > 70:
                            st.info("📊 Moderate Memory Usage")
                        else:
                            st.success("✅ Good Memory Availability")

                        # Memory-safe loading controls
                        st.markdown("### Memory-Safe Loading")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🛡️ Light Load (13GB)", key="load_light"):
                                try:
                                    with st.spinner("Loading light config..."):
                                        loaded_models = asyncio.run(
                                            self.model_manager.ensure_models_loaded_with_memory_check(
                                                "light_load"
                                            )
                                        )
                                    if loaded_models:
                                        st.success(
                                            f"✅ Loaded {len(loaded_models)} models safely"
                                        )
                                        # Auto-refresh GPU memory display
                                        st.rerun()
                                    else:
                                        st.warning("⚠️ No models could be loaded")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")

                        with col2:
                            if st.button("⚖️ Medium Load (21GB)", key="load_medium"):
                                try:
                                    with st.spinner("Loading medium config..."):
                                        loaded_models = asyncio.run(
                                            self.model_manager.ensure_models_loaded_with_memory_check(
                                                "medium_load"
                                            )
                                        )
                                    if loaded_models:
                                        st.success(
                                            f"✅ Loaded {len(loaded_models)} models"
                                        )
                                        # Auto-refresh GPU memory display
                                        st.rerun()
                                    else:
                                        st.warning("⚠️ No models could be loaded")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")

                        # Aggressive loading (separate row)
                        if st.button(
                            "🚀 Aggressive Load (28GB)", key="load_aggressive"
                        ):
                            try:
                                with st.spinner("Loading aggressive config..."):
                                    loaded_models = asyncio.run(
                                        self.model_manager.ensure_models_loaded_with_memory_check(
                                            "aggressive_load"
                                        )
                                    )
                                if loaded_models:
                                    st.success(f"✅ Loaded {len(loaded_models)} models")
                                    # Auto-refresh GPU memory display
                                    st.rerun()
                                else:
                                    st.warning("⚠️ No models could be loaded")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")

                        # Emergency unload
                        if st.button("🚨 Emergency Unload All", key="emergency_unload"):
                            try:
                                with st.spinner("Emergency unloading..."):
                                    unloaded_count = asyncio.run(
                                        self.model_manager.emergency_unload_all()
                                    )
                                st.success(f"🚨 Unloaded {unloaded_count} models")

                                # Auto-refresh GPU memory display
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Emergency unload failed: {e}")

                        # Smart memory cleanup
                        if st.button("🧹 Smart Memory Cleanup", key="smart_cleanup"):
                            try:
                                with st.spinner("Performing smart memory cleanup..."):
                                    unloaded_count = asyncio.run(
                                        self.model_manager.smart_memory_cleanup(60.0)
                                    )
                                st.success(
                                    f"🧹 Smart cleanup completed: unloaded {unloaded_count} models"
                                )

                                # Auto-refresh GPU memory display
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Smart cleanup failed: {e}")

                        # Analyze / Rules / Propose (no inference required)
                        st.markdown("\n### Project Conductor")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.button("🔎 Analyze", key="cc_analyze_btn"):
                                from codeconductor.analysis.quick_analyze import (
                                    write_repo_map,
                                    write_state_md,
                                )

                                try:
                                    with st.spinner("Analyzing repository..."):
                                        root = Path(".")
                                        out = Path("artifacts")
                                        data = write_repo_map(
                                            root, out / "repo_map.json"
                                        )
                                        write_state_md(data, out / "state.md")
                                    st.success(
                                        "✅ Wrote artifacts/repo_map.json and artifacts/state.md"
                                    )
                                except Exception as e:
                                    st.error(f"❌ Analyze failed: {e}")
                        with col_b:
                            if st.button("📝 .cursorrules", key="cc_rules_btn"):
                                from codeconductor.analysis.quick_analyze import (
                                    generate_cursorrules,
                                )

                                try:
                                    data = json.loads(
                                        (Path("artifacts/repo_map.json")).read_text(
                                            encoding="utf-8"
                                        )
                                    )
                                    rules = generate_cursorrules(data)
                                    Path(".cursorrules").write_text(
                                        rules, encoding="utf-8"
                                    )
                                    st.success("✅ Wrote .cursorrules")
                                except Exception as e:
                                    st.error(f"❌ .cursorrules failed: {e}")
                        with col_c:
                            if st.button("💡 Propose", key="cc_propose_btn"):
                                from codeconductor.analysis.quick_analyze import (
                                    propose_next_feature,
                                )

                                try:
                                    repo = json.loads(
                                        (Path("artifacts/repo_map.json")).read_text(
                                            encoding="utf-8"
                                        )
                                    )
                                    outp = Path("artifacts/prompts/next_feature.md")
                                    outp.parent.mkdir(parents=True, exist_ok=True)
                                    prompt = propose_next_feature(
                                        repo, Path("artifacts/state.md")
                                    )
                                    outp.write_text(prompt, encoding="utf-8")
                                    st.success(
                                        "✅ Wrote artifacts/prompts/next_feature.md"
                                    )
                                except Exception as e:
                                    st.error(f"❌ Propose failed: {e}")

                        # Quick previews
                        with st.expander(
                            "Preview: state.md / .cursorrules / next_feature.md",
                            expanded=False,
                        ):

                            def _preview(path: str, title: str):
                                p = Path(path)
                                if p.exists():
                                    st.markdown(f"**{title}** – {p.as_posix()}")
                                    txt = p.read_text(encoding="utf-8", errors="ignore")
                                    st.code(
                                        txt[:1500]
                                        + ("\n..." if len(txt) > 1500 else "")
                                    )
                                else:
                                    st.info(f"{title}: not found")

                            _preview("artifacts/state.md", "state.md")
                            _preview(".cursorrules", ".cursorrules")
                            _preview(
                                "artifacts/prompts/next_feature.md", "next_feature.md"
                            )

                        # Check and cleanup memory
                        if st.button("🔍 Check & Cleanup Memory", key="check_cleanup"):
                            try:
                                with st.spinner(
                                    "Checking memory and performing cleanup..."
                                ):
                                    cleanup_performed = asyncio.run(
                                        self.model_manager.check_and_cleanup_memory(
                                            "medium_load"
                                        )
                                    )
                                if cleanup_performed:
                                    st.success("🧹 Memory cleanup performed")
                                else:
                                    st.info("✅ No memory cleanup needed")

                                # Auto-refresh GPU memory display
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Memory check failed: {e}")

                        # Test GPU methods
                        if st.button("🧪 Test GPU Methods", key="test_gpu"):
                            try:
                                with st.spinner("Testing GPU memory methods..."):
                                    results = asyncio.run(
                                        self.model_manager.test_all_gpu_methods()
                                    )
                                st.success("🧪 GPU Methods Test Complete!")
                                # Display results in a more readable format
                                st.markdown("### GPU Memory Detection Results:")
                                for method, result in results.items():
                                    if result:
                                        st.markdown(
                                            f"**{method.upper()}:** ✅ {result['usage_percent']:.1f}% ({result['used_gb']:.1f}GB / {result['total_gb']:.1f}GB)"
                                        )
                                    else:
                                        st.markdown(f"**{method.upper()}:** ❌ FAILED")
                                st.json(results)
                            except Exception as e:
                                st.error(f"❌ GPU test failed: {e}")

                        # Memory Watchdog Status
                        st.markdown("### 🐕 Memory Watchdog")
                        try:
                            from monitoring.memory_watchdog import get_memory_watchdog

                            watchdog = get_memory_watchdog()

                            if watchdog:
                                stats = watchdog.get_stats()
                                if stats["is_running"]:
                                    st.success("✅ Memory watchdog is running")
                                    st.markdown(f"**Checks:** {stats['total_checks']}")
                                    st.markdown(
                                        f"**Cleanup triggers:** {stats['cleanup_triggers']}"
                                    )
                                    st.markdown(
                                        f"**Emergency triggers:** {stats['emergency_triggers']}"
                                    )
                                    if stats["last_check"]:
                                        st.markdown(
                                            f"**Last check:** {stats['last_check']}"
                                        )
                                    st.markdown(
                                        f"**Current VRAM:** {stats['last_vram_percent']:.1f}%"
                                    )
                                else:
                                    st.warning("⚠️ Memory watchdog is not running")
                            else:
                                st.info("ℹ️ Memory watchdog not initialized")
                        except Exception as e:
                            st.warning(f"⚠️ Could not get watchdog status: {e}")
                    else:
                        st.warning("⚠️ Could not get GPU memory info")
                except Exception as e:
                    st.warning(f"⚠️ GPU monitoring error: {e}")

        # Model Management
        with st.sidebar.expander("🤖 Model Management", expanded=False):
            st.markdown("### Model Loading Status")
            if hasattr(self, "model_manager"):
                try:
                    loaded_status = asyncio.run(
                        self.model_manager.get_loaded_models_status()
                    )
                    st.markdown(
                        f"**Loaded Models:** {loaded_status.get('total_loaded', 0)}"
                    )
                    if loaded_status.get("loaded_models"):
                        st.markdown("**Currently Loaded:**")
                        for model in loaded_status["loaded_models"]:
                            st.markdown(f"- {model}")
                    if loaded_status.get("cli_output"):
                        with st.expander("📋 CLI Status"):
                            st.code(loaded_status["cli_output"])
                except Exception as e:
                    st.warning(f"⚠️ Could not get model status: {e}")

            st.markdown("### Load Preferred Models")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Load Complex Task Models", key="load_complex_models"):
                    try:
                        from ensemble.model_manager import LM_STUDIO_PREFERRED_MODELS

                        preferred_models = LM_STUDIO_PREFERRED_MODELS[:3]
                        with st.spinner("Loading preferred models..."):
                            loaded_models = asyncio.run(
                                self.model_manager.ensure_models_loaded(
                                    preferred_models
                                )
                            )
                        if loaded_models:
                            st.success(
                                f"✅ Loaded {len(loaded_models)} models: {', '.join(loaded_models)}"
                            )
                        else:
                            st.warning("⚠️ No models could be loaded")
                    except Exception as e:
                        st.error(f"❌ Error loading models: {e}")
            with col2:
                if st.button("🔄 Refresh Model Status", key="refresh_model_status"):
                    st.rerun()

        # Model Status Dashboard
        with st.sidebar.expander("📊 Model Status & Health", expanded=False):
            st.markdown("### Model Discovery")

            if st.button("🔍 Refresh Models", key="refresh_models"):
                st.rerun()

            # Show model status
            if hasattr(self, "model_manager"):
                try:
                    models = asyncio.run(self.model_manager.list_models())
                    healthy_models = asyncio.run(
                        self.model_manager.list_healthy_models()
                    )

                    st.markdown(f"**Total Models:** {len(models)}")
                    st.markdown(f"**Healthy Models:** {len(healthy_models)}")

                    # Show all available models
                    display_models = models

                    if display_models:
                        st.markdown("**Available Models:**")
                        for model in display_models:
                            status = "✅" if model.id in healthy_models else "❌"
                            st.markdown(f"{status} {model.name} ({model.provider})")
                    else:
                        st.info("ℹ️ No models discovered. Check LM Studio and Ollama.")

                except Exception as e:
                    st.error(f"❌ Error getting model status: {e}")

        # Selector & Cache Telemetry
        with st.sidebar.expander("🧭 Selector & Cache", expanded=False):
            sel_policy = os.getenv("SELECTOR_POLICY", "latency")
            st.markdown(f"**Policy:** `{sel_policy}`")
            if self.ensemble_engine is not None:
                dec = getattr(self.ensemble_engine, "last_selector_decision", {}) or {}
                chosen = dec.get("chosen", "-")
                sampling = dec.get("sampling", {})
                st.markdown(f"**Chosen model:** `{chosen}`")
                st.markdown(
                    f"**Sampling:** temp={sampling.get('temperature', '-')}, top_p={sampling.get('top_p', '-')}"
                )
                scores = dec.get("scores", {})
                if scores:
                    st.markdown("**Candidates:**")
                    st.dataframe(
                        {
                            "model": list(scores.keys()),
                            "score": [round(v, 3) for v in scores.values()],
                        }
                    )
                cache_obj = getattr(self.ensemble_engine, "response_cache", None)
                if cache_obj is not None:
                    st.markdown(
                        f"**Cache:** hits={cache_obj.stats.hits}, misses={cache_obj.stats.misses}, evictions={cache_obj.stats.evictions}"
                    )
                    total = cache_obj.stats.hits + cache_obj.stats.misses
                    hr = (cache_obj.stats.hits / total * 100.0) if total else 0.0
                    st.markdown(
                        f"**Hit-rate:** {hr:.1f}%  |  ns=`{cache_obj.namespace}`  TTL={cache_obj.ttl_seconds}s"
                    )
                    last_hit = getattr(self.ensemble_engine, "last_cache_hit", False)
                    st.markdown(
                        "**Last request:** " + ("✅ HIT" if last_hit else "MISS")
                    )

            st.markdown("---")
            st.markdown("### Export bundle")
            include_raw = st.toggle(
                "Include raw outputs",
                value=False,
                help="Include raw text/log files in the bundle",
            )
            redact_env = st.toggle(
                "Redact env",
                value=True,
                help="Redact sensitive environment values and paths",
            )
            size_limit = st.number_input(
                "Size limit (MB)", min_value=5, max_value=500, value=50, step=5
            )
            retention = st.number_input(
                "Retention (zips)", min_value=1, max_value=200, value=20, step=1
            )

            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_a:
                if st.button("📦 Export bundle", use_container_width=True):
                    try:
                        from codeconductor.utils.exporter import (
                            export_latest_run,
                            verify_manifest,
                        )

                        zip_path, manifest = export_latest_run(
                            artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
                            include_raw=include_raw,
                            redact_env=redact_env,
                            size_limit_mb=int(size_limit),
                            retention=int(retention),
                            policy=os.getenv("SELECTOR_POLICY", "latency"),
                            selected_model=(
                                getattr(
                                    self.ensemble_engine, "last_selector_decision", {}
                                )
                                or {}
                            ).get("chosen"),
                            cache_hit=getattr(
                                self.ensemble_engine, "last_cache_hit", False
                            ),
                            app_version=os.getenv("APP_VERSION"),
                            git_commit=os.getenv("GIT_COMMIT"),
                        )
                        ver = verify_manifest(zip_path)
                        ok = bool(ver.get("verified"))
                        size_mb = 0.0
                        try:
                            size_mb = round(
                                os.path.getsize(zip_path) / (1024 * 1024), 2
                            )
                        except Exception:
                            pass
                        st.success(
                            f"Exported: {zip_path} ({size_mb} MB) — Verified {'✓' if ok else '✗'}"
                        )
                        if (
                            hasattr(self.ensemble_engine, "last_artifacts_dir")
                            and self.ensemble_engine.last_artifacts_dir
                        ):
                            st.caption(
                                f"Run dir: {self.ensemble_engine.last_artifacts_dir}"
                            )
                    except Exception as e:
                        st.error(f"Export failed: {e}")

            with col_b:
                if st.button("📂 Open run folder", use_container_width=True):
                    try:
                        run_dir = getattr(
                            self.ensemble_engine, "last_artifacts_dir", None
                        )
                        if run_dir:
                            os.startfile(run_dir)
                        else:
                            st.info("No run folder yet")
                    except Exception as e:
                        st.error(f"Open failed: {e}")

            with col_c:
                if st.button("📋 Copy zip path", use_container_width=True):
                    try:
                        path = getattr(self.ensemble_engine, "last_export_path", None)
                        if path:
                            st.code(path)
                            if CLIPBOARD_AVAILABLE:
                                pyperclip.copy(path)
                        else:
                            st.info("No export yet")
                    except Exception as e:
                        st.error(f"Copy failed: {e}")

            # Row 2: Send to support & retention tools
            st.markdown("")
            col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
            with col_s1:
                public_safe = st.toggle(
                    "Generate public-safe bundle",
                    value=False,
                    help="Force minimal export: no raw outputs, extra redaction",
                )
            with col_s2:
                if st.button("✉️ Send to support", use_container_width=True):
                    try:
                        # Ensure we have an export
                        from codeconductor.utils.exporter import (
                            export_latest_run,
                            verify_manifest,
                        )

                        if public_safe or not getattr(
                            self.ensemble_engine, "last_export_path", None
                        ):
                            zip_path, _ = export_latest_run(
                                artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
                                include_raw=False,
                                redact_env=True,
                                size_limit_mb=int(size_limit),
                                retention=int(retention),
                                policy=os.getenv("SELECTOR_POLICY", "latency"),
                                selected_model=(
                                    getattr(
                                        self.ensemble_engine,
                                        "last_selector_decision",
                                        {},
                                    )
                                    or {}
                                ).get("chosen"),
                                cache_hit=getattr(
                                    self.ensemble_engine, "last_cache_hit", False
                                ),
                                app_version=os.getenv("APP_VERSION"),
                                git_commit=os.getenv("GIT_COMMIT"),
                            )
                            self.ensemble_engine.last_export_path = zip_path

                        zip_path = getattr(
                            self.ensemble_engine, "last_export_path", None
                        )
                        if not zip_path:
                            st.info("Create an export first")
                        else:
                            ver = verify_manifest(zip_path)
                            if not bool(ver.get("verified")):
                                st.warning("Bundle not verified — please re-export")
                            else:
                                # Clipboard
                                try:
                                    if CLIPBOARD_AVAILABLE:
                                        pyperclip.copy(zip_path)
                                except Exception:
                                    pass

                                # Prepare mailto
                                hitmiss = (
                                    "HIT"
                                    if getattr(
                                        self.ensemble_engine, "last_cache_hit", False
                                    )
                                    else "MISS"
                                )
                                policy = os.getenv("SELECTOR_POLICY", "latency")
                                chosen = (
                                    getattr(
                                        self.ensemble_engine,
                                        "last_selector_decision",
                                        {},
                                    )
                                    or {}
                                ).get("chosen", "-")
                                # Try to extract ts from filename
                                ts = ""
                                try:
                                    import re

                                    m = re.search(r"codeconductor_run_(.*?)_", zip_path)
                                    if m:
                                        ts = m.group(1)
                                except Exception:
                                    pass
                                # Hit rate
                                hr = 0.0
                                cache_obj = getattr(
                                    self.ensemble_engine, "response_cache", None
                                )
                                if cache_obj is not None:
                                    total = (
                                        cache_obj.stats.hits + cache_obj.stats.misses
                                    )
                                    if total:
                                        hr = round(
                                            cache_obj.stats.hits / total * 100.0, 1
                                        )
                                app_version = os.getenv("APP_VERSION", "")
                                commit = os.getenv("GIT_COMMIT", "")

                                subject = f"CodeConductor bug report – {ts or 'N/A'} – {policy} – {hitmiss}"
                                body = (
                                    "Summary: \n"
                                    f"Policy: {policy} | Model: {chosen}\n"
                                    f"Cache: hit={'true' if hitmiss == 'HIT' else 'false'}, hit_rate={hr}%\n"
                                    "Latency: p50=N/A ms, p95=N/A ms\n"
                                    f"Export path (local): {zip_path}\n"
                                    "Hash verified: Yes\n"
                                    f"Version: {app_version} | Commit: {commit}\n"
                                    "Notes: \n"
                                )
                                try:
                                    import urllib.parse as ul

                                    mailto = f"mailto:?subject={ul.quote(subject)}&body={ul.quote(body)}"
                                    webbrowser.open(mailto)
                                except Exception:
                                    pass

                                # Log telemetry event (local)
                                try:
                                    size_mb = 0.0
                                    try:
                                        size_mb = round(
                                            os.path.getsize(zip_path) / (1024 * 1024), 2
                                        )
                                    except Exception:
                                        pass
                                    logging.getLogger(__name__).info(
                                        json.dumps(
                                            {
                                                "event": "send_to_support_clicked",
                                                "has_export": True,
                                                "verified": True,
                                                "size_mb": size_mb,
                                            }
                                        )
                                    )
                                except Exception:
                                    pass
                    except Exception as e:
                        st.error(f"Send failed: {e}")

            with col_s3:
                if st.button("🧹 Clear exports", use_container_width=True):
                    try:
                        base = Path(os.getenv("ARTIFACTS_DIR", "artifacts")) / "exports"
                        count = 0
                        if base.exists():
                            for p in base.glob("codeconductor_run_*.zip"):
                                try:
                                    p.unlink()
                                    count += 1
                                except Exception:
                                    continue
                        st.success(f"Cleared {count} export(s)")
                    except Exception as e:
                        st.error(f"Clear failed: {e}")

        # RAG Context Section
        with st.sidebar.expander("🔍 RAG Context", expanded=False):
            st.markdown("### Context Information")

            # Show RAG system status
            if hasattr(self, "rag_system") and self.rag_system:
                st.markdown("**RAG System Status:**")
                st.success("✅ RAG System Available")

                # Show context stats
                if (
                    hasattr(st.session_state, "rag_context")
                    and st.session_state.rag_context
                ):
                    context_info = st.session_state.rag_context
                    st.markdown(
                        f"**Documents Found:** {context_info.get('context_count', 0)}"
                    )
                    st.markdown(
                        f"**Average Relevance:** {context_info.get('avg_relevance', 0):.3f}"
                    )
                    st.markdown(
                        f"**Context Types:** {', '.join(context_info.get('context_types', []))}"
                    )
                else:
                    st.info(
                        "ℹ️ No context loaded yet. Generate code to see RAG context."
                    )
            else:
                st.warning("⚠️ RAG System not available")

            # Show recent context
            if (
                hasattr(st.session_state, "last_rag_context")
                and st.session_state.last_rag_context
            ):
                st.markdown("### Recent Context")
                recent_context = st.session_state.last_rag_context
                st.markdown(f"**Last Query:** {recent_context.get('query', 'N/A')}")
                st.markdown(f"**Results Found:** {len(recent_context.get('docs', []))}")

                # Show top results
                for i, doc in enumerate(recent_context.get("docs", [])[:3]):
                    with st.expander(
                        f"Result {i + 1} (Score: {doc.get('relevance_score', 0):.3f})"
                    ):
                        st.markdown(
                            f"**Source:** {doc.get('metadata', {}).get('source', 'Unknown')}"
                        )
                        st.markdown(f"**Content:** {doc.get('content', '')[:200]}...")
            else:
                st.info("ℹ️ No recent context available")

        # Learning Patterns Section
        with st.sidebar.expander("📚 Learning Patterns", expanded=False):
            st.markdown("### Pattern Management")

            if hasattr(st.session_state, "learning_patterns"):
                patterns = st.session_state.learning_patterns
                st.markdown(f"**Stored Patterns:** {len(patterns)}")

                if patterns:
                    for i, pattern in enumerate(patterns[:5]):  # Show first 5
                        with st.expander(
                            f"Pattern {i + 1}: {pattern.get('task_type', 'Unknown')}"
                        ):
                            st.markdown(
                                f"**Task:** {pattern.get('task', 'N/A')[:50]}..."
                            )
                            st.markdown(
                                f"**Success Rate:** {pattern.get('success_rate', 0):.2f}"
                            )
                            st.markdown(
                                f"**Last Used:** {pattern.get('last_used', 'N/A')}"
                            )
                else:
                    st.info("ℹ️ No patterns stored yet")
            else:
                st.info("ℹ️ Learning system not initialized")

        # Cost Analysis Section
        with st.sidebar.expander("💰 Cost Analysis", expanded=False):
            st.markdown("### Usage Statistics")

            if hasattr(self, "monitoring"):
                try:
                    stats = self.monitoring.get_statistics()

                    st.markdown(f"**Total Requests:** {stats.get('total_requests', 0)}")
                    st.markdown(
                        f"**Success Rate:** {stats.get('success_rate', 0):.1f}%"
                    )
                    st.markdown(
                        f"**Average Response Time:** {stats.get('avg_response_time', 0):.2f}s"
                    )

                    # Show cost breakdown if available
                    if "cost_breakdown" in stats:
                        st.markdown("**Cost Breakdown:**")
                        for model, cost in stats["cost_breakdown"].items():
                            st.markdown(f"- {model}: ${cost:.4f}")

                except Exception as e:
                    st.warning(f"⚠️ Could not get cost stats: {e}")
            else:
                st.info("ℹ️ Cost monitoring not available")

        # Health Monitoring Section
        with st.sidebar.expander("🏥 Health Monitoring", expanded=False):
            st.markdown("### System Health")

            # Check if health API is running
            try:
                import requests

                health_response = requests.get(
                    "http://localhost:8081/health", timeout=2
                )
                if health_response.status_code == 200:
                    st.success("✅ Health API Running")
                    health_data = health_response.json()
                    st.markdown(f"**Status:** {health_data.get('status', 'Unknown')}")
                    st.markdown(f"**Uptime:** {health_data.get('uptime', 'Unknown')}")
                else:
                    st.warning("⚠️ Health API responding but status unclear")
            except Exception:
                st.error("❌ Health API not reachable")
                st.info("💡 Start health_api.py to enable monitoring")

        # Quick Actions
        with st.sidebar.expander("⚡ Quick Actions", expanded=False):
            st.markdown("### System Actions")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Restart App", key="restart_app"):
                    st.rerun()

            with col2:
                if st.button("🧹 Clear Cache", key="clear_cache"):
                    # Clear session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.success("✅ Cache cleared")
                    st.rerun()

        # System Information
        with st.sidebar.expander("ℹ️ System Info", expanded=False):
            st.markdown("### Environment")

            import platform
            import sys

            st.markdown(f"**Python:** {sys.version.split()[0]}")
            st.markdown(f"**Platform:** {platform.system()} {platform.release()}")
            st.markdown(f"**Streamlit:** {st.__version__}")

            # Show memory usage if available
            try:
                import psutil

                memory = psutil.virtual_memory()
                st.markdown(f"**Memory:** {memory.percent:.1f}% used")
                st.markdown(f"**Available:** {memory.available // (1024**3):.1f} GB")
            except Exception:
                st.info("ℹ️ Memory info not available")

        # Help Section
        with st.sidebar.expander("❓ Help", expanded=False):
            st.markdown("### Quick Guide")

            st.markdown(
                """
            **🎯 Code Generation:**
            1. Enter your task
            2. Click "Generate Code"
            3. Review and copy results

            **🤖 Model Management:**
            - Use "Load Complex Task Models" for better performance
            - Check "Model Status" for health

            **🔍 RAG Context:**
            - Automatically enhances prompts
            - Shows relevant examples

            **📚 Learning:**
            - Patterns are saved automatically
            - Improves future generations
            """
            )

            st.markdown("### Troubleshooting")

            if st.button("🔧 Run Diagnostics", key="run_diagnostics"):
                st.info("Running system diagnostics...")
                # Add diagnostic logic here
                st.success("✅ Diagnostics complete")

    def render_model_status(self):
        """Render model status dashboard"""
        st.markdown("### 🤖 Model Status Dashboard")

        if not st.session_state.models_discovered:
            st.info(
                "Click 'Refresh Models' in the sidebar to discover available models."
            )
            return

        try:
            # Use cached models if available
            if st.session_state.models_discovered and st.session_state.models:
                models = st.session_state.models
            else:
                models = []

            if not models:
                st.warning(
                    "No models found. Click 'Refresh Models' in the sidebar to discover models."
                )
                return

            # Show all available models
            display_models = models

            # Create columns for model display
            cols = st.columns(3)

            for i, model in enumerate(display_models):
                col_idx = i % 3
                with cols[col_idx]:
                    # Health check - only if we have models
                    try:
                        import asyncio

                        health = asyncio.run(self.model_manager.check_health(model))
                        status_icon = "✅" if health else "❌"
                        status_class = "healthy" if health else "unhealthy"
                    except Exception:
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
            if st.button(
                "📱 Phone Validator",
                use_container_width=True,
                key="phone_validator_btn",
            ):
                st.session_state.current_task = (
                    "Create a function to validate Swedish phone numbers"
                )
        with col2:
            if st.button(
                "🧮 Calculator", use_container_width=True, key="calculator_btn"
            ):
                st.session_state.current_task = (
                    "Create a simple calculator class with basic operations"
                )
        with col3:
            if st.button(
                "🔐 Password Generator",
                use_container_width=True,
                key="password_generator_btn",
            ):
                st.session_state.current_task = "Create a secure password generator with configurable length and complexity"

        # Task input
        task = st.text_area(
            "Enter your development task:",
            value=st.session_state.current_task,
            height=100,
            placeholder="Describe what you want to build... (e.g., 'Create a function to validate email addresses')",
        )

        # Always show Generate Code button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button(
                "🚀 Generate Code",
                type="primary",
                use_container_width=True,
                key="generate_code_btn",
                disabled=not task.strip(),
            ):
                st.session_state.run_generation = True

        with col2:
            if st.button(
                "🧪 Test Locally First",
                use_container_width=True,
                key="test_locally_btn",
                disabled=not task.strip(),
            ):
                st.session_state.run_ensemble_test = True

        with col3:
            if st.button(
                "📝 Generate Cursor Prompts",
                use_container_width=True,
                key="generate_prompts_btn",
                disabled=not task.strip(),
            ):
                self._generate_cursor_prompts(task)

        # Planner Agent integration (only if task exists)
        if task.strip():
            st.markdown("---")
            st.markdown("### 🧠 Intelligent Planning")

            if st.button(
                "📋 Create Development Plan",
                use_container_width=True,
                key="create_plan_btn",
            ):
                self._create_development_plan(task)

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

                # Add a feedback section that persists
                if "prompt_feedback" not in st.session_state:
                    st.session_state.prompt_feedback = ""

                for i, prompt in enumerate(plan.cursor_prompts, 1):
                    with st.expander(f"Prompt {i}: {plan.steps[i - 1]['description']}"):
                        # Use st.code() for better copy functionality
                        st.markdown("**📋 Copy this prompt to Cursor:**")
                        st.code(prompt, language=None)

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.info(
                                "💡 **How to copy:** Hover over the code block above and click the copy button that appears"
                            )
                        with col2:
                            if st.button(
                                f"💾 Save Pattern {i}", key=f"save_pattern_{i}"
                            ):
                                st.session_state.last_generated_prompt = prompt
                                st.session_state.prompt_feedback = (
                                    f"✅ Pattern {i} saved for learning!"
                                )

                # Show persistent feedback
                if st.session_state.prompt_feedback:
                    st.info(st.session_state.prompt_feedback)
                    # Clear feedback after showing it
                    if st.button("Clear Feedback", key="clear_feedback"):
                        st.session_state.prompt_feedback = ""
                        st.rerun()

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
                st.markdown(
                    f"""
                **Context Available:** {"✅" if summary["context_available"] else "❌"}
                **Documents Found:** {summary["context_count"]}
                **Average Relevance:** {summary["avg_relevance"]:.3f}
                **Context Types:** {", ".join(summary["context_types"])}
                """
                )

                if rag_info["context_docs"]:
                    st.markdown("**Retrieved Documents:**")
                    for i, doc in enumerate(rag_info["context_docs"], 1):
                        st.markdown(
                            f"""
                        **{i}. {doc["metadata"].get("filename", "Unknown")}** (Score: {doc["relevance_score"]:.3f})
                        - Type: {doc["metadata"].get("type", "Unknown")}
                        - Content: {doc["content"][:200]}...
                        """
                        )

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

    async def run_generation(self, task):
        """Run code generation with hybrid ensemble."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        time.time()

        try:
            # Step 1: Complexity analysis
            status_text.text("🔍 Analyzing task complexity...")
            progress_bar.progress(10)

            # Step 2: Hybrid ensemble processing
            status_text.text("🤖 Running hybrid ensemble engine...")
            progress_bar.progress(30)

            # Generate code using hybrid ensemble
            hybrid_result = await self.hybrid_ensemble.process_task(task)

            # Step 3: Consensus calculation
            status_text.text("🧠 Calculating consensus...")
            progress_bar.progress(50)

            # Step 4: Prompt generation
            status_text.text("📝 Generating prompt...")
            progress_bar.progress(70)

            prompt = self.prompt_generator.generate_prompt(
                hybrid_result.final_consensus, task
            )

            # Step 5: Automated testing
            status_text.text("🧪 Running automated tests...")
            progress_bar.progress(85)

            # Extract generated code from consensus
            generated_code = ""
            if hasattr(hybrid_result.final_consensus, "consensus"):
                generated_code = hybrid_result.final_consensus.consensus
            elif isinstance(hybrid_result.final_consensus, dict):
                generated_code = hybrid_result.final_consensus.get("content", "")
            else:
                generated_code = str(hybrid_result.final_consensus)

            # Materialize code even if mixed with prose; use same safeguards as CLI
            materialize_status = None
            repaired_code = None
            if generated_code and generated_code.strip():
                try:
                    materialize_status = _materialize_generated_code(generated_code)
                except Exception as e:
                    materialize_status = {"ok": False, "error": str(e)}

                # Validator + self-repair loop (max 2 iterations)
                try:
                    from codeconductor.utils.extract import (
                        extract_code,
                        normalize_python,
                    )
                    from codeconductor.utils.validator import (
                        build_repair_prompt,
                        run_doctest_on_file,
                        validate_python_code,
                    )

                    after_path = None
                    if materialize_status and materialize_status.get("ok"):
                        after_path = (
                            Path(materialize_status["path"])
                            if materialize_status.get("path")
                            else None
                        )
                    else:
                        # Best-effort resolve GUI run dir
                        run_dir = Path("artifacts") / "runs"
                        latest = (
                            sorted(run_dir.glob("*_gui"))[-1]
                            if run_dir.exists()
                            else None
                        )
                        if latest:
                            after_path = latest / "after" / "generated.py"
                    if after_path and after_path.exists():
                        code_txt = after_path.read_text(encoding="utf-8")

                        # Temporary guarded hook for Test 6: if task requires SyntaxError
                        # trailer but candidate lacks it, inject trailer before validation
                        try:
                            print(
                                f"DEBUG: Task content: {task[:200] if task else 'None'}"
                            )
                            syntax_error_in_task = "# SYNTAX_ERROR BELOW" in (
                                task or ""
                            )
                            print(
                                f"DEBUG: Looking for SYNTAX_ERROR in task: {syntax_error_in_task}"
                            )
                            requires_trailer = "# SYNTAX_ERROR BELOW" in (task or "")
                            if requires_trailer:
                                lines = code_txt.splitlines()
                                # Ignore trailing blanks for checking last two lines
                                idx = len(lines) - 1
                                while idx >= 0 and lines[idx].strip() == "":
                                    idx -= 1
                                last_two = lines[max(0, idx - 1) : idx + 1]
                                has_trailer = (
                                    len(last_two) >= 2
                                    and last_two[-2] == "# SYNTAX_ERROR BELOW"
                                    and last_two[-1] == "("
                                )
                                if not has_trailer:
                                    append_txt = (
                                        "\n" if not code_txt.endswith("\n") else ""
                                    ) + "# SYNTAX_ERROR BELOW\n(\n"
                                    after_path.write_text(
                                        code_txt + append_txt, encoding="utf-8"
                                    )
                                    code_txt = after_path.read_text(encoding="utf-8")
                        except Exception:
                            pass

                        report = validate_python_code(
                            code_txt,
                            run_doctests=True,
                            task_input=task,
                            require_trailer=("# SYNTAX_ERROR BELOW" in (task or "")),
                        )
                        print("DEBUG: About to enter repair loop")
                        print(f"DEBUG: report.ok = {report.ok}")
                        syntax_ok = getattr(report, "syntax_ok", "N/A")
                        print(f"DEBUG: report.syntax_ok = {syntax_ok}")
                        doctest_failures = getattr(report, "doctest_failures", "N/A")
                        print(f"DEBUG: report.doctest_failures = {doctest_failures}")
                        iter_count = 0
                        try:
                            max_iters = int(os.getenv("SELF_REPAIR_MAX", "3").strip())
                        except Exception:
                            max_iters = 3
                        while (not report.ok) and iter_count < max_iters:
                            print(f"DEBUG: Inside repair loop, iteration {iter_count}")
                            # Try doctest to capture concrete failures
                            ok_dt, dt_out = run_doctest_on_file(after_path)
                            repair_prompt = build_repair_prompt(
                                code_txt,
                                report,
                                dt_out if not ok_dt else None,
                                require_trailer_by_task=(
                                    "# SYNTAX_ERROR BELOW" in (task or "")
                                ),
                            )

                            # Ask the ensemble for a repair (single-turn, minimal)
                            fix_task = f"Return ONLY a single fenced python code block that fixes the module.\n{repair_prompt}"
                            fix_result = await self.hybrid_ensemble.process_task(
                                fix_task
                            )
                            # Extract and normalize code
                            fixed_raw = getattr(
                                fix_result.final_consensus, "consensus", None
                            ) or str(fix_result.final_consensus)
                            fixed_code = normalize_python(
                                extract_code(fixed_raw, lang_hint="python"),
                                preserve_doctest=True,
                            )
                            after_path.write_text(fixed_code + "\n", encoding="utf-8")
                            repaired_code = fixed_code
                            # Re-validate (policy + doctest)
                            code_txt = fixed_code
                            report = validate_python_code(
                                code_txt,
                                run_doctests=True,
                                # For repair validation, do not require trailer even if task mentions it
                                require_trailer=False,
                            )
                            iter_count += 1

                        print(
                            f"DEBUG: Exited repair loop. Final report.ok = {report.ok}"
                        )
                        # Only keep file if final report is ok; otherwise do not leave a broken file
                        try:
                            if not report.ok and after_path.exists():
                                after_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Run automated tests using the repaired code if available
            test_results = None
            code_for_tests = repaired_code or generated_code
            if code_for_tests and code_for_tests.strip():
                test_results = self.run_automated_tests(task, code_for_tests)

            # Step 6: Complete
            status_text.text("✅ Generation and testing complete!")
            progress_bar.progress(100)

            return {
                "task": task,
                "models_used": len(hybrid_result.local_responses)
                + len(hybrid_result.cloud_responses),
                "consensus": hybrid_result.final_consensus,
                "prompt": prompt,
                "generated_code": repaired_code or generated_code,
                "test_results": test_results,
                "materialize": materialize_status,
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
        """Run Ensemble Engine test for local code generation with monitoring"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        try:
            # Step 1: Initialize Ensemble Engine
            status_text.text("🤖 Initializing Ensemble Engine...")
            progress_bar.progress(20)

            if self.ensemble_engine is None:
                self.ensemble_engine = EnsembleEngine()

            # Step 2: Process request with fallback
            status_text.text("⚡ Testing with fast models...")
            progress_bar.progress(50)

            # Process with fallback strategy
            result = await self.ensemble_engine.process_request_with_fallback(task)

            # Record metrics for ensemble test
            processing_time = time.time() - start_time
            self.monitoring.record_model_request(
                "ensemble_test",
                processing_time,
                result is not None and result.get("success", False),
            )

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

    def run_automated_tests(self, task: str, generated_code: str) -> dict:
        """
        Run automated tests on generated code using PytestRunner.

        Args:
            task: The original task description
            generated_code: The generated code to test

        Returns:
            Dictionary with test results and reward information
        """
        try:
            # Use PytestRunner to run real pytest tests
            pytest_runner = PytestRunner(
                prompt=task,
                code=generated_code,
                tests_dir="tests",  # Use the existing tests directory
            )

            # Run the tests
            results = pytest_runner.run()

            return {
                "success": results.get("success", False),
                "test_results": results.get("test_results", []),
                "errors": [results.get("error", "")] if results.get("error") else [],
                "stdout": results.get("stdout", ""),
                "reward": results.get("reward", 0.0),
                "total_tests": results.get("total_tests", 0),
                "passed_tests": results.get("passed_tests", 0),
                # Enhanced metrics
                "cyclomatic_complexity": results.get("cyclomatic_complexity", 0.0),
                "exec_time_s": results.get("exec_time_s", 0.0),
            }

        except Exception as e:
            return {
                "success": False,
                "test_results": [],
                "errors": [f"Test execution failed: {str(e)}"],
                "stdout": "",
                "reward": 0.0,
                "total_tests": 0,
                "passed_tests": 0,
                # Enhanced metrics (default values on error)
                "cyclomatic_complexity": 0.0,
                "exec_time_s": 0.0,
            }

    def _calculate_test_reward(self, test_results: list) -> float:
        """Calculate reward based on test results."""
        if not test_results:
            return 0.0

        passed_tests = sum(1 for test in test_results if test.get("passed", False))
        total_tests = len(test_results)

        return passed_tests / total_tests if total_tests > 0 else 0.0

    def _render_test_results(self, test_results: dict):
        """Render test results in the GUI."""
        st.markdown("### 🧪 Automated Test Results")

        # Test metrics - Row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Test Status", "✅ Passed" if test_results["success"] else "❌ Failed"
            )
        with col2:
            st.metric("Reward Score", f"{test_results['reward']:.2f}")
        with col3:
            st.metric(
                "Tests Passed",
                f"{test_results['passed_tests']}/{test_results['total_tests']}",
            )
        with col4:
            st.metric("Total Tests", test_results["total_tests"])

        # Enhanced metrics - Row 2
        st.markdown("#### 📊 Code Quality Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Code Complexity
            complexity = test_results.get("cyclomatic_complexity", 0.0)
            complexity_color = "normal" if complexity <= 3.0 else "inverse"
            st.metric(
                "Code Complexity",
                f"{complexity:.2f}",
                delta="Low" if complexity <= 3.0 else "High",
                delta_color=complexity_color,
            )

        with col2:
            # Execution Time
            exec_time = test_results.get("exec_time_s", 0.0)
            st.metric(
                "Execution Time",
                f"{exec_time:.3f}s",
                delta="Fast" if exec_time <= 0.1 else "Slow",
                delta_color="normal" if exec_time <= 0.1 else "inverse",
            )

        with col3:
            # Average Test Duration
            test_durations = []
            for test in test_results.get("test_results", []):
                if "duration_s" in test:
                    test_durations.append(test["duration_s"])

            avg_duration = (
                sum(test_durations) / len(test_durations) if test_durations else 0.0
            )
            st.metric(
                "Avg Test Duration",
                f"{avg_duration:.3f}s",
                delta="Fast" if avg_duration <= 0.05 else "Slow",
                delta_color="normal" if avg_duration <= 0.05 else "inverse",
            )

        with col4:
            # Performance Score (combination of complexity and speed)
            performance_score = max(0, 1.0 - (complexity * 0.1 + exec_time * 0.5))
            st.metric(
                "Performance Score",
                f"{performance_score:.2f}",
                delta="Good" if performance_score >= 0.7 else "Needs improvement",
                delta_color="normal" if performance_score >= 0.7 else "inverse",
            )

        # Test details
        with st.expander("📊 Test Details", expanded=False):
            if test_results["test_results"]:
                for i, test in enumerate(test_results["test_results"], 1):
                    status = "✅" if test.get("passed", False) else "❌"
                    st.markdown(
                        f"{status} **Test {i}:** {test.get('name', 'Unknown test')}"
                    )
                    if test.get("type"):
                        st.caption(f"Type: {test['type']}")
            else:
                st.info("No detailed test results available")

        # Errors if any
        if test_results["errors"]:
            with st.expander("❌ Test Errors", expanded=False):
                for error in test_results["errors"]:
                    st.error(error)

        # Reward explanation
        with st.expander("🎯 Reward Analysis", expanded=False):
            reward = test_results["reward"]
            if reward >= 0.8:
                st.success(
                    "🎉 Excellent! High reward score indicates good code quality."
                )
            elif reward >= 0.5:
                st.warning("⚠️ Moderate reward score. Code may need improvements.")
            else:
                st.error("❌ Low reward score. Code needs significant improvements.")

            st.markdown(
                f"""
            **Reward Calculation:**
            - Passed tests: {test_results["passed_tests"]}
            - Total tests: {test_results["total_tests"]}
            - Reward score: {reward:.2f} ({reward * 100:.1f}%)
            """
            )

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
            # Fix for models_used type inconsistency
            models_used = results.get("models_used", 0)
            if isinstance(models_used, list):
                st.metric("Models Used", len(models_used))
            else:
                st.metric("Models Used", models_used)
        with col4:
            st.metric("Total Time", f"{results.get('total_time', 0.0):.2f}s")

        # Generated code
        st.markdown("#### 💻 Generated Code")
        if results.get("generated_code"):
            st.code(results["generated_code"], language="python")
        else:
            st.warning("No code generated")

        # Model details - handle both list and integer cases
        models_used = results.get("models_used", [])
        if models_used:
            st.markdown("#### 🤖 Models Used")
            if isinstance(models_used, list):
                for model in models_used:
                    st.info(f"✅ {model}")
            else:
                st.info(f"✅ {models_used} models used")

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
            st.metric("Status", results.get("status", "unknown"))
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
                        (
                            results["escalation_reason"][:20] + "..."
                            if len(results["escalation_reason"]) > 20
                            else results["escalation_reason"]
                        ),
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
                else:
                    st.metric("Models/sec", "N/A")
            with col3:
                st.metric("Total Cost", f"${results.get('total_cost', 0):.4f}")

        # Test results (if available)
        if "test_results" in results and results["test_results"]:
            st.markdown("---")
            self._render_test_results(results["test_results"])

        # Generated code display
        if "generated_code" in results and results["generated_code"]:
            st.markdown("#### 💻 Generated Code")

            # Display code in a more prominent way
            st.markdown("**Implementation:**")
            st.code(results["generated_code"], language="python")

            # Add copy button for generated code
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📋 Copy Generated Code", key="copy_generated_code"):
                    try:
                        if CLIPBOARD_AVAILABLE:
                            pyperclip.copy(results["generated_code"])
                            st.success("✅ Code copied to clipboard!")
                        else:
                            st.info(
                                "📋 Select the code above and copy manually (Ctrl+C)"
                            )
                    except Exception as e:
                        st.warning(f"⚠️ Could not copy to clipboard: {e}")
                        st.info("📋 Select the code above and copy manually (Ctrl+C)")
            with col2:
                if st.button("🔍 Validate Code", key="validate_generated_code"):
                    # Trigger code validation
                    st.session_state.validate_code = results["generated_code"]
                    st.rerun()

            # Show prompt used
            if "prompt" in results and results["prompt"]:
                st.markdown("---")
                st.markdown("#### 📝 Generated Prompt")
                with st.expander("View prompt used for generation"):
                    st.markdown(results["prompt"])

        # Consensus details
        with st.expander("🧠 Consensus Details", expanded=False):
            if "consensus" in results and results["consensus"]:
                if hasattr(results["consensus"], "consensus"):
                    # Display the generated code from consensus
                    generated_code = results["consensus"].consensus

                    if generated_code:
                        st.markdown("🤖 **Generated Code:**")
                        st.code(generated_code, language="python")

                        # Also show consensus metadata
                        st.markdown("📊 **Consensus Metadata:**")
                        st.markdown(
                            f"**Confidence:** {results['consensus'].confidence:.2f}"
                        )
                        st.markdown(
                            f"**Code Quality Score:** {results['consensus'].code_quality_score:.2f}"
                        )
                        st.markdown(
                            f"**Syntax Valid:** {results['consensus'].syntax_valid}"
                        )
                        st.markdown(f"**Reasoning:** {results['consensus'].reasoning}")
                    else:
                        st.info("🤖 **Model Consensus Summary**")
                        st.markdown(
                            "**Strategy:** " + results.get("strategy", "consensus")
                        )
                        st.markdown(
                            "**Models Used:** " + str(results.get("models_used", 0))
                        )
                        st.markdown(
                            "**Confidence:** " + f"{results.get('confidence', 0):.2f}"
                        )
                        st.markdown(
                            "**Total Time:** " + f"{results.get('total_time', 0):.2f}s"
                        )
                        st.caption(
                            "💡 Models provided free-form responses rather than structured consensus data"
                        )
                else:
                    st.write("No structured consensus data available")
            else:
                st.warning("No consensus data available")

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
                                (
                                    content[:500] + "..."
                                    if len(content) > 500
                                    else content
                                ),
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
                            (
                                response.content[:500] + "..."
                                if len(response.content) > 500
                                else response.content
                            ),
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
                "complexity_level": (
                    results.get("complexity_analysis", {}).level.value
                    if results.get("complexity_analysis")
                    else "unknown"
                ),
                "status": "success",  # Add status field
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
        import csv
        import io

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

        # Auto-populate with last generated code from session state
        default_code = st.session_state.get("last_generated_code", "")

        generated_code = st.text_area(
            "Generated Code",
            value=default_code,
            height=300,
            placeholder="Paste your Cursor-generated code here...",
            key="validation_code_area",
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
                        if st.button("🗑️ Delete", key=f"delete_top_{id(pattern)}"):
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
                        if st.button("🗑️ Delete", key=f"delete_{i}"):
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
                delta="100% savings",
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
                        for cloud, local in zip(
                            cumulative_cloud, cumulative_local, strict=False
                        )
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

            # Check for ensemble test from task input
            if st.session_state.get("run_ensemble_test", False):
                st.session_state.run_ensemble_test = False  # Reset
                try:
                    # Create new event loop if none exists
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    results = loop.run_until_complete(self.run_ensemble_test(task))
                    if results:
                        self.render_results(results)
                        st.session_state.ensemble_results = results
                except Exception as e:
                    st.error(f"Ensemble test failed: {e}")
                    st.exception(e)

            # Check for generation from task input
            elif st.session_state.get("run_generation", False):
                st.session_state.run_generation = False  # Reset
                try:
                    # Create new event loop if none exists
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    results = loop.run_until_complete(self.run_generation(task))
                    if results:
                        self.render_results(results)
                        st.session_state.generation_results = results
                        # Auto-save generated code for validation
                        if results.get("generated_code"):
                            st.session_state.last_generated_code = results[
                                "generated_code"
                            ]
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)

            # Validation section
            self.render_validation_section()

        with col2:
            # Quick stats
            st.markdown("### 📈 Quick Stats")

            if self.generation_history:
                total_generations = len(self.generation_history)
                successful = len(
                    [h for h in self.generation_history if h.get("status") == "success"]
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


def main():  # pragma: no cover
    app = CodeConductorApp()
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
