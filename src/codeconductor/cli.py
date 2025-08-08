#!/usr/bin/env python3
"""
CodeConductor CLI - Command line interface for the AI development platform
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import os
import re

import yaml
from codeconductor.ensemble.model_manager import ModelManager


def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        "models": {
            "primary": "mistralai/codestral-22b-v0.1",
            "fallback": "meta-llama-3.1-8b-instruct",
            "fast": "microsoft/phi-3-mini-4k",
        },
        "agents": {"count": 2, "types": ["architect", "coder", "tester", "reviewer"]},
        "performance": {"max_tokens": 2048, "temperature": 0.2, "timeout": 30},
        "rag": {"enabled": True, "vector_db_path": "./vector_db"},
        "rlhf": {"enabled": True, "learning_rate": 0.1, "max_weight": 10.0},
    }


async def detect_models() -> list:
    """Detect available models in LM Studio"""
    try:
        from codeconductor.ensemble.model_manager import discover_models

        models = await discover_models()
        return [model.id for model in models if model.is_available]
    except Exception as e:
        print(f"⚠️  Warning: Could not detect models: {e}")
        return []


def init_command(args: argparse.Namespace) -> int:
    """Initialize CodeConductor configuration"""
    print("🚀 Initializing CodeConductor...")

    # Create config directory
    config_dir = Path.home() / ".codeconductor"
    config_dir.mkdir(exist_ok=True)

    # Create default config
    config = create_default_config()

    # Detect available models
    print("🔍 Detecting available models...")
    try:
        loop = asyncio.get_event_loop()
        models = loop.run_until_complete(detect_models())

        if models:
            print(f"✅ Found {len(models)} models:")
            for model in models[:5]:  # Show first 5
                print(f"  - {model}")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")

            # Update config with detected models
            config["models"]["available"] = models
        else:
            print("⚠️  No models detected. Please install models in LM Studio first.")

    except Exception as e:
        print(f"⚠️  Could not detect models: {e}")

    # Write config file
    config_file = config_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ Configuration saved to: {config_file}")
    print("\n📋 Next steps:")
    print("1. Install models in LM Studio")
    print("2. Run: python test_master_simple.py")
    print("3. Start coding with: streamlit run src/codeconductor/app.py")

    return 0


def test_command(args: argparse.Namespace) -> int:
    """Run test suite"""
    print("🧪 Running CodeConductor test suite...")
    # Prefer subprocess to avoid import/path issues
    cmd = [sys.executable, "test_master_simple.py"]
    if args.self_reflection:
        cmd.append("--self_reflection")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("❌ test_master_simple.py not found in current directory")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CodeConductor - Local AI agents that debate before coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  codeconductor init          # Initialize configuration
  codeconductor test          # Run test suite
  codeconductor --help        # Show this help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize CodeConductor configuration"
    )
    init_parser.set_defaults(func=init_command)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.add_argument(
        "--self_reflection", action="store_true", help="Enable self-reflection loop"
    )
    test_parser.add_argument(
        "--quick", action="store_true", help="Reserved for future quick mode"
    )
    test_parser.add_argument(
        "--rounds", type=int, default=1, help="Debate rounds per task (default: 1)"
    )
    test_parser.add_argument(
        "--timeout-per-turn",
        type=int,
        default=60,
        help="Seconds per agent turn (default: 60)",
    )
    test_parser.set_defaults(func=test_command)

    # Doctor command
    def doctor_command(args: argparse.Namespace) -> int:
        """System diagnostics for CodeConductor."""
        print("🔎 CodeConductor Doctor")
        print("=" * 30)

        quick = os.getenv("CC_QUICK_CI") == "1"
        print(f"Mode: {'MOCK (CC_QUICK_CI=1)' if quick else 'REAL'}")

        enc = os.getenv("PYTHONIOENCODING") or "(not set)"
        print(f"PYTHONIOENCODING: {enc}")
        if os.name == "nt" and enc.lower() != "utf-8":
            print("Hint: On Windows, set PYTHONIOENCODING=utf-8 for clean logs")

        # Model discovery (safe in mock mode; short network probes otherwise)
        async def _check_models() -> None:
            mm = ModelManager()
            try:
                models = await mm.list_models()
                ids = [m.id for m in models]
                print(f"Models discovered: {len(ids)} -> {ids[:5]}")
            except Exception as e:
                print(f"Model discovery error: {e}")

        asyncio.get_event_loop().run_until_complete(_check_models())

        # Optional health pings when not in mock mode
        if not quick:
            try:
                import requests

                # App health (if you run a local API)
                try:
                    r = requests.get("http://localhost:8000/health", timeout=2)
                    print(f"/health: {r.status_code} {r.text[:80]}")
                except Exception as e:
                    print(f"/health unreachable: {e}")

                # LM Studio models
                try:
                    r = requests.get("http://localhost:1234/v1/models", timeout=2)
                    print(f"LM Studio /v1/models: {r.status_code}")
                except Exception as e:
                    print(f"LM Studio unreachable: {e}")

                # Ollama tags
                try:
                    r = requests.get("http://localhost:11434/api/tags", timeout=2)
                    print(f"Ollama /api/tags: {r.status_code}")
                except Exception as e:
                    print(f"Ollama unreachable: {e}")
            except Exception as e:
                print(f"Diagnostics network step skipped: {e}")

        # Real-mode performance probe with simple timings and report export
        if getattr(args, "real", False) and not quick:
            try:
                import time
                from datetime import datetime
                import requests  # type: ignore
            except Exception:
                time = None
                datetime = None
                requests = None

            backend = None
            model_id = getattr(args, "model", None) or "mistral-7b-instruct-v0.1"
            tokens = int(getattr(args, "tokens", 128) or 128)

            # Helper: check if endpoint is up
            def _is_up(url: str) -> bool:
                if not requests:
                    return False
                try:
                    r = requests.get(url, timeout=2)
                    return r.status_code < 500
                except Exception:
                    return False

            if _is_up("http://localhost:1234/v1/models"):
                backend = "lmstudio"
            elif _is_up("http://localhost:11434/api/tags"):
                backend = "ollama"
            elif os.getenv("VLLM_URL"):
                backend = "vllm"
            else:
                backend = "unknown"

            # Phase timings
            if time is not None:
                t0 = time.perf_counter()
                prepare_start = t0
                # (No heavy prepare here, placeholder)
                prepare_end = time.perf_counter()
                request_start = prepare_end
            else:
                prepare_start = prepare_end = request_start = 0.0

            response_text = ""
            success = False
            error = None

            try:
                if backend == "lmstudio" and requests:
                    url = "http://localhost:1234/v1/chat/completions"
                    payload = {
                        "model": model_id,
                        "messages": [
                            {"role": "user", "content": "Return the word ok."}
                        ],
                        "max_tokens": tokens,
                        "temperature": 0.1,
                    }
                    r = requests.post(url, json=payload, timeout=15)
                    r.raise_for_status()
                    data = r.json()
                    response_text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    success = True
                elif backend == "ollama" and requests:
                    url = "http://localhost:11434/api/generate"
                    payload = {
                        "model": model_id,
                        "prompt": "Return the word ok.",
                        "stream": False,
                        "options": {"num_predict": tokens},
                    }
                    r = requests.post(url, json=payload, timeout=15)
                    r.raise_for_status()
                    data = r.json()
                    response_text = data.get("response", "")
                    success = True
                elif backend == "vllm" and requests and os.getenv("VLLM_URL"):
                    base = os.getenv("VLLM_URL")
                    url = f"{base.rstrip('/')}/v1/completions"
                    payload = {
                        "model": model_id,
                        "prompt": "Return the word ok.",
                        "max_tokens": tokens,
                        "temperature": 0.1,
                    }
                    r = requests.post(url, json=payload, timeout=15)
                    r.raise_for_status()
                    data = r.json()
                    response_text = data.get("choices", [{}])[0].get("text", "")
                    success = True
                else:
                    error = "No reachable backend"
            except Exception as e:  # pragma: no cover
                error = str(e)

            if time is not None:
                request_end = time.perf_counter()
                extract_start = request_end
            else:
                request_end = extract_start = 0.0

            # Simple extract/validate
            num_tokens = max(1, len(response_text.split())) if response_text else 1

            if time is not None:
                extract_end = time.perf_counter()
                validate_start = extract_end
            else:
                extract_end = validate_start = 0.0

            contains_ok = "ok" in (response_text or "").lower()

            if time is not None:
                validate_end = time.perf_counter()
                t1 = validate_end
                total_ms = (t1 - t0) * 1000.0
                prepare_ctx_ms = (prepare_end - prepare_start) * 1000.0
                request_ms = (request_end - request_start) * 1000.0
                extract_ms = (extract_end - extract_start) * 1000.0
                validate_ms = (validate_end - validate_start) * 1000.0
            else:
                total_ms = prepare_ctx_ms = request_ms = extract_ms = validate_ms = 0.0

            # GPU and system memory
            loop = asyncio.get_event_loop()
            try:
                gpu_info = loop.run_until_complete(ModelManager().get_gpu_memory_info())
            except Exception:
                gpu_info = None

            try:
                import psutil  # type: ignore

                vm = psutil.virtual_memory()
                sys_mem_used_mb = int((vm.total - vm.available) / (1024 * 1024))
            except Exception:
                sys_mem_used_mb = None

            decode_tok_s = (
                num_tokens / max(0.001, (request_ms / 1000.0)) if request_ms else None
            )

            from datetime import datetime as _dt  # local import if earlier failed

            report = {
                "timestamp": (_dt.utcnow().isoformat() + "Z") if _dt else None,
                "backend": backend,
                "model_id": model_id,
                "quant": None,
                "context_len": None,
                "ttft_ms": None,
                "decode_tok_s": round(decode_tok_s, 2) if decode_tok_s else None,
                "total_ms": round(total_ms, 1),
                "gpu_mem_used_mb": int(gpu_info["used_gb"] * 1024)
                if gpu_info
                else None,
                "sys_mem_used_mb": sys_mem_used_mb,
                "success": bool(success and contains_ok),
                "error": error,
                "num_tokens": num_tokens,
                "phases": {
                    "prepare_ctx_ms": round(prepare_ctx_ms, 1),
                    "request_ms": round(request_ms, 1),
                    "extract_ms": round(extract_ms, 1),
                    "validate_ms": round(validate_ms, 1),
                },
            }

            ts = _dt.utcnow().strftime("%Y%m%d_%H%M%S") if _dt else "now"
            out = f"doctor_report_{ts}.json"
            try:
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"Saved report: {out}")
            except Exception as e:
                print(f"Failed to write report: {e}")

            if getattr(args, "profile", False):
                cpath = f"profile_turns_{ts}.csv"
                try:
                    with open(cpath, "w", encoding="utf-8") as f:
                        f.write("phase,ms\n")
                        f.write(f"prepare_ctx_ms,{round(prepare_ctx_ms, 1)}\n")
                        f.write(f"request_ms,{round(request_ms, 1)}\n")
                        f.write(f"extract_ms,{round(extract_ms, 1)}\n")
                        f.write(f"validate_ms,{round(validate_ms, 1)}\n")
                    print(f"Saved profile: {cpath}")
                except Exception as e:
                    print(f"Failed to write profile CSV: {e}")

        print("Doctor complete")
        return 0

    doctor_parser = subparsers.add_parser(
        "doctor", help="Show diagnostics and environment health"
    )
    doctor_parser.add_argument(
        "--real",
        action="store_true",
        help="Run real backend probes and record performance",
    )
    doctor_parser.add_argument(
        "--model", type=str, default=None, help="Model id to probe (backend dependent)"
    )
    doctor_parser.add_argument(
        "--tokens", type=int, default=128, help="Max tokens for doctor probe"
    )
    doctor_parser.add_argument(
        "--profile", action="store_true", help="Write per-phase CSV timings"
    )
    doctor_parser.set_defaults(func=doctor_command)

    # Diagnostics helpers (preflight)
    def diag_cursor_command(args: argparse.Namespace) -> int:
        """Show Cursor preflight: env vars and latest diagnostics summary.

        - Prints CURSOR_MODE and CURSOR_API_BASE
        - Reads artifacts/diagnostics/diagnose_latest.txt if present
        - Extracts Ollama status and Cursor API status
        - If a detected Cursor API port is present, prints ready setx lines
        """
        print("Cursor preflight")
        print("=" * 30)

        cursor_mode = os.getenv("CURSOR_MODE") or "(not set)"
        cursor_api_base = os.getenv("CURSOR_API_BASE") or "(not set)"
        print(f"CURSOR_MODE       : {cursor_mode}")
        print(f"CURSOR_API_BASE   : {cursor_api_base}")

        latest_path = Path("artifacts/diagnostics/diagnose_latest.txt")
        if not latest_path.exists():
            print("diagnose_latest.txt not found. Run the PowerShell script:")
            print("  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/diagnose_cursor.ps1 -Ports 11434 3000")
            return 0

        try:
            text = latest_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Failed to read {latest_path}: {e}")
            return 1

        # Parse statuses
        ollama_up = "Port 11434 -> TcpTestSucceeded=True" in text
        detected_port: Optional[str] = None
        m = re.search(r"Detected Cursor API port:\s*(\d+)", text)
        if m:
            detected_port = m.group(1)

        # Heuristic for Cursor API up: any GET /api/health with 2xx/3xx
        cursor_api_up = False
        for line in text.splitlines():
            if "GET /api/health" in line or "GET /health" in line:
                # Ignore Ollama line explicitly
                if "11434" in line:
                    continue
                # Treat presence as signal; detailed status codes often included
                cursor_api_up = True
                break

        print("")
        print("Latest diagnostics summary (diagnose_latest.txt):")
        print(f" - Ollama (11434): {'UP' if ollama_up else 'DOWN'}")
        print(f" - Cursor API    : {'UP (health endpoint responded)' if cursor_api_up else 'NOT LISTENING'}")

        if detected_port:
            print("")
            print(f"Detected Cursor API port: {detected_port}")
            print("Ready-to-run commands (persist for new sessions):")
            print(f"  [Environment]::SetEnvironmentVariable('CURSOR_API_BASE','http://127.0.0.1:{detected_port}','User')")
            print("  [Environment]::SetEnvironmentVariable('CURSOR_MODE','auto','User')")

        # CI hint: don't fail pipeline if Cursor isn't listening
        if os.getenv("CI") == "true":
            print("CI note: diagnostics printed for logging; not failing pipeline if Cursor is not listening.")

        return 0

    diag_parser = subparsers.add_parser("diag", help="Diagnostics utilities")
    diag_subparsers = diag_parser.add_subparsers(dest="diag_cmd", help="Diagnostics commands")
    diag_cursor_parser = diag_subparsers.add_parser(
        "cursor", help="Show Cursor env and latest diagnostics summary"
    )
    diag_cursor_parser.set_defaults(func=diag_cursor_command)

    # Run command (quick debate entry-point with personas)
    def run_command(args: argparse.Namespace) -> int:
        """Run a single debate with optional personas and flags."""
        import asyncio as _asyncio
        from codeconductor.debate.personas import (
            load_personas_yaml,
            build_agents_from_personas,
        )
        from codeconductor.debate.local_ai_agent import LocalDebateManager
        from codeconductor.ensemble.single_model_engine import SingleModelEngine

        if not args.prompt and not args.prompt_file:
            print("❌ Provide --prompt or --prompt-file")
            return 1
        prompt = args.prompt
        if args.prompt_file:
            try:
                p = Path(args.prompt_file)
                prompt = p.read_text(encoding="utf-8")
            except Exception as e:
                print(f"❌ Failed to read prompt file: {e}")
                return 1
        if not prompt:
            print("❌ Empty prompt")
            return 1

        roles = [
            r.strip()
            for r in (args.agents or "architect,coder").split(",")
            if r.strip()
        ]
        personas = load_personas_yaml(args.personas)
        agents = build_agents_from_personas(personas, roles)

        async def _run() -> int:
            engine = SingleModelEngine()
            await engine.initialize()
            try:
                debate = LocalDebateManager(agents)
                debate.set_shared_engine(engine)
                responses = await debate.conduct_debate(
                    prompt,
                    timeout_per_turn=float(args.timeout_per_turn),
                    rounds=int(args.rounds),
                )
                # Minimal output
                print(
                    json.dumps({"responses": responses}, ensure_ascii=False, indent=2)
                )
                return 0
            finally:
                await engine.cleanup()

        return _asyncio.run(_run())

    run_parser = subparsers.add_parser(
        "run", help="Run a single debate with optional personas"
    )
    run_parser.add_argument(
        "--personas", type=str, default=None, help="Path to personas YAML"
    )
    run_parser.add_argument(
        "--agents",
        type=str,
        default="architect,coder",
        help="Comma-separated roles to use",
    )
    run_parser.add_argument(
        "--prompt", type=str, default=None, help="Inline prompt text"
    )
    run_parser.add_argument(
        "--prompt-file", type=str, default=None, help="Path to prompt file"
    )
    run_parser.add_argument("--rounds", type=int, default=1, help="Debate rounds")
    run_parser.add_argument(
        "--timeout-per-turn", type=int, default=60, help="Seconds per agent turn"
    )
    run_parser.set_defaults(func=run_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
