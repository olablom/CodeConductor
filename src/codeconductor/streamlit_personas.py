#!/usr/bin/env python3
"""
Streamlit Personas Runner for CodeConductor

Minimal UI: pick personas/agents, prompt, rounds, timeout, model; run debate and save artifacts.
Tested manually - excluded from coverage
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import streamlit as st  # pragma: no cover

from codeconductor.debate.local_ai_agent import LocalDebateManager
from codeconductor.debate.personas import build_agents_from_personas, load_personas_yaml
from codeconductor.ensemble.single_model_engine import SingleModelEngine

ARTIFACTS_DIR = Path("artifacts/streamlit_runs")


def run_sync(coro):
    return asyncio.run(coro)


def main() -> None:  # pragma: no cover
    st.set_page_config(
        page_title="CodeConductor – Personas", page_icon="🎭", layout="wide"
    )
    st.title("🎭 CodeConductor – Personas Runner")
    st.caption(
        "Pick personas and run a quick debate; results saved under artifacts/streamlit_runs"
    )

    with st.sidebar:
        st.header("Configuration")
        personas_path = st.text_input("Personas YAML", value="agents/personas.yaml")
        rounds = st.slider("Rounds", min_value=1, max_value=3, value=1, step=1)
        timeout_per_turn = st.slider(
            "Timeout per turn (s)", min_value=15, max_value=180, value=60, step=5
        )
        model_hint = st.text_input("Preferred model (hint)", value="")

        # Load personas and select roles
        try:
            personas = load_personas_yaml(personas_path)
            roles_available = list(personas.keys())
        except Exception as e:
            st.error(f"Failed to load personas: {e}")
            roles_available = []

        roles_selected: list[str] = st.multiselect(
            "Select agents (roles)",
            roles_available,
            default=[r for r in roles_available[:2]],
        )

    prompt = st.text_area(
        "Prompt",
        value="Implement a small FastAPI /items endpoint with tests",
        height=120,
    )

    run_clicked = st.button("Run Debate", type="primary")

    if run_clicked:
        if not roles_selected:
            st.warning("Select at least one agent role")
            return
        if not prompt.strip():
            st.warning("Provide a prompt")
            return

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ARTIFACTS_DIR / f"run_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Starting engine and running debate..."):
            try:
                # Build agents
                personas_map = load_personas_yaml(personas_path)
                agents = build_agents_from_personas(personas_map, roles_selected)

                # Engine
                engine = SingleModelEngine()
                run_sync(engine.initialize())

                try:
                    debate = LocalDebateManager(agents)
                    debate.set_shared_engine(engine)
                    responses = run_sync(
                        debate.conduct_debate(
                            prompt,
                            timeout_per_turn=float(timeout_per_turn),
                            rounds=int(rounds),
                        )
                    )
                finally:
                    run_sync(engine.cleanup())

                st.success("Debate complete")
                # Display and save
                st.subheader("Responses")
                for idx, r in enumerate(responses):
                    st.markdown(f"**Agent {idx + 1}**")
                    st.code(r, language="markdown")

                result = {
                    "timestamp": ts,
                    "personas_file": personas_path,
                    "roles": roles_selected,
                    "rounds": rounds,
                    "timeout_per_turn": timeout_per_turn,
                    "model_hint": model_hint,
                    "prompt": prompt,
                    "responses": responses,
                }
                (out_dir / "result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                st.info(f"Saved: {out_dir / 'result.json'}")

            except Exception as e:
                st.error(f"Run failed: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
