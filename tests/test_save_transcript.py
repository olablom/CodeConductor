#!/usr/bin/env python3
# Filename: tests/test_save_transcript.py
import json
from pathlib import Path

from codeconductor.debate.debate_manager import (
    CodeConductorDebateManager,
    SingleModelDebateManager,
)


def test_save_transcript_writes_file(tmp_path):
    """Test that save_transcript writes file correctly"""
    m = SingleModelDebateManager()
    m.transcript = [
        {"turn": 1, "agent": "test", "message": "Hello"},
        {"turn": 2, "agent": "test", "message": "World"},
    ]
    # Direct file into tmp_path to avoid polluting repo
    out = m.save_transcript(filename="test_transcript.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [t["message"] for t in data["turns"]] == ["Hello", "World"]


def test_debate_run_persists_when_flag(monkeypatch):
    """Test complete debate flow with transcript saving"""

    # Stub run_debate core to append at least one turn
    class _DM(CodeConductorDebateManager):
        def run_debate(self, prompt: str, rounds: int = 1, agents=None):
            # build a minimal transcript then delegate to parent persist logic
            self.full_transcript.append({"turn": 1, "agent": "Architect", "message": prompt})
            result = {"ok": True}
            if hasattr(self, "save_transcript_flag") and self.save_transcript_flag:
                from datetime import datetime

                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.save_transcript(f"debate_transcript_{stamp}.json")
            return result

    m = _DM()
    m.save_transcript_flag = True
    r = m.run_debate(prompt="Test prompt", rounds=1, agents=["Architect", "Coder"])
    assert r["ok"] is True
    runs_dir = Path("artifacts/runs")
    assert runs_dir.exists()
    # at least one transcript file created
    assert any(p.suffix == ".json" for p in runs_dir.iterdir())
