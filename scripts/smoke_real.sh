#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-mistral-7b-instruct-v0.1}"
TOKENS="${2:-128}"

export PYTHONIOENCODING=utf-8
ts=$(date +%Y%m%d_%H%M%S)
outDir="artifacts/run_${ts}"
mkdir -p "$outDir"

echo "[1/3] Doctor (real)"
python -m codeconductor.cli doctor --real --model "$MODEL" --tokens "$TOKENS" --profile | tee "$outDir/doctor_stdout.txt"

echo "[2/3] Personas debate"
codeconductor run --personas agents/personas.yaml --agents architect,coder,bug_hunter \
  --prompt "Implement a small FastAPI /items endpoint with tests" \
  --rounds 1 --timeout-per-turn 60 | tee "$outDir/run_stdout.txt"

echo "[3/3] Focused suite (real)"
python tests/test_codeconductor_2agents_focused.py | tee "$outDir/focused_stdout.txt"

echo "Done. Artifacts at $outDir"
