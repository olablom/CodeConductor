param(
    [string]$Model = "mistral-7b-instruct-v0.1",
    [int]$Tokens = 128
)

$env:PYTHONIOENCODING = "utf-8"
$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$outDir = Join-Path -Path (Resolve-Path ".").Path -ChildPath "artifacts\run_$ts"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "[1/3] Doctor (real)"
python -m codeconductor.cli doctor --real --model $Model --tokens $Tokens --profile | Tee-Object -FilePath (Join-Path $outDir "doctor_stdout.txt")

Write-Host "[2/3] Personas debate"
codeconductor run --personas agents/personas.yaml --agents architect, coder, bug_hunter `
    --prompt "Implement a small FastAPI /items endpoint with tests" `
    --rounds 1 --timeout-per-turn 60 | Tee-Object -FilePath (Join-Path $outDir "run_stdout.txt")

Write-Host "[3/3] Focused suite (real)"
python tests/test_codeconductor_2agents_focused.py | Tee-Object -FilePath (Join-Path $outDir "focused_stdout.txt")

Write-Host "Done. Artifacts at $outDir"
