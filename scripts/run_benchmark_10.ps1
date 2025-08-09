# filename: scripts/run_benchmark_10.ps1
# Purpose: Run N local CodeConductor runs, export bundles, aggregate KPIs → p50/p95 TTFT, success-rate, winner CodeBLEU stats.

param(
  [int]$Runs = 10,
  [string]$Prompt = 'Create a Python Fibonacci function (iterative)'
)

$ErrorActionPreference = 'Stop'

# Resolve repo root and venv python
$repo = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $repo 'venv\Scripts\python.exe'
$PYTHON = if (Test-Path $venvPy) { $venvPy } else { 'python' }

# Ensure PYTHONPATH for local package imports
$env:PYTHONPATH = (Join-Path $repo 'src')
$env:PYTHONIOENCODING = 'utf-8'

# --- Env for engines & selector (tune as needed) ---
$env:ENGINE_BACKENDS = 'vllm,lmstudio'
$env:ALLOW_NET = '0'
$env:TEMP = '0.1'
$env:TOP_P = '0.9'
$env:MAX_TOKENS = '256'
$env:CODEBLEU_WEIGHTS = '0.2,0.6,0.2'
$env:CODEBLEU_LANG = 'python'
$env:CODEBLEU_NORMALIZE = '1'
$env:CODEBLEU_STRIP_COMMENTS = '1'
$env:CODEBLEU_STRIP_DOCSTRINGS = '1'

$art = Join-Path $repo 'artifacts'
$benchDir = Join-Path $art 'benchmarks'
New-Item -ItemType Directory -Force -Path $benchDir | Out-Null

function Get-LatestRunDir {
  $runsPath = Join-Path $art 'runs'
  if (-not (Test-Path $runsPath)) { return $null }
  $runs = Get-ChildItem -Path $runsPath -Directory -ErrorAction SilentlyContinue
  if (-not $runs) { return $null }
  return ($runs | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
}

$rows = @()

for ($i = 1; $i -le $Runs; $i++) {
  Write-Host ("[Run {0}/{1}] Starting..." -f $i, $Runs)

  # 1) Execute local run via venv python
  & $PYTHON (Join-Path $repo 'scripts\generate_ensemble_run.py') --prompt $Prompt | Out-Null

  # 2) Export latest bundle (public_safe v1)
  $exportJson = & $PYTHON (Join-Path $repo 'scripts\export_latest.py')
  $export = $null
  try { $export = $exportJson | ConvertFrom-Json } catch {}

  # 3) Locate latest run dir and parse KPI
  $latest = Get-LatestRunDir
  if (-not $latest) { Write-Warning 'No run dir found'; continue }
  $kpiPath = Join-Path $latest.FullName 'kpi.json'
  if (-not (Test-Path $kpiPath)) { Write-Warning ("No KPI file at {0}" -f $kpiPath); continue }

  $kpi = Get-Content $kpiPath -Raw | ConvertFrom-Json

  $rows += [pscustomobject]@{
    run_id               = $kpi.run_id
    ttft_ms              = [int]$kpi.ttft_ms
    first_prompt_success = [bool]$kpi.first_prompt_success
    winner_model         = $kpi.winner_model
    winner_score         = [double]$kpi.winner_score
    pass_rate_after      = $kpi.pass_rate_after
    zip                  = ($export.zip  | ForEach-Object { $_ }) -join ''
    verified             = ($export.verified | ForEach-Object { $_ }) -join ''
  }

  Write-Host ("[Run {0}] ttft_ms={1} success={2} winner={3} score={4}" -f $i, $kpi.ttft_ms, $kpi.first_prompt_success, $kpi.winner_model, $kpi.winner_score)
}

# --- Aggregate stats ---
if ($rows.Count -eq 0) {
  Write-Error 'No KPI rows collected. Check your run command and artifacts directory.'
  exit 1
}

function Get-Percentile($arr, [double]$p) {
  $sorted = $arr | Sort-Object
  $n = $sorted.Count
  if ($n -eq 1) { return $sorted[0] }
  $idx = [math]::Round(($n - 1) * $p)
  return [int]$sorted[$idx]
}

$ttfts = $rows | ForEach-Object { $_.ttft_ms }
$p50 = Get-Percentile $ttfts 0.50
$p95 = Get-Percentile $ttfts 0.95

$successRate = ($rows | Where-Object { $_.first_prompt_success }).Count / $rows.Count
$winnerMedian = ($rows | Sort-Object winner_score | Select-Object -ExpandProperty winner_score) | Select-Object -Index ([int][math]::Floor(($rows.Count - 1) * 0.5))
$winnerMean = (($rows | Measure-Object -Property winner_score -Average).Average)

$summary = [pscustomobject]@{
  runs                 = $rows.Count
  ttft_ms_p50          = [int]$p50
  ttft_ms_p95          = [int]$p95
  first_prompt_success = [math]::Round($successRate, 4)
  winner_score_median  = [math]::Round($winnerMedian, 4)
  winner_score_mean    = [math]::Round($winnerMean, 4)
  timestamp_utc        = (Get-Date).ToUniversalTime().ToString('o')
}

$ts = (Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss')
$csvPath = Join-Path $benchDir ("runs_{0}.csv" -f $ts)
$jsPath = Join-Path $benchDir ("summary_{0}.json" -f $ts)

$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
$summary | ConvertTo-Json -Depth 3 | Out-File -FilePath $jsPath -Encoding utf8

$summary | ConvertTo-Json -Depth 3
