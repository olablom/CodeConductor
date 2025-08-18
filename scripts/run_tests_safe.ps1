# Ultra-safe test runner for PowerShell
# This script sets environment variables BEFORE Python starts

Write-Host "🚨 ULTRA-SAFE TEST MODE - PowerShell Edition" -ForegroundColor Red
Write-Host "=" * 60 -ForegroundColor Red

# Set environment variables BEFORE any Python imports
$env:CC_HARD_CPU_ONLY = "1"
$env:CC_GPU_DISABLED = "1"
$env:CC_ULTRA_MOCK = "1"
$env:CC_TESTING_MODE = "1"

# Block CUDA completely
$env:CUDA_VISIBLE_DEVICES = ""
$env:VLLM_NO_CUDA = "1"

# Block HuggingFace/Transformers
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

# Block PyTorch GPU
$env:TORCH_USE_CUDA_DISABLED = "1"
$env:PYTORCH_ENABLE_MPS_FALLBACK = "0"

# Block other GPU libraries
$env:XLA_PYTHON_CLIENT_PREALLOCATE = "false"
$env:XLA_PYTHON_CLIENT_ALLOCATOR = "platform"

Write-Host "🔒 Environment variables set:" -ForegroundColor Green
Write-Host "  CC_HARD_CPU_ONLY: $env:CC_HARD_CPU_ONLY" -ForegroundColor Yellow
Write-Host "  CC_GPU_DISABLED: $env:CC_GPU_DISABLED" -ForegroundColor Yellow
Write-Host "  CUDA_VISIBLE_DEVICES: '$env:CUDA_VISIBLE_DEVICES'" -ForegroundColor Yellow
Write-Host "  HF_HUB_OFFLINE: $env:HF_HUB_OFFLINE" -ForegroundColor Yellow

# Verify Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Python found: $(python --version)" -ForegroundColor Green

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Blue
    & "venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
}

Write-Host "🧪 Starting pytest with GPU protection..." -ForegroundColor Blue

# Run pytest with filtered tests
try {
    python -m pytest tests/ -k "not gpu and not vllm and not master" -q --tb=short -ra --maxfail=3
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "❌ Error running tests: $_" -ForegroundColor Red
    $exitCode = 1
}

Write-Host "🏁 Tests completed with exit code: $exitCode" -ForegroundColor Cyan

# Clean up environment variables
Remove-Item Env:CC_HARD_CPU_ONLY -ErrorAction SilentlyContinue
Remove-Item Env:CC_GPU_DISABLED -ErrorAction SilentlyContinue
Remove-Item Env:CC_ULTRA_MOCK -ErrorAction SilentlyContinue
Remove-Item Env:CC_TESTING_MODE -ErrorAction SilentlyContinue

exit $exitCode
