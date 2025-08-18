#!/bin/bash
# Ultra-safe test runner for Git Bash
# This script sets environment variables BEFORE Python starts

echo "🚨 ULTRA-SAFE TEST MODE - Bash Edition"
echo "============================================================"

# Set environment variables BEFORE any Python imports
export CC_HARD_CPU_ONLY=1
export CC_GPU_DISABLED=1
export CC_ULTRA_MOCK=1
export CC_TESTING_MODE=1

# Block CUDA completely
export CUDA_VISIBLE_DEVICES=""
export VLLM_NO_CUDA=1

# Block HuggingFace/Transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Block PyTorch GPU
export TORCH_USE_CUDA_DISABLED=1
export PYTORCH_ENABLE_MPS_FALLBACK=0

# Block other GPU libraries
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "🔒 Environment variables set:"
echo "  CC_HARD_CPU_ONLY: $CC_HARD_CPU_ONLY"
echo "  CC_GPU_DISABLED: $CC_GPU_DISABLED"
echo "  CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "  HF_HUB_OFFLINE: $HF_HUB_OFFLINE"

# Verify Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found in PATH"
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Activate virtual environment if it exists
if [ -f "venv/Scripts/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/Scripts/activate
    echo "✅ Virtual environment activated"
fi

echo "🧪 Starting pytest with GPU protection..."

# Run pytest with filtered tests
if python -m pytest tests/ -k "not gpu and not vllm and not master" -q --tb=short -ra --maxfail=3; then
    exit_code=0
else
    exit_code=$?
fi

echo "🏁 Tests completed with exit code: $exit_code"

# Clean up environment variables
unset CC_HARD_CPU_ONLY
unset CC_GPU_DISABLED
unset CC_ULTRA_MOCK
unset CC_TESTING_MODE

exit $exit_code
