#!/usr/bin/env bash
# UV, BENCH_VENV, PYTHON_VERSION, UV_CACHE_DIR may be overridden.
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/.." && pwd)
UV=${UV:-uv}
BENCH_VENV=${BENCH_VENV:-"$REPO_DIR/.venv"}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$REPO_DIR/.cache/uv"}
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --check ) ]]; then
    echo 'Usage: bash benchmarks/setup.sh [--check]' >&2; exit 2
fi
if [[ ${1:-} != --check ]]; then
    if [[ ! -x "$BENCH_VENV/bin/python" ]]; then
        "$UV" venv --python "${PYTHON_VERSION:-3.12}" "$BENCH_VENV"
    fi
    "$UV" pip install --python "$BENCH_VENV/bin/python" -r "$BENCH_DIR/requirements.lock"
    # Build against this environment's torch/glibc; release wheels may require a newer glibc.
    FLASH_ATTENTION_FORCE_BUILD=${FLASH_ATTENTION_FORCE_BUILD:-TRUE} MAX_JOBS=${MAX_JOBS:-8} \
        "$UV" pip install --python "$BENCH_VENV/bin/python" --no-deps --no-build-isolation \
        "${FLASH_ATTN_PACKAGE:-flash-attn==2.8.3}"
fi
"$UV" pip check --python "$BENCH_VENV/bin/python"
"$BENCH_VENV/bin/python" - <<'PY'
import torch, torchvision, diffusers, transformers, av, omegaconf, peft, flash_attn
print(f'torch={torch.__version__}, CUDA={torch.version.cuda}, torchvision={torchvision.__version__}')
print(f'diffusers={diffusers.__version__}, transformers={transformers.__version__}')
print(f'Visible GPUs: {torch.cuda.device_count()}')
PY
mkdir -p "$REPO_DIR/.cache/benchmark"
"$UV" pip freeze --python "$BENCH_VENV/bin/python" > "$REPO_DIR/.cache/benchmark/environment.freeze.txt"
