#!/usr/bin/env bash
# Entry point for the dense memory/time sweep on Cloud.ru.
#
#   mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/efficient-training-membench-cloud \
#     --branch main --entry scripts/cloud_run.sh --image torch28 --no-pip --gpus 1 \
#     --note membench-dense \
#     --args "--models 500m --variants adamw_bf16_state_fp32 --microbatches 1,2"
#
# First argument may instead be:
#   probe    report what the image provides and what the sweep still needs
#   export   print one line per recorded point, for `mlsub logs`
#   peek     print the newest log and which points exist so far
#   disk     free space, and a write probe on every volume
#
# A failed mlsub job shows no logs at all, so output is teed to a persistent volume
# and this script always exits zero; the real status is the EXIT= line.
set -u

# /home/jovyan reached 0 bytes free on 2026-08-27 -- the platform could not create
# its own log symlinks there. Everything this job writes, including the pip prefix,
# goes to nfs3 instead. mlsub points PYTHONUSERBASE at /home/jovyan by default, so
# overriding it here is what keeps an install from failing on ENOSPC.
ROOT=${MEMBENCH_ROOT:-/workspace-SR006.nfs3/dense-membench}
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
export PYTHONUSERBASE="$ROOT/pyuser"
mkdir -p "$LOGS" "$RESULTS" "$PYTHONUSERBASE"

export PYTHONPATH="$(pwd):$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# tiktoken: models/base.py imports it at module level.
# transformers: COAT's QuantizationConfig subclasses PretrainedConfig.
# loguru: MuonLite logs Newton-Schulz compilations through it.
REQUIRED=(tiktoken transformers loguru)

ensure_dependencies() {
  local missing=()
  for package in "${REQUIRED[@]}"; do
    python3 -c "import $package" 2>/dev/null || missing+=("$package")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "installing into $PYTHONUSERBASE: ${missing[*]}"
    # --user, never --target: --target ignores what the image already has and would
    # pull a second copy of torch.
    python3 -m pip install --user --disable-pip-version-check -q "${missing[@]}" || return 1
  fi
  for package in "${REQUIRED[@]}"; do
    python3 -c "import $package" || return 1
  done
}

case "${1:-}" in
  probe)
    echo "python: $(python3 -V 2>&1)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || true
    python3 - <<'PY'
import importlib.util
import sys
print("sys.path[0:3]", sys.path[:3])
for name in ("torch", "triton", "tiktoken", "transformers", "loguru",
             "models.utils", "optim.fp8_state", "optim.sota_opt.soap.soap_harvard",
             "optim.sota_opt.fp8_soap", "third_party.lite.muonlite",
             "third_party.coat.utils._fp8_quantization_config",
             "third_party.coat.optimizer.triton_fp8_adamw"):
    try:
        found = importlib.util.find_spec(name) is not None
    except Exception as error:
        found = f"error: {type(error).__name__}: {error}"
    print(f"{name:52s} {found}")
PY
    ensure_dependencies && echo "DEPENDENCIES=ok" || echo "DEPENDENCIES=failed"
    python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
    exit 0
    ;;
  export)
    python3 -m bench.run_sweep --export-only --output-dir "$RESULTS"
    exit 0
    ;;
  peek)
    echo "=== recorded points ==="
    ls -1 "$RESULTS/runs" 2>/dev/null | sort || echo "none yet"
    newest=$(ls -t "$LOGS"/*.log 2>/dev/null | head -1)
    echo "=== tail of ${newest:-no log} ==="
    [ -n "$newest" ] && tail -"${2:-150}" "$newest"
    exit 0
    ;;
  disk)
    df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
    echo "recorded points: $(ls -1 "$RESULTS/runs" 2>/dev/null | wc -l)"
    for target in "$RESULTS" /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan; do
      if probe=$(mktemp "$target/.membench-probe.XXXXXX" 2>&1); then
        echo "writable: $target"; rm -f "$probe"
      else
        echo "NOT writable: $target ($probe)"
      fi
    done
    exit 0
    ;;
esac

LOG="$LOGS/$(date -u +%F_%H%M%S)-$$.log"
{
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  cat UPSTREAM.txt 2>/dev/null || echo "no UPSTREAM.txt"
  echo "python: $(python3 -V 2>&1)"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader || \
    echo "nvidia-smi unavailable"
  df -h "$ROOT" | tail -1
  ensure_dependencies || { echo "FATAL: dependencies unavailable"; exit 3; }
  python3 -m bench.run_sweep --output-dir "$RESULTS" "$@"
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo "EXIT=$status"
echo "log: $LOG"
echo "=== points ==="
python3 -m bench.run_sweep --export-only --output-dir "$RESULTS" 2>/dev/null || true
exit 0
