#!/usr/bin/env bash
set -Eeuo pipefail

# Idempotent global setup for the PB-CWM Hopper campaign.
# This deliberately installs only pbcwm/requirements.txt.  The repository
# root requirements.txt belonged to the removed legacy HyperCRL codebase.

ROOT="${PB_CWM_ROOT:-/workspace/HyperCRL}"
STAMP="${PB_CWM_ENV_STAMP:-/workspace/.pbcwm_hopper_cuda_env_v1}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "python3 is required" >&2
  exit 1
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "A virtual environment is active; unset VIRTUAL_ENV and rerun for the global install." >&2
  exit 1
fi

if [[ -f "${STAMP}" ]]; then
  echo "PB-CWM global environment already verified: ${STAMP}"
  "${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "the recorded global environment no longer has CUDA"
x = torch.ones((32, 32), device="cuda")
_ = x @ x
torch.cuda.synchronize()
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing; stop before installing a CPU torch by mistake" >&2
  exit 1
fi
nvidia-smi

UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" && -x /workspace/.local/bin/uv ]]; then
  UV_BIN=/workspace/.local/bin/uv
fi
if [[ -z "${UV_BIN}" ]]; then
  UV_DIR="${UV_DIR:-/workspace/.local/bin}"
  mkdir -p "${UV_DIR}"
  curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="${UV_DIR}/uv" sh
  UV_BIN="${UV_DIR}/uv"
fi
"${UV_BIN}" --version

cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import sys
assert sys.version_info >= (3, 11), sys.version
print(sys.version)
PY

# One resolver/install transaction.  --system means no virtualenv.  The
# automatic backend selects a CUDA PyTorch wheel from the detected driver.
"${UV_BIN}" pip install --system --python "${PYTHON_BIN}" \
  --torch-backend=auto \
  -r pbcwm/requirements.txt

"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
import torch
import gymnasium
import gpytorch
import yaml

assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.version.cuda is not None, "torch has no CUDA runtime"
x = torch.ones((256, 256), device="cuda")
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gymnasium": gymnasium.__version__,
    "gpytorch": gpytorch.__version__,
    "yaml": yaml.__version__,
}, sort_keys=True))

from pbcwm.protocol.config import load_protocol_config
protocol = load_protocol_config("pbcwm/configs/protocol_hopper_screen_v1.yaml")
assert protocol.world_model.update_interval_steps == 4
assert protocol.environment("Hopper-v5").planner_population == 64
assert protocol.evaluation.planning_episodes_stage_end == 3
PY

mkdir -p "$(dirname "${STAMP}")"
printf 'verified_at=%s\npython=%s\ntorch=%s\ntorch_cuda=%s\ngpu=%s\n' \
  "$(date -Is)" \
  "${PYTHON_BIN}" \
  "$(${PYTHON_BIN} -c 'import torch; print(torch.__version__)')" \
  "$(${PYTHON_BIN} -c 'import torch; print(torch.version.cuda)')" \
  "$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.get_device_name(0))')" \
  > "${STAMP}"
echo "PB-CWM global environment installed and verified once: ${STAMP}"
