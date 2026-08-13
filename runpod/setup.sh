#!/usr/bin/env bash
# Authenticates GCP + GitHub, then clones/pulls the repo and its DVC data.
#
# Usage: setup.sh [path-to-secrets.env]   (defaults to /workspace/secrets.env)
#
# Safe to re-run: on a persistent RunPod Network Volume, the first run does
# the real setup (clone, decode creds); every later run on the same volume
# just re-activates GCP auth and does an incremental git/dvc pull.
set -euo pipefail

SECRETS_FILE="${1:-/workspace/secrets.env}"
WORKSPACE_DIR="/workspace"
REPO_DIR="${WORKSPACE_DIR}/PSPNet_mlops"
REPO_URL="https://github.com/Jin4202/PSPNet-mlops.git"
GCP_KEY_PATH="${HOME}/.config/gcloud/runpod-sa-key.json"

log() { echo "[setup] $*"; }

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "[setup] ERROR: secrets file not found at ${SECRETS_FILE}" >&2
  echo "[setup] Copy runpod/secrets.env.example, fill it in, and upload it there first." >&2
  exit 1
fi

chmod 600 "${SECRETS_FILE}"
set -a
# shellcheck disable=SC1090
source "${SECRETS_FILE}"
set +a

# ── GCP auth ────────────────────────────────────────────────────────────
if [[ -z "${GCP_SA_KEY_B64:-}" ]]; then
  echo "[setup] ERROR: GCP_SA_KEY_B64 is not set in ${SECRETS_FILE}" >&2
  exit 1
fi

log "Decoding GCP service-account key ..."
mkdir -p "$(dirname "${GCP_KEY_PATH}")"
echo "${GCP_SA_KEY_B64}" | base64 -d > "${GCP_KEY_PATH}"
chmod 600 "${GCP_KEY_PATH}"

export GOOGLE_APPLICATION_CREDENTIALS="${GCP_KEY_PATH}"
if ! grep -q "GOOGLE_APPLICATION_CREDENTIALS=" "${HOME}/.bashrc" 2>/dev/null; then
  echo "export GOOGLE_APPLICATION_CREDENTIALS=\"${GCP_KEY_PATH}\"" >> "${HOME}/.bashrc"
fi

gcloud auth activate-service-account --key-file="${GCP_KEY_PATH}"
log "GCP auth OK ($(gcloud config get-value account 2>/dev/null))."
log "Note: GOOGLE_APPLICATION_CREDENTIALS is only exported in this script's process."
log "  A NEW shell (e.g. a fresh SSH session) picks it up automatically via ~/.bashrc."
log "  In THIS shell, if you run dvc/gcloud commands manually after this script exits, run:"
log "    export GOOGLE_APPLICATION_CREDENTIALS=\"${GCP_KEY_PATH}\""

# ── GitHub auth (optional — repo is public, only needed for push / gh CLI) ──
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  log "Authenticating GitHub CLI ..."
  echo "${GITHUB_TOKEN}" | gh auth login --with-token
  gh auth setup-git
  log "GitHub auth OK."
else
  log "GITHUB_TOKEN not set — skipping GitHub auth (fine for clone/pull, needed for push)."
fi

# ── Clone or pull the repo ──────────────────────────────────────────────
if [[ -d "${REPO_DIR}/.git" ]]; then
  log "Repo already present at ${REPO_DIR} — pulling latest ..."
  git -C "${REPO_DIR}" pull
else
  log "Cloning repo into ${REPO_DIR} ..."
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

# ── DVC pull (data + checkpoint) ────────────────────────────────────────
log "Pulling DVC-tracked data ..."
(cd "${REPO_DIR}" && dvc pull)

log "Done."
cat <<EOF

Next steps:
  cd ${REPO_DIR}

  # Terminal A — MLflow server
  mlflow server --host 0.0.0.0 --port 5000 \\
    --backend-store-uri sqlite:///mlflow.db \\
    --default-artifact-root ./mlartifacts --allowed-hosts "*"

  # Terminal B — train
  python src/train.py --config configs/config.yaml
EOF
