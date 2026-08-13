# RunPod Training Setup

## Goal
Get a RunPod pod running training with minimal manual work: a custom image with the
Python environment pre-installed, plus a `setup.sh` that handles GCP/GitHub auth and
pulls the repo + DVC data — no browser-based auth flow needed on the pod itself.

## Assumption: `/workspace` is a persistent Network Volume
This flow assumes the pod's `/workspace` is backed by a RunPod **Network Volume**, not
ephemeral container storage. That's what makes `secrets.env` upload and GCP auth
effectively **one-time** (they survive pod stop/restart on the same volume) rather than
a per-pod chore, and what makes `dvc pull` incremental instead of a full re-download
every time. If you ever launch a pod without a volume attached, every step below reverts
to "from scratch" on that pod — expected, not a bug.

## One-time: create a scoped GCP service account

Uses a dedicated `runpod-training` service account, scoped to just the DVC bucket
(`roles/storage.objectAdmin` on that bucket only, not project-wide) — narrower than the
CI deploy service account documented in `dev_note.md`, and easy to revoke independently
if the key ever leaks.

```bash
PROJECT_ID=bamboo-climate-497404-a7
SA_EMAIL=runpod-training@${PROJECT_ID}.iam.gserviceaccount.com

gcloud iam service-accounts create runpod-training \
  --project="${PROJECT_ID}" \
  --display-name="RunPod training (DVC access)"

gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin" \
  gs://pspnet-mlops-dvc

# Base64-encode the key onto one line for GCP_SA_KEY_B64 in secrets.env
gcloud iam service-accounts keys create - --iam-account="${SA_EMAIL}" \
  | base64 | tr -d '\n' > sa-key-b64.txt
```

Paste the contents of `sa-key-b64.txt` into `GCP_SA_KEY_B64` in your local copy of
`runpod/secrets.env.example` (saved as `secrets.env`), then delete `sa-key-b64.txt`
locally — it's the raw key, don't leave it lying around.

**Rotation**: if the key ever leaks, revoke it immediately and issue a new one:
```bash
gcloud iam service-accounts keys list --iam-account="${SA_EMAIL}"
gcloud iam service-accounts keys delete <KEY_ID> --iam-account="${SA_EMAIL}"
```

## Building and pushing the image

Only needed when `requirements.txt` changes — not for ordinary code changes, which
`setup.sh` pulls fresh at pod start instead.

```bash
docker build -f runpod/Dockerfile -t jinseokheo/pspnet-runpod:latest .
docker push jinseokheo/pspnet-runpod:latest
```

Make `jinseokheo/pspnet-runpod` a **private** Docker Hub repo. The image itself never
contains secrets by design — `requirements.txt` is the only thing baked in — but keeping
it private avoids exposing the project's setup unnecessarily.

## Per-pod flow

**First pod on a given volume (one-time):**
1. Launch a pod from `jinseokheo/pspnet-runpod:latest`, with a Network Volume mounted at
   `/workspace`. Set **Container Disk to at least 40-50GB** — the image itself is ~13GB,
   and a too-small Container Disk can cause the image to only partially extract, silently
   missing files (this happened: `gcloud`/`gh`/`dvc`/`mlflow` all appeared "not found" on
   a pod with a 20GB disk, despite being genuinely present in the image — confirmed via a
   local `docker run` of the exact same image). Container Disk is separate from the
   Network Volume: it's the pod's own ephemeral root filesystem that has to hold and run
   the image, not the persistent `/workspace` mount.
2. SSH in (Pod → Connect → SSH over exposed TCP).
3. Upload `secrets.env` to `/workspace/secrets.env` (paste into `nano secrets.env`, or `scp`).
4. Run `setup.sh`.

**Every later pod start on the same volume:**
1. SSH in.
2. Run `setup.sh` again — idempotent: `secrets.env` and the DVC cache are already on the
   volume, so it just re-activates GCP auth and does an incremental `git pull`/`dvc pull`.

`setup.sh` prints the MLflow server and training commands (matching `README.md`'s
Quickstart) once it finishes.

## Running the serving API on a pod (for benchmarking)

The pod normally only runs training — the FastAPI app has never been started there.
`fastapi`/`uvicorn` are in `requirements.txt`, which `runpod/Dockerfile` installs, so in
principle no image change is needed to also serve from the pod, e.g. to benchmark GPU
inference against the Cloud Run deployment with `scripts/load_test.py`.

**In practice, as of 2026-08-13 the currently-pushed `jinseokheo/pspnet-runpod:latest`
image is stale** relative to `requirements.txt` — it predates `fastapi`/`uvicorn` being
added and has `torch==2.4.1+cu124` baked in instead of the pinned `2.12.0`. That old torch
build threw `CUDA error: no kernel image is available for execution on the device` on
every request when tested against an RTX PRO 4500 pod (a newer GPU architecture than
`2.4.1+cu124` supports) — not a resource/concurrency problem, every request failed
identically. **Fix this properly by rebuilding and repushing the image** (see "Building
and pushing the image" above) so future pods don't need manual patching. Until then, patch
a running pod manually:
```bash
python3 -m pip install --ignore-installed blinker \
  fastapi uvicorn python-multipart pillow numpy PyYAML mlflow prometheus-fastapi-instrumentator
python3 -m pip install torch==2.12.0 torchvision==0.27.0
```
(`--ignore-installed blinker` works around the same distutils-uninstall conflict as the
image build itself; the torch reinstall pulls in kernel support for newer GPU
architectures — confirm with `nvidia-smi --query-gpu=name,compute_cap --format=csv` if a
future pod hits the same CUDA error on a different GPU model.)

1. On the pod (after `setup.sh` has run at least once, so `checkpoints/best.pth` exists
   from training):
   ```bash
   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   (`-m uvicorn` avoids relying on the script being on `PATH`.) Serves from
   `checkpoints/best.pth` by default (`CHECKPOINT_PATH` in `app/model_loader.py`). Set
   `MLFLOW_TRACKING_URI=http://localhost:5000` first to serve from the pod's own MLflow
   registry instead, if one is running.
2. Expose port 8000 externally — same mechanism as MLflow (5000) and Jupyter (8888): Pod
   → Edit Pod → add HTTP Port 8000. This may require a pod restart if the port list isn't
   live-editable; the Network Volume, `secrets.env`, and DVC cache all persist across a
   restart on the same volume.
3. RunPod gives an HTTPS proxy URL for the exposed port (Connect panel), pattern
   `https://<POD_ID>-8000.proxy.runpod.net`. Verify with `curl <url>/health` before
   benchmarking.
