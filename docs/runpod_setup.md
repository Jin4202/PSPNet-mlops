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
docker build -f runpod/Dockerfile -t <registry>/pspnet-runpod:latest .
docker push <registry>/pspnet-runpod:latest
```

Push to a **private** registry (or a private Docker Hub repo). The image itself never
contains secrets by design — `requirements.txt` is the only thing baked in — but keeping
it private avoids exposing the project's setup unnecessarily.

## Per-pod flow

**First pod on a given volume (one-time):**
1. Launch a pod from `<registry>/pspnet-runpod:latest`, with a Network Volume mounted at `/workspace`.
2. SSH in (Pod → Connect → SSH over exposed TCP).
3. Upload `secrets.env` to `/workspace/secrets.env` (paste into `nano secrets.env`, or `scp`).
4. Run `setup.sh`.

**Every later pod start on the same volume:**
1. SSH in.
2. Run `setup.sh` again — idempotent: `secrets.env` and the DVC cache are already on the
   volume, so it just re-activates GCP auth and does an incremental `git pull`/`dvc pull`.

`setup.sh` prints the MLflow server and training commands (matching `README.md`'s
Quickstart) once it finishes.
