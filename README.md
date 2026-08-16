# PSPNet MLOps Pipeline

End-to-end MLOps system for semantic segmentation (PSPNet on CamVid).
Training → Serving → Monitoring → Automated Retraining.

---

## Overview

This repo trains a PSPNet segmentation model on the CamVid driving-scene dataset and
runs the whole thing as a pipeline. Data is versioned with DVC, runs are tracked in
MLflow, training executes as a Prefect flow behind a validation gate, and the registered
model ships to Cloud Run through GitHub Actions. I built it to practice the operational
side of ML, so the interesting part is the plumbing around the model rather than the
model itself.

---

## Architecture

```
CamVid Dataset (DVC + GCS)
        ↓
Prefect Flow (load → prepare → train → evaluate → register)
   tracked throughout in MLflow (params, metrics, loss curves)
        ↓
Validation Gate (blocks registry on low mIoU)
        ↓
MLflow Model Registry
        ↓
Champion-Challenger (promotes only if it beats the current champion)
        ↓
GitHub Actions CI/CD (lint → test → build → deploy)
        ↓
FastAPI Serving (Docker) on GCP Cloud Run
        ↓
Prometheus Metrics  →  Grafana dashboard
        ↓
Evidently Drift Detection  →  Auto-Retrain Trigger
```

Each stage runs on its own. `flows/training_flow.py` drives training end to end, `app/`
serves the registered model, and CI/CD connects the two. A push to `main` is what moves
a validated model from the registry to a live endpoint.

---

## Results

| Metric | Value |
|---|---|
| Val mIoU | 0.5597 |
| Model | PSPNet (ResNet-50 + PPM), 100 epochs |
| Model Registry | `pspnet-camvid v1` |
| Training GPU | NVIDIA A40 (RunPod) |

Serving throughput (100 requests, concurrency 10). The full concurrency sweep is in
[Benchmarking](#benchmarking).

| Target | Success | Throughput | p99 Latency |
|---|---|---|---|
| Cloud Run (`--cpu=2`) | 11/100 | 0.71 req/s | 15.69s |
| RunPod GPU (RTX PRO 4500) | 100/100 | 7.45 req/s | 3.19s |

The RunPod GPU pod holds concurrency 10 with zero failures. The current Cloud Run
config does not. Cloud Run's own logs confirm this is an out-of-memory crash loop
(`--memory=2Gi` isn't enough for 10 concurrent PSPNet forward passes), not a
request-queue or admission-control limit. [Benchmarking](#benchmarking) has the full
breakdown.

---

## Stack

| Category | Tool |
|---|---|
| Model | PSPNet (ResNet-50 + PPM) |
| Data Version Control | DVC + GCS |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | Prefect |
| Serving | FastAPI + Docker |
| CI/CD | GitHub Actions |
| Drift Detection | Evidently AI |
| Monitoring | Prometheus + Grafana |
| Deployment | GCP Cloud Run |

---

## Quickstart (RunPod)

Pods launch from a custom image (`runpod/Dockerfile`) that already has the training
environment installed. A `setup.sh` baked into the image handles GCP and GitHub auth and
pulls the repo plus the DVC data, so there is no browser auth flow to run on the pod.
One-time setup (building the image, creating the scoped GCP service account) is in
[`docs/runpod_setup.md`](docs/runpod_setup.md).

**First pod on a given Network Volume.** This part runs once. `secrets.env` and the DVC
cache live on the volume and survive across pods.

1. Launch a pod from the built image with a Network Volume mounted at `/workspace` and
   **Container Disk set to at least 40-50GB**. Too small and the image can silently
   extract only part of itself (see `docs/runpod_setup.md`).
2. SSH in (Pod → Connect → SSH over exposed TCP).
3. Copy `runpod/secrets.env.example` to `secrets.env` locally, fill in your GCP
   service-account key (see `docs/runpod_setup.md`), and upload it to
   `/workspace/secrets.env`. Pasting into `nano secrets.env` works, so does `scp`.
4. Run `setup.sh`.

**Every later pod on the same volume.** SSH in and run `setup.sh` again. It is
idempotent and picks up the credentials and data already on the volume.

### MLflow Server (Terminal A)
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --allowed-hosts "*"
```

> The MLflow UI on RunPod is under Pod → Connect → HTTP 5000.

### Train (Terminal B)
```bash
python src/train.py --config configs/config.yaml
```

---

## Deployment

Every push to `main` makes GitHub Actions build the serving image, push it to Artifact
Registry, and deploy it to Cloud Run (`pspnet-serving`, `us-central1`). Auth goes
through Workload Identity Federation, so GitHub's OIDC token is exchanged for
short-lived GCP credentials and no service-account key sits in the repo. Full setup
(IAM roles, WIF pool and provider, repo variables) is in `docs/dev_note.md`.

**Model promotion.** Passing the validation gate isn't the same as being the best model
available. `flows/training_flow.py`'s `promote_champion_task` runs right after
registration and only promotes a new version to the `"Production"` registry stage if its
`best_val_miou` beats whatever's already there (the first model registered becomes
champion unconditionally). `app/model_loader.py` can serve from that stage by setting
`MODEL_STAGE_OR_VERSION=Production`. The live Cloud Run deployment doesn't do this
today (Cloud Run can't reach the MLflow tracking server, so it falls back to the
checkpoint baked into the image), but any deployment target with MLflow access gets the
gate automatically. Disable via `champion_challenger.enabled: false` in
`configs/config.yaml` if needed.

---

## Dataset

**CamVid-11**, urban driving scene semantic segmentation.
11 classes. Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car,
Pedestrian, Bicyclist.

| Split | Images |
|---|---|
| Train | 367 |
| Val | 101 |
| Test | 233 |

---

## Model

**PSPNet** (Pyramid Scene Parsing Network)

```
Input (3, 201, 201)
  → ResNet-50 backbone (deep base, dilated conv)
  → Pyramid Pooling Module (bins: 1×1, 2×2, 3×3, 6×6)
  → Main Classifier
  → Output (11, 201, 201)
```

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet-50 |
| Epochs | 100 |
| Batch size | 8 |
| Base LR | 0.01 (poly decay) |
| Aux loss weight | 0.4 |
| Crop size | 201 × 201 |

---

## Environment

| Component | Spec |
|---|---|
| Training | RunPod, NVIDIA A40, CUDA 12.8.1 (`runpod/Dockerfile` base image), PyTorch 2.12.0 |
| Serving (Cloud Run) | `--cpu=2 --memory=2Gi`, `us-central1`, Python 3.12-slim, PyTorch 2.12.0 (CPU-only) |
| CI | GitHub Actions `ubuntu-latest`, Python 3.12 |

Ad hoc benchmark runs (`scripts/load_test.py`) record their own client-side `cpu_count`
and `platform` per row in `results/benchmarks.csv`. A single "dev machine" spec listed
here would just go stale.

---

## Benchmarking

`scripts/load_test.py` fires a batch of concurrent requests at `/predict` and reports
measured throughput (req/s) and latency percentiles, appending each run as a row to
`results/benchmarks.csv` (`--tag` labels the target) so results stay comparable across
targets and over time.

```bash
python scripts/load_test.py \
  --url http://localhost:8000/predict \
  --image data/camvid/test/0001TP_008550.png \
  --requests 100 --concurrency 10 --tag local
```

**Throughput saturation on RunPod GPU** (RTX PRO 4000 Blackwell, 24467 MiB VRAM,
2026-08-16). Ran a concurrency ladder (5 through 100) against a live pod with
`nvidia-smi --query-gpu=... -l 1` logging continuously in the background for the whole
session, each run's window sliced out of that trace by timestamp afterward — not a single
manual `nvidia-smi` check, which risks catching the GPU after a burst has already
drained:

| Concurrency | Throughput | GPU util (mean / max) | VRAM used |
|---|---|---|---|
| 5   | 4.37 req/s  | 1.4% / 6%   | 2604 MiB |
| 10  | 7.84 req/s  | 4.5% / 17%  | 2604 MiB |
| 15  | 9.82 req/s  | 6.1% / 24%  | 2604 MiB |
| 20  | 12.85 req/s | 11.4% / 35% | 2604 MiB |
| 25  | 14.64 req/s | 7.6% / 16%  | 2604 MiB |
| 30  | 16.20 req/s | 7.5% / 20%  | 2604 MiB |
| 40  | 14.96 req/s | 9.0% / 22%  | 2604 MiB |
| 60  | 18.16 req/s | 11.1% / 28% | 2604 MiB |
| 80  | 18.75 req/s | 10.7% / 33% | 2604 MiB |
| 100 | 18.28 req/s | 8.6% / 37%  | 2604 MiB |

Throughput rises with diminishing returns and soft-plateaus around concurrency 60-80
(~18-19 req/s). GPU utilization stays low at every point (mean 1.4%-11.4%, max never above
37%), and VRAM sits completely flat at 2604 MiB out of 24467 MiB (~11%) regardless of
load — **the GPU is not the bottleneck** at any concurrency tested here.

A second run with the client co-located on the pod itself (looping back through the proxy
instead of over the public internet — not directly comparable to the table above, but
useful for isolating server-side behavior from network latency) sharpened the same
picture: throughput rose faster (peaking ~52 req/s around concurrency 40-60) and then
**declined** through 80-100, while GPU utilization stayed just as low (mean under 30%,
brief maxes up to 71%) and VRAM stayed pinned at the same 2604 MiB. A curve that rises,
peaks, and then falls while the compute resource underneath it stays underutilized is the
signature of contention, not a hard ceiling — most likely CPU-side thread/GIL contention or
a race around the single shared model instance (`app/model_loader.py`), which is invoked
concurrently via `run_in_threadpool` with no lock, semaphore, or queue around it. Adding an
explicit concurrency guard around inference is the natural next step to confirm and fix
this.

---

## Monitoring

`monitoring/docker-compose.yml` runs the serving app plus a Prometheus + Grafana stack,
fully provisioned (datasource and dashboard load automatically, no manual UI setup):

```bash
docker compose -f monitoring/docker-compose.yml up --build
```

Send some traffic so the dashboard has something to show:
```bash
python scripts/load_test.py --url http://localhost:8000/predict \
  --image data/camvid/test/0001TP_008550.png --requests 50 --concurrency 5 --no-record
```

Then open Grafana at `localhost:3000` (`admin` / `admin`). The "PSPNet Serving"
dashboard is already there, with request rate, error rate, overall request latency
(p50/p95/p99), and inference-only latency (p50/p95/p99, isolating model compute from
request overhead, same distinction as the Benchmarking section above). Prometheus itself
is at `localhost:9090` if you want to query metrics directly; `/targets` should show the
app as `UP`. Needs `checkpoints/best.pth` locally first (`dvc pull`) since the `app`
service builds from `serving/Dockerfile`, same prerequisite as any local run of the
serving image.

---

## Drift Detection

`scripts/check_drift.py` checks whether a batch of images has drifted from the CamVid
training distribution, using Evidently over per-image summary statistics (channel
mean/std, brightness, a sharpness proxy; see `src/drift.py` for why raw pixels aren't
used directly). Reference set is `data/camvid/train.txt`. Current set defaults to the
CamVid test split:

```bash
python scripts/check_drift.py --current-dir data/camvid/test
```

Exit code 0 = drift share at or under `drift_detection.drift_share_threshold` in
`configs/config.yaml`, 1 = blocked. Either way an HTML report is saved to
`results/drift_report_<timestamp>.html`.

**A genuine finding while building this**: the CamVid train/test split itself shows real
drift, every one of the 8 feature columns, drift_share=1.00. It's not a bug: train and
test are different video sequences, and the test split really does run ~8-9/255 darker
on average (checked directly, mean brightness 107.9 on train vs. 99.1 on test). A random
sample drawn from `train.txt` itself passes cleanly (drift_share=0.00) against the full
train set, confirming the detector isn't just always flagging drift. Point `--current-dir`
at that kind of same-distribution sample for a clean PASSED baseline, or at deliberately
corrupted images to demonstrate a clearer drift case.

**Triggering a retrain.** `scripts/trigger_retrain_on_drift.py` runs the same check
and, if blocked, calls Prefect's `run_deployment` API against
`training-flow/pspnet-training` (the deployment `flows/training_flow.py::create_deployment`
registers) rather than waiting for someone to notice. It checks `PREFECT_API_URL` is
reachable first and exits 1 (not a silent no-op) if drift is detected but there's no
worker to hand it to:

```bash
python scripts/trigger_retrain_on_drift.py --current-dir data/camvid/test
```

It fires the trigger and returns immediately. It doesn't wait for the retrain to
finish, since the flow's own validation gate and Champion-Challenger step (above)
already decide whether the retrained model is any good. This script is testable end
to end without a live worker (`tests/test_drift_retrain_integration.py` mocks Prefect
and drift results to check both branches).

**Live RunPod verification.** Ran the full path on a real RunPod pod on 2026-08-15:
a Prefect server, worker, and the `training-flow/pspnet-training` deployment all on
the pod, then fired the trigger against the corrupted images from the drift
demonstration below. The trigger correctly refused to fire when `PREFECT_API_URL`
wasn't set in the shell (failed closed, exit 1, no silent no-op), then fired cleanly
once it was. The worker picked up the flow run and executed `load-data`, `prepare-model`,
and `train-model` end to end (a throwaway 3-epoch config, since the point was
exercising the wiring, not producing a good model), reaching best val mIoU 0.2251.
`register-model` then correctly raised `[Validation Gate BLOCKED] Val mIoU 0.2251 <
threshold 0.55`, and the flow failed cleanly rather than registering or promoting
that model — the gate did its job against a real drift-triggered retrain, not just
in tests.

**Demonstrating the effect.** `scripts/demo_drift_scenario.py` corrupts real CamVid
test images (darkened + blurred, simulating low-light/foggy driving) and measures the
impact directly rather than asserting it:

```bash
python scripts/demo_drift_scenario.py --num-images 30
```

Real run on the checkpoint in this repo: **mIoU dropped from 0.4341 to 0.0483** on the
corrupted batch (delta -0.3858), and the drift check correctly flagged the corrupted
directory as BLOCKED (drift_share=1.00). Corrupted images and a drift report are left
under `results/drift_demo/` for inspection.

---

## Project Structure

```
PSPNet_mlops/
├── configs/
│   └── config.yaml        # All hyperparameters
├── data/
│   ├── camvid/            # CamVid dataset (DVC managed)
│   └── camvid.dvc         # DVC pointer
├── flows/
│   └── training_flow.py   # Prefect training pipeline
├── src/
│   ├── models/
│   │   ├── resnet.py      # ResNet-50 backbone (deep base + dilated)
│   │   └── pspnet.py      # PSPNet + PPM
│   ├── data/
│   │   └── dataset.py     # CamVid dataloader
│   ├── train.py           # Training loop + MLflow tracking
│   └── evaluate.py        # mIoU evaluation
└── checkpoints/           # Saved model weights
```

---

## Roadmap

**Shipped.** Model and training pipeline, DVC-versioned data, MLflow tracking and model
registry, Prefect orchestration with a validation gate, FastAPI serving deployed to
Cloud Run via CI/CD, Prometheus metrics, the load-testing tooling, a provisioned
Grafana/Prometheus dashboard, Evidently drift detection with a threshold gate,
Champion-Challenger promotion, a drift-triggered retraining hook verified end to end
against a live RunPod Prefect worker, a real corrupted-image drift demonstration
(measured, not asserted), an automated test covering the local half of drift to retrain
to evaluate to register/promote, and a live concurrency sweep on the RunPod GPU pod
(pushed from 20 up to 3000 concurrent requests) that found a soft latency ceiling around
40 requests, a rising failure rate well past that, and one unreproduced server crash
pointing to a real concurrency-safety gap in the serving code. Details are in the
sections above.