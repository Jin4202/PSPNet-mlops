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
GitHub Actions CI/CD (lint → test → build → deploy)
        ↓
FastAPI Serving (Docker) on GCP Cloud Run
        ↓
Prometheus Metrics  →  Grafana dashboard (planned)
        ↓
Evidently Drift Detection  →  Auto-Retrain Trigger (planned)
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
measured throughput (req/s) and latency percentiles. It works against a local server, a
live Cloud Run URL, or an API started on a RunPod pod (see `docs/runpod_setup.md`).
Every run is appended as a row to `results/benchmarks.csv`, and `--tag` labels which
target it hit, so results stay comparable across targets and over time.

```bash
python scripts/load_test.py \
  --url http://localhost:8000/predict \
  --image data/camvid/test/0001TP_008550.png \
  --requests 100 --concurrency 10 --tag local
```

**Concurrency.** `/predict` runs inference through `run_in_threadpool` instead of
blocking the event loop inline, so the server can keep answering other requests like
`/health` while inference is in flight. On my dev machine (11 CPU cores) that barely
moved raw `/predict` throughput under load. I measured about 5 req/s at both concurrency
1 and 12, before and after the change, with p99 latency still going from ~0.2s to ~2.4s.
The cause is PyTorch defaulting to 5 intra-op threads per inference call, so concurrent
requests mostly contend for the same cores instead of running in parallel. Getting work
off the event loop does not help when the CPU is the bottleneck. Capping threads per
request with `OMP_NUM_THREADS=1` measured about 14% more throughput at concurrency 12
and about 29% worse single-request latency, a straight latency-versus-throughput
tradeoff. I left it off, since the right setting depends on how much CPU the deployment
actually has, and Cloud Run at `--cpu=2` has very different oversubscription math than
an 11-core dev machine. Worth revisiting if concurrent throughput becomes a real
bottleneck.

**Live Cloud Run results.** 100 requests, `--cpu=2 --memory=2Gi`, same test image each
run. Full rows are in `results/benchmarks.csv`.

| Concurrency | Success | Throughput | Latency (median / p99 / max) |
|---|---|---|---|
| 1 | 100/100 | 0.62 req/s | 1.59s / 1.92s / 1.92s |
| 2 | 100/100 | 0.83 req/s | 1.67s / 23.92s / 23.92s |
| 10 | 11/100 | 0.71 req/s | 14.10s / 15.69s / 15.69s (89 requests failed with HTTP 503) |

Concurrency 1 is clean and predictable. At concurrency 2, throughput barely improves and
one request spikes to about 24s. Cloud Run's own logs confirm this is a genuine cold
start, not CPU contention: `Starting new instance. Reason: AUTOSCALING` fires right
after that request, and the service actually churns through 3 instances over the course
of the 100-request run despite never running more than 2 requests at once.

At concurrency 10, the root cause is confirmed directly from Cloud Run's logs:
**out-of-memory crash-looping**, not simple request-queue overload. The logs show
explicit `Memory limit of 2048 MiB exceeded` errors, each followed by `the container
instance was found to be using too much memory and was terminated`, then
`Starting new instance`. Over the ~2.5 minutes this run took, that cycle repeated
**9 times**: 9 OOM kills and 9 fresh cold-started instances, each one immediately hit by
more of the still-queued concurrent requests before it could stabilize.
`containerConcurrency` is 80 and `maxScale` is 20, so Cloud Run's own admission control
was never the bottleneck; a full ResNet-50/PSPNet forward pass under 10-way concurrency
simply doesn't fit in `--memory=2Gi`. The fix is straightforward: raise the memory limit
(`gcloud run services update pspnet-serving --memory=4Gi` or higher) and re-run this
benchmark, just not yet done here.

**Live RunPod GPU results.** 100 requests, RTX PRO 4500, same test image each run. Full
rows are in `results/benchmarks.csv`.

| Concurrency | Success | Throughput | Latency (median / p99 / max) |
|---|---|---|---|
| 1 | 100/100 | 0.85 req/s | 0.97s / 2.65s / 2.65s |
| 2 | 100/100 | 2.04 req/s | 0.85s / 2.20s / 2.20s |
| 10 | 100/100 | 7.45 req/s | 1.16s / 3.19s / 3.19s |

Zero failures at any concurrency level, with no cold-start cliff and no dropped
requests, unlike Cloud Run. Throughput scales cleanly (0.85 → 2.04 → 7.45 req/s), which
fits one dedicated GPU taking on more in-flight requests without contention. One oddity
is that latency is not monotonic with concurrency. Concurrency 1 (0.97s median, 2.65s
p99) is worse than concurrency 2 (0.85s, 2.20s). A direct `/predict` call measured
single-request inference at about 334ms, so most of the latency in every row above is
request and network overhead through the RunPod proxy tunnel rather than GPU compute.
That probably explains the ordering, with proxy jitter dominating over real concurrency
effects at this scale, though I have not confirmed it against server-side logs. At
concurrency 10, GPU inference does roughly **10x the throughput of Cloud Run** at the
same concurrency (7.45 vs 0.71 req/s). The two are not directly comparable, since one is
a dedicated GPU pod and the other an autoscaled `--cpu=2` container, but it does put a
number on the gap between the two serving options I have now.

*(Getting a clean run took some manual patching on the pod. The RunPod image's baked-in
`torch==2.4.1+cu124` predates this GPU's architecture and threw `CUDA error: no kernel
image is available for execution on the device` on every request, so I installed
`torch==2.12.0` and `torchvision==0.27.0` to match `requirements.txt`. `fastapi` and
`uvicorn` were missing entirely, since the serving app had never run on this image
before. Both trace back to the same cause. `runpod/Dockerfile` has not been rebuilt
since `requirements.txt` picked up those deps and versions. A rebuild and repush would
save future pods from the same patching, see `docs/runpod_setup.md`.)*

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
Cloud Run via CI/CD, Prometheus metrics, and the load-testing tooling above. Details are
in the sections above.

**Next.**
- [ ] Grafana dashboard for the exposed Prometheus metrics
- [ ] Evidently AI drift detection plus threshold-triggered retraining
- [ ] Champion-Challenger, so a retrained model is promoted only when it beats the incumbent
- [ ] Drift demonstration scenario (OOD images → measurable mIoU degradation)
- [ ] End-to-end automated test (drift → retrain → evaluate → deploy)
- [ ] Architecture diagram, demo video, resume writeup