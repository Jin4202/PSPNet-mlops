# PSPNet MLOps Pipeline

End-to-end MLOps system for semantic segmentation (PSPNet/CamVid).  
Training → Serving → Monitoring → Automated Retraining.

---

## Progress

### Model
- [x] `configs/config.yaml` — centralized hyperparameter management
- [x] `src/models/resnet.py` — ResNet-50 (Deep Base + Dilated Conv)
- [x] `src/models/pspnet.py` — PSPNet + PPM
- [x] `src/data/dataset.py` — CamVid dataloader + augmentation pipeline
- [x] `src/evaluate.py` — mIoU evaluation
- [x] `src/train.py` — training loop
- [x] Baseline training complete (100 epochs, RunPod A40)
- [x] **Best Val mIoU: 0.5597**

### Data Version Control
- [x] DVC init + GCS remote connected (`gs://pspnet-mlops-dvc`)
- [x] CamVid dataset `dvc push` (1,406 files) + `dvc pull` reproducibility verified

### Experiment Tracking
- [x] MLflow instrumentation in `src/train.py` (hyperparameters, mIoU, loss curves)
- [x] MLflow Model Registry registered (`pspnet-camvid v1`)

### Pipeline Orchestration
- [x] Prefect install + local server running
- [x] `flows/training_flow.py` — Prefect Flow with 5 tasks
  - [x] `load_data_task`
  - [x] `prepare_model_task`
  - [x] `train_model_task`
  - [x] `evaluate_model_task`
  - [x] `register_model_task`
- [x] Validation Gate — block Model Registry registration when mIoU is below threshold
- [x] `configs/config.yaml` refactor (add `validation_gate` section)
- [x] Prefect Deployment created (`prefect deploy`)
- [x] Single-command execution verified: `prefect deployment run training-flow/pspnet-training`
- [x] Flow run history visible in Prefect UI

### Serving
- [x] FastAPI inference endpoint (`/predict`: image upload → segmentation mask)
- [x] Multi-stage Docker build (minimized image size)
- [x] Inference time included in API response
- [x] Unit tests (preprocessing, API endpoint)
- [x] Fix `/predict` blocking the event loop — inference now runs via `run_in_threadpool`; see "Benchmarking" section for the remaining CPU-contention caveat

### CI/CD & Deployment
- [x] GitHub repo setup + project structure
- [x] GitHub Actions workflow — push → lint/test → Docker build → Artifact Registry push → Cloud Run deploy
- [x] Model validation gate in CI/CD pipeline
- [x] Cloud Run deploy job wired up via Workload Identity Federation (no static keys)
- [x] GCP Cloud Run deployment verified live
- [ ] Cloud deployment stability verified

### Monitoring
- [x] Prometheus metrics exposed (`GET /metrics` — request volume/latency/error rate via `prometheus-fastapi-instrumentator`, plus a custom `pspnet_inference_duration_seconds` histogram isolating model inference time)
- [ ] Grafana dashboard configured

### Drift Detection & Auto-Retraining
- [ ] Evidently AI — input distribution drift monitoring + threshold definition
- [ ] Prefect Flow auto-triggered when drift threshold is exceeded
- [ ] Champion-Challenger — retrained model promoted to production only when it outperforms incumbent
- [ ] Drift demonstration scenario (OOD images → measurable mIoU degradation)
- [ ] End-to-end integration test (drift → retraining → evaluation → deployment, fully automated)

### Portfolio / Documentation
- [ ] Architecture diagram created + added to top of README
- [ ] README finalized (problem → architecture → components → result metrics)
- [ ] Demo video recorded (3–5 min: normal inference → drift event → automated retraining)
- [ ] Resume bullet written + technical decision retrospective documented

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

Pods launch from a custom image (`runpod/Dockerfile`) with the training environment
pre-installed; a `setup.sh` baked into that image handles GCP/GitHub auth and pulls the
repo + DVC data — no browser-based auth flow needed on the pod. Full one-time setup
(building the image, creating the scoped GCP service account) is in
[`docs/runpod_setup.md`](docs/runpod_setup.md).

**First pod on a given Network Volume** (one-time — `secrets.env` and the DVC cache
persist on the volume across future pods, so this isn't a repeat-every-time step):
1. Launch a pod from the built image, with a Network Volume mounted at `/workspace` and
   **Container Disk set to at least 40-50GB** (too small and the image can silently only
   partially extract — see `docs/runpod_setup.md`).
2. SSH in (Pod → Connect → SSH over exposed TCP).
3. Copy `runpod/secrets.env.example` to `secrets.env` locally, fill in your GCP
   service-account key (see `docs/runpod_setup.md`), and upload it to
   `/workspace/secrets.env` (paste into `nano secrets.env`, or `scp`).
4. Run `setup.sh`.

**Every later pod on the same volume:** SSH in, run `setup.sh` again — it's idempotent
and picks up the credentials/data already on the volume.

### MLflow Server (Terminal A)
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --allowed-hosts "*"
```

> MLflow UI on RunPod: Pod → Connect → HTTP 5000

### Train (Terminal B)
```bash
python src/train.py --config configs/config.yaml
```

---

## Deployment

On every push to `main`, GitHub Actions builds the serving image, pushes it to Artifact Registry, and deploys it to Cloud Run (`pspnet-serving`, `us-central1`). Authentication uses Workload Identity Federation — GitHub's OIDC token is exchanged for short-lived GCP credentials, so no service-account key is stored in the repo. See `docs/dev_note.md` for the full setup (IAM roles, WIF pool/provider, repo variables).

---

## Environment

| Component | Spec |
|---|---|
| Training | RunPod, NVIDIA A40, CUDA 12.8.1 (`runpod/Dockerfile` base image), PyTorch 2.12.0 |
| Serving (Cloud Run) | `--cpu=2 --memory=2Gi`, `us-central1`, Python 3.12-slim, PyTorch 2.12.0 (CPU-only) |
| CI | GitHub Actions `ubuntu-latest`, Python 3.12 |

Ad hoc benchmark runs (`scripts/load_test.py`) record their own client-side `cpu_count`/
`platform` per row in `results/benchmarks.csv` rather than duplicating a single "dev
machine" spec here, since that would go stale.

---

## Benchmarking

`scripts/load_test.py` fires a pack of concurrent requests at `/predict` and reports
measured throughput (req/s) and latency percentiles. Works against a local server, a
live Cloud Run URL, or an API started on a RunPod pod (see `docs/runpod_setup.md`).
Every run is also appended as a row to `results/benchmarks.csv` (`--tag` labels which
target it hit), so results stay comparable across targets and over time:

```bash
python scripts/load_test.py \
  --url http://localhost:8000/predict \
  --image data/camvid/test/0001TP_008550.png \
  --requests 100 --concurrency 10 --tag local
```

**Concurrency**: `/predict` now runs inference via `run_in_threadpool` instead of blocking
the event loop inline, so the server can keep handling other requests (e.g. `/health`)
while inference is in flight. That said, on this machine (11 CPU cores) it barely moved
raw `/predict` throughput under load — measured ~5 req/s at both concurrency 1 and 12
before *and* after the fix, with p99 latency still going from ~0.2s to ~2.4s. Root cause:
PyTorch defaults to 5 intra-op threads per inference call, so concurrent requests mostly
contend for the same cores rather than truly parallelizing — moving work off the event
loop doesn't help if the CPU itself is the bottleneck. Constraining PyTorch to fewer
threads per request (e.g. `OMP_NUM_THREADS=1`) measured a real ~14% throughput gain at
concurrency 12, but at the cost of ~29% worse single-request latency — a genuine
latency-vs-throughput tradeoff, deliberately not applied here since the right setting
depends on the deployment's actual CPU allocation (Cloud Run currently runs `--cpu=2`,
very different oversubscription math than this 11-core dev machine) and wasn't tuned for
it. Revisit if concurrent throughput becomes a real bottleneck in practice.

**Live Cloud Run results** (100 requests, `--cpu=2 --memory=2Gi`, same test image each
run — full rows in `results/benchmarks.csv`):

| Concurrency | Success | Throughput | Latency (median / p99 / max) |
|---|---|---|---|
| 1 | 100/100 | 0.62 req/s | 1.59s / 1.92s / 1.92s |
| 2 | 100/100 | 0.83 req/s | 1.67s / 23.92s / 23.92s |
| 10 | 11/100 | 0.71 req/s | 14.10s / 15.69s / 15.69s (89 requests failed with HTTP 503) |

At concurrency 1 the deployment is clean and predictable. At concurrency 2, throughput
barely improves but one request spikes to ~24s — consistent with Cloud Run cold-starting
a second instance mid-run (full container boot + model load) rather than CPU contention
within a single instance. At concurrency 10 the single `--cpu=2` instance can't keep up
and Cloud Run starts shedding load with 503s — this deployment's real concurrency
ceiling is well below 10, not the ~80 req/instance Cloud Run defaults to. Not yet root
caused against actual Cloud Run logs/metrics; revisit before relying on this deployment
under real concurrent load.

**Live RunPod GPU results** (100 requests, RTX PRO 4500, same test image each run — full
rows in `results/benchmarks.csv`):

| Concurrency | Success | Throughput | Latency (median / p99 / max) |
|---|---|---|---|
| 1 | 100/100 | 0.85 req/s | 0.97s / 2.65s / 2.65s |
| 2 | 100/100 | 2.04 req/s | 0.85s / 2.20s / 2.20s |
| 10 | 100/100 | 7.45 req/s | 1.16s / 3.19s / 3.19s |

Zero failures at any concurrency level — no cold-start cliff, no dropped requests,
unlike Cloud Run. Throughput scales cleanly with concurrency (0.85 → 2.04 → 7.45 req/s),
consistent with one dedicated GPU handling more in-flight requests without contention.
One oddity: latency is **not** monotonic with concurrency — concurrency 1's median/p99
(0.97s / 2.65s) is actually worse than concurrency 2's (0.85s / 2.20s). Single-request
inference itself measured ~334ms (`inference_time_ms` from a direct `/predict` call), so
most of the latency in every row above is request/network overhead through RunPod's proxy
tunnel, not GPU compute — likely explains the non-monotonic latency (proxy-path jitter
dominating over true concurrency effects at this scale), though not confirmed against
server-side logs. At concurrency 10, GPU inference is roughly **10x Cloud Run's
throughput** at the same concurrency (7.45 vs. 0.71 req/s). Not a fully apples-to-apples
comparison (dedicated GPU pod vs. a `--cpu=2` autoscaled container), but it quantifies the
gap between the two current serving options.

*(Getting a clean run required patching the pod's environment on the fly: the RunPod
image's baked-in `torch==2.4.1+cu124` predates this GPU's architecture and threw `CUDA
error: no kernel image is available for execution on the device` on every request; a
manual `pip install torch==2.12.0 torchvision==0.27.0` — matching `requirements.txt` —
fixed it. `fastapi`/`uvicorn` were also missing entirely, since the serving app has
never run on this image before. Both trace back to the same cause: `runpod/Dockerfile`
hasn't been rebuilt since `requirements.txt` picked up these deps/versions. Worth a
rebuild + repush so future pods don't need this manual patching — see
`docs/runpod_setup.md`.)*

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

## Dataset

**CamVid-11** — Urban driving scene semantic segmentation  
11 classes: Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist

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

## Results

| Metric | Value |
|---|---|
| Val mIoU | 0.5597 |
| Model Registry | pspnet-camvid v1 |
| GPU | NVIDIA A40 (RunPod) |
