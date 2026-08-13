# PSPNet MLOps Pipeline

End-to-end MLOps system for semantic segmentation (PSPNet/CamVid).  
Training → Serving → Monitoring → Automated Retraining.

---

## Progress

### Week 1 — Foundation: Experiment Tracking + Version Control ✅
- [x] GitHub repo setup + project structure
- [x] `configs/config.yaml` — centralized hyperparameter management
- [x] `src/models/resnet.py` — ResNet-50 (Deep Base + Dilated Conv)
- [x] `src/models/pspnet.py` — PSPNet + PPM
- [x] `src/data/dataset.py` — CamVid dataloader + augmentation pipeline
- [x] `src/evaluate.py` — mIoU evaluation
- [x] `src/train.py` — training loop + MLflow instrumentation (hyperparameters, mIoU, loss curves)
- [x] DVC init + GCS remote connected (`gs://pspnet-mlops-dvc`)
- [x] CamVid dataset `dvc push` (1,406 files) + `dvc pull` reproducibility verified
- [x] Baseline training complete (100 epochs, RunPod A40)
- [x] MLflow Model Registry registered (`pspnet-camvid v1`)
- [x] **Best Val mIoU: 0.5597**

### Week 2 — Training Pipeline Automation 🔄
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

### Week 3 — Serving API + CI/CD ⬜
- [x] FastAPI inference endpoint (`/predict`: image upload → segmentation mask)
- [x] Multi-stage Docker build (minimized image size)
- [x] Inference time included in API response
- [x] GitHub Actions workflow — push → lint/test → Docker build → Artifact Registry push → Cloud Run deploy
- [x] Unit tests (preprocessing, API endpoint)
- [x] Model validation gate in CI/CD pipeline
- [x] Cloud Run deploy job wired up via Workload Identity Federation (no static keys)
- [x] GCP Cloud Run deployment verified live

### Week 4 — Monitoring + Drift Detection + Auto-Retraining ⬜
- [ ] Prometheus metrics exposed (request volume, latency, error rate)
- [ ] Grafana dashboard configured
- [ ] Evidently AI — input distribution drift monitoring + threshold definition
- [ ] Prefect Flow auto-triggered when drift threshold is exceeded
- [ ] Champion-Challenger — retrained model promoted to production only when it outperforms incumbent
- [ ] Drift demonstration scenario (OOD images → measurable mIoU degradation)
- [ ] End-to-end integration test (drift → retraining → evaluation → deployment, fully automated)

### Week 5 — Portfolio Finalization ⬜
- [ ] Cloud deployment stability verified
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
1. Launch a pod from the built image, with a Network Volume mounted at `/workspace`.
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

## Project Structure

```
PSPNet_mlops/
├── configs/
│   └── config.yaml        # All hyperparameters
├── data/
│   ├── camvid/            # CamVid dataset (DVC managed)
│   └── camvid.dvc         # DVC pointer
├── flows/
│   └── training_flow.py   # Prefect training pipeline (Week 2+)
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
