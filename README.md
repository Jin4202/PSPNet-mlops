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
- [ ] Prefect install + local server running
- [ ] `flows/training_flow.py` — Prefect Flow with 5 tasks
  - [ ] `load_data_task`
  - [ ] `prepare_model_task`
  - [ ] `train_model_task`
  - [ ] `evaluate_model_task`
  - [ ] `register_model_task`
- [ ] Validation Gate — block Model Registry registration when mIoU is below threshold
- [ ] `configs/config.yaml` refactor (add `validation_gate` section)
- [ ] Prefect Deployment created (`prefect deploy`)
- [ ] Single-command execution verified: `prefect deployment run training-flow/pspnet-training`
- [ ] Flow run history visible in Prefect UI

### Week 3 — Serving API + CI/CD ⬜
- [ ] FastAPI inference endpoint (`/predict`: image upload → segmentation mask)
- [ ] Multi-stage Docker build (minimized image size)
- [ ] Inference time included in API response
- [ ] GitHub Actions workflow — push → lint/test → Docker build → GCR push → Cloud Run deploy
- [ ] Unit tests (preprocessing, API endpoint)
- [ ] Model validation gate in CI/CD pipeline
- [ ] GCP Cloud Run deployment verified

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

### 1. Clone
```bash
git clone https://github.com/Jin4202/PSPNet-mlops.git
cd PSPNet_mlops
```

### 2. Install
```bash
pip install torch torchvision mlflow pyyaml dvc dvc-gs pillow numpy --ignore-installed blinker
```

### 3. GCS Auth

**Step 1 — Install gcloud CLI (local)**
```bash
brew install --cask google-cloud-sdk
```

**Step 2 — Login & project setup (local)**
```bash
gcloud auth login
gcloud config set project <project-id>
gcloud auth application-default login
gcloud auth application-default set-quota-project <project-id>
```

> Check project-id: `gcloud projects describe <project-number> --format="value(projectId)"`

**Step 3 — Copy credentials to RunPod (local)**
```bash
# RunPod SSH: Pod → Connect → SSH over exposed TCP
# e.g. ssh root@69.30.85.12 -p 22166 -i ~/.ssh/id_ed25519

ssh root@<runpod-ip> -p <runpod-port> -i ~/.ssh/id_ed25519 \
  "mkdir -p /root/.config/gcloud"

scp -P <runpod-port> -i ~/.ssh/id_ed25519 \
  ~/.config/gcloud/application_default_credentials.json \
  root@<runpod-ip>:/root/.config/gcloud/application_default_credentials.json
```

### 4. Data
```bash
dvc pull
```

### 5. MLflow Server (Terminal A)
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --allowed-hosts "*"
```

> MLflow UI on RunPod: Pod → Connect → HTTP 5000

### 6. Train (Terminal B)
```bash
python src/train.py --config configs/config.yaml
```

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