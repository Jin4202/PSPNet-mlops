# PSPNet MLOps Pipeline

End-to-end MLOps system for semantic segmentation (PSPNet/CamVid).  
Training → Serving → Monitoring → Automated Retraining.

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

> project-id check: `gcloud projects describe <project-number> --format="value(projectId)"`

**Step 3 — Copy credentials to RunPod (local)**
```bash
# RunPod SSH information: Pod → Connect → SSH over exposed TCP
# Ex: ssh root@69.30.85.12 -p 22166 -i ~/.ssh/id_ed25519

# RunPod directory create
ssh root@<runpod-ip> -p <runpod-port> -i ~/.ssh/id_ed25519 \
  "mkdir -p /root/.config/gcloud"

# credentials copy
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

> RunPod MLflow UI: Pod → Connect → HTTP 5000

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
| Val mIoU | - |
| Val allAcc | - |

*To be updated after baseline run.*