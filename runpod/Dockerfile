# syntax=docker/dockerfile:1
#
# RunPod training/dev image. Bakes in the Python environment only — source
# code is cloned/pulled fresh by setup.sh at pod start (see docs/runpod_setup.md),
# so rebuilding this image is only needed when requirements.txt changes.
#
# Built FROM RunPod's official PyTorch template so its sshd/Jupyter/GPU setup
# keep working unmodified — verify this tag is still current against
# https://hub.docker.com/r/runpod/pytorch/tags before building.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# gcloud CLI — non-interactive install, needed for `gcloud auth activate-service-account`
# and by dvc-gs under the hood.
RUN curl -sSL https://sdk.cloud.google.com > /tmp/gcloud-install.sh \
    && bash /tmp/gcloud-install.sh --disable-prompts --install-dir=/opt \
    && rm /tmp/gcloud-install.sh
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

# gh CLI — official apt repo, used by setup.sh for `gh auth login` / `gh auth setup-git`.
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Only the dependency manifest is copied — no source code, no secrets.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runpod/setup.sh /usr/local/bin/setup.sh
RUN chmod +x /usr/local/bin/setup.sh

# No ENTRYPOINT/CMD override — inherit the base image's, so the pod boots
# exactly like a stock RunPod PyTorch pod. Run `setup.sh` manually after SSHing in.
