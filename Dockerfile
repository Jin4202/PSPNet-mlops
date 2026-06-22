# syntax=docker/dockerfile:1

# ---- Stage 1: build a venv with serving deps only --------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements-serving.txt .

# CPU-only torch/torchvision wheels — the serving image never trains, so the
# CUDA runtime bundled with the default PyPI wheels would only bloat the
# image (~2GB+) for no benefit.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements-serving.txt

# ---- Stage 2: minimal runtime image -----------------------------------------
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /srv

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

COPY app/ app/
COPY src/ src/
COPY configs/ configs/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
