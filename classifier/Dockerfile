# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Export the HuggingFace model to ONNX format. This stage pulls in torch
# for the export process, which is discarded in the final image.
FROM python:3.11 AS builder

# Install all builder deps: torch from PyTorch index + others from PyPI
RUN pip install --no-cache-dir \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/ && \
    pip install --no-cache-dir \
    transformers==4.45.2 \
    optimum[onnxruntime]==1.23.1 \
    onnxruntime==1.20.1 \
    onnxscript

# Download model from HuggingFace Hub and export to ONNX
RUN optimum-cli export onnx --model cross-encoder/nli-deberta-v3-small /model

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
# Slim final image with only runtime deps. No torch, no build tools.
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-exported ONNX model from builder (no runtime export needed)
COPY --from=builder /model /app/model

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
