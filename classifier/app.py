#!/usr/bin/env python3
"""
FrugaLLM v3 — Micro-Classifier Service
========================================

A lightweight FastAPI service that runs an ONNX-optimized zero-shot
classification model to detect "Empty Promises" — instances where an LLM
generates conversational text promising to execute a tool but fails to
actually output the required JSON tool call.

The model loads once at startup and stays in memory. All inference runs
on CPU via ONNX Runtime for blazing-fast, GPU-free classification.

Endpoints:
    POST /classify  — Classify text for empty promise detection
    GET  /health    — Readiness probe for Docker healthcheck
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("frugallm-classifier")

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "cross-encoder/nli-deberta-v3-small"
MODEL_DIR = "/app/model"  # Pre-exported ONNX model baked in at Docker build time
CANDIDATE_LABELS = [
    "promising to execute a technical action or delegate a task",
    "general conversational response",
]
CONFIDENCE_THRESHOLD = 0.85

# ─── Global Pipeline Reference ───────────────────────────────────────────────
_pipeline = None


# ─── Lifespan (Model Loading) ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX zero-shot classifier once at startup."""
    global _pipeline

    log.info(f"🧠 Loading pre-exported ONNX model from {MODEL_DIR}")
    start = time.monotonic()

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            session_options=sess_options,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        _pipeline = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
        )

        elapsed = time.monotonic() - start
        log.info(f"✅ Model loaded in {elapsed:.1f}s — ready for classification")

    except Exception as e:
        log.error(f"💥 Failed to load model: {e}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {e}") from e

    yield  # Application runs here

    log.info("🛑 Classifier shutting down")
    _pipeline = None


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="FrugaLLM Micro-Classifier",
    description="ONNX zero-shot classifier for Empty Promise detection",
    version="3.0.0",
    lifespan=lifespan,
)


# ─── Models ───────────────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="LLM output text to classify")


class ClassifyResponse(BaseModel):
    is_empty_promise: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str = Field(..., description="Winning classification label")
    inference_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model: str


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """
    Classify LLM output text for empty promise detection.

    Returns True if the text is classified as "promising to execute a
    technical action or delegate a task" with confidence > 0.85.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.monotonic()

    # Run zero-shot classification (CPU-bound — runs in the calling thread
    # since uvicorn uses a single worker and the model is lightweight)
    result = _pipeline(
        request.text,
        candidate_labels=CANDIDATE_LABELS,
        multi_label=False,
    )

    elapsed_ms = (time.monotonic() - start) * 1000

    # result format: {"labels": [...], "scores": [...], "sequence": "..."}
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    is_empty_promise = (
        top_label == CANDIDATE_LABELS[0]
        and top_score > CONFIDENCE_THRESHOLD
    )

    log.info(
        f"📊 Classification: label={top_label!r} "
        f"score={top_score:.4f} "
        f"is_empty_promise={is_empty_promise} "
        f"latency={elapsed_ms:.0f}ms"
    )

    return ClassifyResponse(
        is_empty_promise=is_empty_promise,
        confidence=round(top_score, 4),
        label=top_label,
        inference_ms=round(elapsed_ms, 1),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness probe — returns 200 only when the model is loaded."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    return HealthResponse(status="ready", model=MODEL_ID)
