#!/usr/bin/env python3
"""
FrugaLLM v3 — Gatekeeper Middleware Service
=============================================

A FastAPI reverse proxy that sits between Hermes and LiteLLM, intercepting
all OpenAI-compatible chat completion requests to enforce tool call integrity.

Flow:
    1. Hermes → Gatekeeper (port 5050)
    2. Gatekeeper → LiteLLM (internal port 4000)
    3. If response has tool_calls → pass through immediately
    4. If response is text-only → send to Classifier (internal port 8000)
    5. If classifier says "empty promise" → internally retry with reprimand
       appended to the conversation (up to MAX_RETRIES)
    6. If retries exhausted → return the last reprimand to Hermes
    7. If classifier says NOT empty promise → pass through the original response

Retry Strategy ("Internal Loop"):
    The Gatekeeper owns the retry loop, not Hermes. When an empty promise is
    detected, the Gatekeeper appends the LLM's failed response + a system
    reprimand to the message history and re-calls LiteLLM internally. This
    removes any dependency on Hermes' retry semantics. The loop caps at
    MAX_RETRIES (default: 3) to prevent infinite cycling with persistently
    chatty models.

Streaming:
    When stream=True is requested by Hermes, the Gatekeeper forces stream=False
    upstream for validation. Once a valid response is obtained (either with
    tool_calls or classifier-cleared), it re-wraps the response as proper
    OpenAI-format SSE chat.completion.chunk events. This trades away
    time-to-first-token benefit but is required for correctness since the full
    response must be available before classification can run.

Scope Note:
    This service catches "no tool_calls at all" empty promises. Malformed or
    truncated tool call JSON is a separate failure mode handled by the in-process
    v2 Gatekeeper (Tier 1/2 validation in frugallm/gatekeeper.py), which can
    still run inside LiteLLM alongside this external Gatekeeper.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("frugallm-gatekeeper")

# ─── Configuration ────────────────────────────────────────────────────────────
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000")
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
REQUEST_TIMEOUT = float(os.getenv("GATEKEEPER_TIMEOUT", "300"))
MAX_RETRIES = int(os.getenv("GATEKEEPER_MAX_RETRIES", "3"))

REPRIMAND_MESSAGE = (
    "[SYSTEM REPRIMAND: You detailed a plan and informed the user you were "
    "taking action, but failed to output the corresponding JSON tool call. "
    "Do not apologize. Output the required tool call immediately.]"
)

# ─── HTTP Clients (Connection Pooling) ────────────────────────────────────────
_litellm_client: httpx.AsyncClient | None = None
_classifier_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage httpx connection pools across the app lifecycle."""
    global _litellm_client, _classifier_client

    log.info(
        f"🛡️ Gatekeeper starting — LiteLLM: {LITELLM_URL} | "
        f"Classifier: {CLASSIFIER_URL} | Max retries: {MAX_RETRIES}"
    )

    # Connection pool: reuse TCP connections, HTTP/1.1 keep-alive
    pool_limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    )

    _litellm_client = httpx.AsyncClient(
        base_url=LITELLM_URL,
        limits=pool_limits,
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        http2=False,  # LiteLLM serves HTTP/1.1
    )

    _classifier_client = httpx.AsyncClient(
        base_url=CLASSIFIER_URL,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        timeout=httpx.Timeout(30.0, connect=5.0),
    )

    log.info("✅ Gatekeeper ready — connection pools initialized")
    yield

    log.info("🛑 Gatekeeper shutting down — closing connection pools")
    await _litellm_client.aclose()
    await _classifier_client.aclose()
    _litellm_client = None
    _classifier_client = None


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="FrugaLLM Gatekeeper",
    description="Intelligent middleware for Empty Promise detection and retry",
    version="3.0.0",
    lifespan=lifespan,
)


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Gatekeeper health — also checks upstream service connectivity."""
    status = {"gatekeeper": "ready", "litellm": "unknown", "classifier": "unknown"}

    try:
        resp = await _litellm_client.get("/health/readiness", timeout=5.0)
        status["litellm"] = "healthy" if resp.status_code == 200 else f"error:{resp.status_code}"
    except Exception as e:
        status["litellm"] = f"unreachable: {type(e).__name__}"

    try:
        resp = await _classifier_client.get("/health", timeout=5.0)
        status["classifier"] = "healthy" if resp.status_code == 200 else f"error:{resp.status_code}"
    except Exception as e:
        status["classifier"] = f"unreachable: {type(e).__name__}"

    healthy = status["litellm"] == "healthy" and status["classifier"] == "healthy"
    return JSONResponse(content=status, status_code=200 if healthy else 503)


# ─── Chat Completion Interceptor ─────────────────────────────────────────────
@app.api_route("/v1/chat/completions", methods=["POST"], include_in_schema=False)
@app.api_route("/chat/completions", methods=["POST"], include_in_schema=False)
async def intercept_chat_completion(request: Request):
    """
    Intercept chat completion requests, validate for empty promises,
    and retry internally if detected.
    """
    # ── 1. Read and parse request body ────────────────────────────────────
    body_bytes = await request.body()
    try:
        req_json = json.loads(body_bytes)
    except json.JSONDecodeError:
        return await _proxy_raw(request, body_bytes)

    # ── 2. Handle streaming — force stream=False for validation ───────────
    was_streaming = req_json.get("stream", False)
    if was_streaming:
        log.info("🛡️ Intercepted stream=True — forcing non-streaming for validation")
        req_json["stream"] = False

    # ── 3. Extract original headers for forwarding ────────────────────────
    forward_headers = _build_forward_headers(request)

    # ── 4. Internal retry loop ────────────────────────────────────────────
    # The Gatekeeper owns the retry loop. On empty promise detection, it
    # appends the failed response + reprimand to the conversation and
    # re-calls LiteLLM. Hermes never sees the intermediate failures.
    messages = req_json.get("messages", [])

    for attempt in range(1, MAX_RETRIES + 1):
        log.info(
            f"🛡️ Attempt {attempt}/{MAX_RETRIES} — "
            f"model={req_json.get('model', '?')} "
            f"tools={len(req_json.get('tools', []))} "
            f"messages={len(messages)}"
        )

        req_json["messages"] = messages
        upstream_body = json.dumps(req_json).encode("utf-8")

        # ── Call LiteLLM ──────────────────────────────────────────────────
        try:
            litellm_resp = await _litellm_client.post(
                "/v1/chat/completions",
                content=upstream_body,
                headers=forward_headers,
            )
        except httpx.TimeoutException:
            log.error(f"🛡️ LiteLLM timeout after {REQUEST_TIMEOUT}s on attempt {attempt}")
            return JSONResponse(
                content={"error": {"message": f"Upstream timeout ({REQUEST_TIMEOUT}s)", "type": "timeout"}},
                status_code=504,
            )
        except httpx.ConnectError as e:
            log.error(f"🛡️ Cannot connect to LiteLLM: {type(e).__name__}")
            return JSONResponse(
                content={"error": {"message": "Upstream connection failed", "type": "connection_error"}},
                status_code=502,
            )

        # ── Non-200 → pass through error responses directly ──────────────
        if litellm_resp.status_code != 200:
            log.warning(f"🛡️ LiteLLM returned {litellm_resp.status_code} — passing through")
            return Response(
                content=litellm_resp.content,
                status_code=litellm_resp.status_code,
                media_type=litellm_resp.headers.get("content-type", "application/json"),
            )

        # ── Parse response ────────────────────────────────────────────────
        try:
            resp_json = litellm_resp.json()
        except json.JSONDecodeError:
            log.warning("🛡️ Failed to parse LiteLLM JSON — passing through raw")
            return _maybe_restream(litellm_resp.content, was_streaming, resp_json=None)

        # ── THE CHECK: Does the response contain tool_calls? ──────────────
        choices = resp_json.get("choices", [])
        if not choices:
            log.warning("🛡️ No choices in LiteLLM response — passing through")
            return _maybe_restream(litellm_resp.content, was_streaming, resp_json=resp_json)

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls:
            # ✅ Tool calls present — pass through immediately
            log.info(
                f"✅ Response contains {len(tool_calls)} tool_call(s) — "
                f"passing through (attempt {attempt})"
            )
            if attempt > 1:
                resp_json["_gatekeeper"] = {"retries": attempt - 1, "version": "3.0"}
            return _maybe_restream(litellm_resp.content, was_streaming, resp_json=resp_json)

        # ── No tool_calls — classify the text content ─────────────────────
        content_text = message.get("content", "") or ""
        if not content_text.strip():
            log.info("🛡️ Empty content with no tool_calls — passing through")
            return _maybe_restream(litellm_resp.content, was_streaming, resp_json=resp_json)

        # ── Call classifier ───────────────────────────────────────────────
        log.info(f"🛡️ No tool_calls — classifying {len(content_text)} chars")

        try:
            classify_resp = await _classifier_client.post(
                "/classify",
                json={"text": content_text},
            )
            classify_result = classify_resp.json()
        except Exception as e:
            # Classifier down → fail open, pass through original response
            log.error(f"🛡️ Classifier failed: {type(e).__name__} — passing through")
            return _maybe_restream(litellm_resp.content, was_streaming, resp_json=resp_json)

        is_empty_promise = classify_result.get("is_empty_promise", False)
        confidence = classify_result.get("confidence", 0.0)

        log.info(
            f"📊 Classifier: is_empty_promise={is_empty_promise} "
            f"confidence={confidence:.4f} "
            f"label={classify_result.get('label', '?')!r} "
            f"latency={classify_result.get('inference_ms', '?')}ms"
        )

        # ── PASS-THROUGH: Classifier says it's fine ───────────────────────
        if not is_empty_promise:
            log.info("✅ Classifier cleared response — passing through")
            return _maybe_restream(litellm_resp.content, was_streaming, resp_json=resp_json)

        # ── EMPTY PROMISE DETECTED — prepare internal retry ───────────────
        log.warning(
            f"🚨 EMPTY PROMISE on attempt {attempt}/{MAX_RETRIES} "
            f"(confidence={confidence:.4f}) — "
            f"{'retrying internally' if attempt < MAX_RETRIES else 'exhausted, returning reprimand'}"
        )

        if attempt < MAX_RETRIES:
            # Append the LLM's failed response + reprimand for the next attempt
            # This gives the model context about what it did wrong
            messages = list(messages)  # copy to avoid mutation
            messages.append({"role": "assistant", "content": content_text})
            messages.append({"role": "user", "content": REPRIMAND_MESSAGE})
            continue

    # ── Retries exhausted — return the reprimand to Hermes ────────────────
    log.error(f"💥 Gatekeeper exhausted all {MAX_RETRIES} retries — returning reprimand")

    reprimand_response = _build_reprimand_response(
        model=resp_json.get("model", "unknown"),
        confidence=confidence,
        attempts=MAX_RETRIES,
    )

    return _maybe_restream(
        json.dumps(reprimand_response).encode("utf-8"),
        was_streaming,
        resp_json=reprimand_response,
    )


# ─── Catch-All Proxy ─────────────────────────────────────────────────────────
@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_passthrough(request: Request, path: str):
    """
    Proxy all non-chat-completion requests directly to LiteLLM.
    Handles /v1/models, /health, /v1/embeddings, etc.
    """
    body = await request.body()
    return await _proxy_raw(request, body, path=f"/{path}")


# ═════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═════════════════════════════════════════════════════════════════════════════

def _build_forward_headers(request: Request) -> dict[str, str]:
    """
    Extract headers to forward to LiteLLM.
    Preserves Authorization, Content-Type, and custom headers.
    Drops hop-by-hop headers that shouldn't be forwarded.
    Note: request bodies are never logged to prevent leaking API keys/secrets.
    """
    skip = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in skip:
            headers[key] = value
    headers["content-type"] = "application/json"
    return headers


def _build_reprimand_response(model: str, confidence: float, attempts: int) -> dict:
    """
    Construct a synthetic OpenAI-compatible assistant message containing
    the system reprimand after all internal retries are exhausted.

    Returned as a regular assistant message with finish_reason="stop".
    Hermes will process this as a text response from the LLM and can
    decide whether to retry at its own layer.
    """
    return {
        "id": f"chatcmpl-gk-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": REPRIMAND_MESSAGE,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "_gatekeeper": {
            "version": "3.0",
            "action": "reprimand",
            "attempts": attempts,
            "classifier_confidence": confidence,
        },
    }


def _maybe_restream(
    body_bytes: bytes,
    was_streaming: bool,
    resp_json: dict | None = None,
) -> Response:
    """
    If the original request was streaming, re-wrap the JSON response as
    proper OpenAI-format SSE chat.completion.chunk events so Hermes gets
    the expected streaming format.

    The SSE format matches OpenAI's spec:
        data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{...}}]}
        data: [DONE]

    If not streaming, returns the JSON body directly.
    """
    if not was_streaming:
        return Response(content=body_bytes, media_type="application/json")

    # Build proper chat.completion.chunk from the full response
    if resp_json is None:
        try:
            resp_json = json.loads(body_bytes)
        except json.JSONDecodeError:
            # Can't parse — just wrap raw content as a single SSE event
            return _raw_sse_wrap(body_bytes)

    chunk_id = resp_json.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    model = resp_json.get("model", "unknown")
    choices = resp_json.get("choices", [])

    if not choices:
        return _raw_sse_wrap(body_bytes)

    message = choices[0].get("message", {})
    finish_reason = choices[0].get("finish_reason", "stop")

    async def sse_generator():
        # Chunk 1: role delta (OpenAI always sends role first)
        role_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": resp_json.get("created", int(time.time())),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": message.get("role", "assistant")},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        # Chunk 2: content or tool_calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            # Emit each tool call as a separate chunk (matching OpenAI format)
            for i, tc in enumerate(tool_calls):
                tc_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": resp_json.get("created", int(time.time())),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": i,
                                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("function", {}).get("name", ""),
                                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(tc_chunk)}\n\n"
        else:
            # Emit content as a single chunk
            content = message.get("content")
            if content:
                content_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": resp_json.get("created", int(time.time())),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"

        # Final chunk: finish_reason
        done_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": resp_json.get("created", int(time.time())),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "connection": "keep-alive",
            "x-accel-buffering": "no",
        },
    )


def _raw_sse_wrap(body_bytes: bytes) -> StreamingResponse:
    """Fallback SSE wrapper when we can't parse the response as JSON."""
    async def sse_generator():
        yield f"data: {body_bytes.decode('utf-8', errors='replace')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "connection": "keep-alive"},
    )


async def _proxy_raw(
    request: Request,
    body: bytes,
    path: str | None = None,
) -> Response:
    """
    Transparent reverse proxy — forwards the request to LiteLLM unchanged
    and returns the raw response.
    """
    target_path = path or request.url.path
    headers = _build_forward_headers(request)

    try:
        resp = await _litellm_client.request(
            method=request.method,
            url=target_path,
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )
    except httpx.TimeoutException:
        return JSONResponse(
            content={"error": {"message": "Upstream timeout", "type": "timeout"}},
            status_code=504,
        )
    except httpx.ConnectError:
        return JSONResponse(
            content={"error": {"message": "Upstream connection failed", "type": "connection_error"}},
            status_code=502,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
