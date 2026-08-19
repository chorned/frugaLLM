#!/usr/bin/env python3
"""
FrugaLLM Gatekeeper App — Unit Tests
========================================

Tests for the external Gatekeeper FastAPI middleware (gatekeeper/app.py),
covering the three bug fixes:

1. /health endpoint merges LiteLLM's healthy_endpoints into its response
2. Non-dict JSON payloads (e.g. arrays) are proxied through without crashing
3. /v1/models forwards query parameters to LiteLLM

These tests mock all upstream HTTP calls (LiteLLM, Classifier) using respx,
so they run without any running services.

Run with:
    source test_venv/bin/activate && pytest tests/test_gatekeeper_app.py -v
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

# Ensure the gatekeeper module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gatekeeper')))

# We need to set up mock clients BEFORE importing the app, because the
# lifespan manager creates them.  Instead, we'll import and then patch.
import app as gatekeeper_app


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mock_httpx_response(status_code: int, json_data: dict | list | None = None, content: bytes | None = None) -> httpx.Response:
    """Create a mock httpx.Response."""
    if json_data is not None:
        return httpx.Response(status_code, json=json_data)
    if content is not None:
        return httpx.Response(status_code, content=content)
    return httpx.Response(status_code)


def _make_litellm_client_mock() -> AsyncMock:
    """Create a mock for the LiteLLM async client."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    return mock


def _make_classifier_client_mock() -> AsyncMock:
    """Create a mock for the Classifier async client."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    return mock


# ═════════════════════════════════════════════════════════════════════════════
# Issue 1: /health endpoint provides model roster via /v1/models
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthEndpointMerge:
    """Verify the /health endpoint returns readiness status + model roster."""

    @pytest.mark.asyncio
    async def test_health_returns_model_roster(self):
        """
        The Gatekeeper /health should include healthy_endpoints built from
        LiteLLM's /v1/models, so the CLI's --models flag works.
        """
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        call_log = []

        async def mock_get(path, **kwargs):
            call_log.append(path)
            if path == "/health/readiness":
                return _mock_httpx_response(200, {"status": "healthy", "db": "connected"})
            if path == "/v1/models":
                return _mock_httpx_response(200, {
                    "data": [
                        {"id": "gemini/gemini-3.6-flash", "object": "model"},
                        {"id": "ollama/hermes:latest", "object": "model"},
                        {"id": "thinker", "object": "model"},
                    ],
                    "object": "list",
                })
            return _mock_httpx_response(404)

        litellm_mock.get = mock_get
        classifier_mock.get = AsyncMock(return_value=_mock_httpx_response(200, {"status": "ok"}))

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.get("/health", headers={"Authorization": "Bearer test-key"})

        data = resp.json()

        # Gatekeeper's own fields
        assert data["gatekeeper"] == "ready"
        assert data["litellm"] == "healthy"
        assert data["classifier"] == "healthy"

        # Model roster from /v1/models
        assert "healthy_endpoints" in data
        assert len(data["healthy_endpoints"]) == 3
        assert data["healthy_count"] == 3
        assert data["unhealthy_endpoints"] == []
        assert data["unhealthy_count"] == 0

        # Readiness status
        assert data.get("status") == "healthy"

    @pytest.mark.asyncio
    async def test_health_forwards_auth_for_models(self):
        """The Authorization header should be forwarded to LiteLLM's /v1/models."""
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        captured_headers = {}

        async def capture_get(path, **kwargs):
            if path == "/v1/models":
                captured_headers.update(kwargs.get("headers", {}))
                return _mock_httpx_response(200, {"data": [], "object": "list"})
            # /health/readiness
            return _mock_httpx_response(200, {"status": "healthy"})

        litellm_mock.get = capture_get
        classifier_mock.get = AsyncMock(return_value=_mock_httpx_response(200, {"status": "ok"}))

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                await client.get("/health", headers={"Authorization": "Bearer my-secret-key"})

        assert captured_headers.get("authorization") == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_health_ok_without_model_roster(self):
        """
        If /v1/models fails, /health should still report healthy status
        without the model roster (fail open on optional data).
        """
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        async def mock_get(path, **kwargs):
            if path == "/health/readiness":
                return _mock_httpx_response(200, {"status": "healthy"})
            # /v1/models fails
            raise httpx.TimeoutException("timed out")

        litellm_mock.get = mock_get
        classifier_mock.get = AsyncMock(return_value=_mock_httpx_response(200, {"status": "ok"}))

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.get("/health")

        data = resp.json()

        assert resp.status_code == 200
        assert data["litellm"] == "healthy"
        assert data["classifier"] == "healthy"
        # No healthy_endpoints since /v1/models timed out
        assert "healthy_endpoints" not in data

    @pytest.mark.asyncio
    async def test_health_litellm_error_status(self):
        """When LiteLLM readiness returns a non-200, the status should reflect the error code."""
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        async def mock_get(path, **kwargs):
            if path == "/health/readiness":
                return _mock_httpx_response(503, {"error": "unhealthy"})
            return _mock_httpx_response(200, {"data": []})

        litellm_mock.get = mock_get
        classifier_mock.get = AsyncMock(return_value=_mock_httpx_response(200, {"status": "ok"}))

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.get("/health")

        data = resp.json()

        assert resp.status_code == 503
        assert data["litellm"] == "error:503"

    @pytest.mark.asyncio
    async def test_health_status_field_from_readiness(self):
        """The 'status' field from /health/readiness should be included."""
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        async def mock_get(path, **kwargs):
            if path == "/health/readiness":
                return _mock_httpx_response(200, {"status": "healthy", "db": "connected"})
            return _mock_httpx_response(200, {"data": [], "object": "list"})

        litellm_mock.get = mock_get
        classifier_mock.get = AsyncMock(return_value=_mock_httpx_response(200, {"status": "ok"}))

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.get("/health")

        data = resp.json()
        assert data["status"] == "healthy"
        assert data["litellm"] == "healthy"


# ═════════════════════════════════════════════════════════════════════════════
# Issue 2: Non-dict JSON payloads should not crash
# ═════════════════════════════════════════════════════════════════════════════

class TestNonDictJsonPayload:
    """Verify that non-dict JSON payloads are proxied through without crashing."""

    @pytest.mark.asyncio
    async def test_json_array_payload_proxied(self):
        """
        A valid JSON array sent to /v1/chat/completions should be proxied
        through to LiteLLM unchanged, not crash with AttributeError.
        """
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        litellm_mock.request = AsyncMock(
            return_value=_mock_httpx_response(400, {"error": {"message": "Invalid request"}})
        )

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    content=json.dumps([{"role": "user", "content": "hi"}]),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
                )

        # Should be proxied through (400 from upstream), NOT a 500
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_json_string_payload_proxied(self):
        """A bare JSON string should also be proxied, not crash."""
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        litellm_mock.request = AsyncMock(
            return_value=_mock_httpx_response(400, {"error": {"message": "Invalid"}})
        )

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    content=json.dumps("just a string"),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
                )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_json_integer_payload_proxied(self):
        """A bare JSON integer should also be proxied, not crash."""
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        litellm_mock.request = AsyncMock(
            return_value=_mock_httpx_response(400, {"error": {"message": "Invalid"}})
        )

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    content="42",
                    headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
                )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_dict_payload_still_intercepted(self):
        """
        Normal dict payloads should still go through the Gatekeeper's
        interception logic (not be blindly proxied).
        """
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        litellm_mock.post = AsyncMock(return_value=_mock_httpx_response(200, {
            "id": "test-123",
            "model": "thinker",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }))

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    content=json.dumps({
                        "model": "thinker",
                        "messages": [{"role": "user", "content": "test"}],
                    }),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["tool_calls"] is not None


# ═════════════════════════════════════════════════════════════════════════════
# Issue 3: /v1/models forwards query parameters
# ═════════════════════════════════════════════════════════════════════════════

class TestModelsQueryParams:
    """Verify that /v1/models forwards query parameters to LiteLLM."""

    @pytest.mark.asyncio
    async def test_models_forwards_query_params(self):
        """Query params on /v1/models should be forwarded to LiteLLM."""
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        captured_kwargs = {}

        async def capture_get(path, **kwargs):
            captured_kwargs.update(kwargs)
            return _mock_httpx_response(200, {
                "data": [{"id": "thinker", "object": "model"}],
                "object": "list",
            })

        litellm_mock.get = capture_get

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.get(
                    "/v1/models?user=hermes&format=json",
                    headers={"Authorization": "Bearer test"},
                )

        assert resp.status_code == 200
        # Verify params were forwarded
        params = captured_kwargs.get("params", {})
        assert params.get("user") == "hermes"
        assert params.get("format") == "json"

    @pytest.mark.asyncio
    async def test_models_without_query_params(self):
        """
        /v1/models without query params should still work (no KeyError, etc.).
        """
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        litellm_mock.get = AsyncMock(return_value=_mock_httpx_response(200, {
            "data": [
                {"id": "thinker", "object": "model"},
                {"id": "reasoning", "object": "model"},
            ],
            "object": "list",
        }))

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.get(
                    "/v1/models",
                    headers={"Authorization": "Bearer test"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2

    @pytest.mark.asyncio
    async def test_models_filters_pseudo_models(self):
        """
        The Gatekeeper should filter out pseudo-models (free_balanced,
        free_reasoning, openrouter/google/gemini-2.5-flash:free) from the
        /v1/models listing.
        """
        litellm_mock = _make_litellm_client_mock()
        classifier_mock = _make_classifier_client_mock()

        litellm_mock.get = AsyncMock(return_value=_mock_httpx_response(200, {
            "data": [
                {"id": "thinker", "object": "model"},
                {"id": "free_balanced", "object": "model"},
                {"id": "free_balanced_2", "object": "model"},
                {"id": "free_reasoning", "object": "model"},
                {"id": "openrouter/google/gemini-2.5-flash:free", "object": "model"},
                {"id": "reasoning", "object": "model"},
            ],
            "object": "list",
        }))

        with patch.object(gatekeeper_app, '_litellm_client', litellm_mock), \
             patch.object(gatekeeper_app, '_classifier_client', classifier_mock):

            transport = httpx.ASGITransport(app=gatekeeper_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.get(
                    "/v1/models",
                    headers={"Authorization": "Bearer test"},
                )

        assert resp.status_code == 200
        data = resp.json()
        model_ids = {m["id"] for m in data["data"]}
        # These should be filtered out
        assert "free_balanced" not in model_ids
        assert "free_balanced_2" not in model_ids
        assert "free_reasoning" not in model_ids
        assert "openrouter/google/gemini-2.5-flash:free" not in model_ids
        # These should remain
        assert "thinker" in model_ids
        assert "reasoning" in model_ids
