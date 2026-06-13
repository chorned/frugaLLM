#!/usr/bin/env python3
"""
Hermes Router Proxy — Zero-Dependency OpenAI-Compatible Proxy
==============================================================

Uses ONLY Python stdlib for the HTTP server (no Flask needed).
External deps: requests, openai — both already in the Hermes venv.

Run with the Hermes venv Python:
  ~/.hermes/hermes-agent/venv/bin/python3 ~/.hermes/skills/router_server.py

Architecture:
  Hermes Agent  →  localhost:5050/v1  →  Router Proxy  →  OpenRouter / Ollama
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import urllib.request
import urllib.error

# ─── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROFILE_DIR = _SCRIPT_DIR.parent
_CONFIG_PATH = _PROFILE_DIR / "config.yaml"
_SOUL_PATH = _PROFILE_DIR / "SOUL.md"
_ENV_PATH = _PROFILE_DIR / ".env"

# ─── Environment ──────────────────────────────────────────────────────────────
def _load_env():
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

_load_env()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
LOCAL_OLLAMA_BASE = "http://127.0.0.1:11434"
LOCAL_OLLAMA_MODEL = "hermes:latest"
PROXY_PORT = 5050

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Router] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hermes-router")

# ─── Config Loader ────────────────────────────────────────────────────────────
def _load_routing_config() -> dict:
    defaults = {"current_profile": "engineer", "task_size": "small", "allow_pro": False}
    if not _CONFIG_PATH.exists():
        return defaults
    try:
        text = _CONFIG_PATH.read_text()
        m = re.search(r"^\s*current_profile:\s*(\S+)", text, re.MULTILINE)
        if m: defaults["current_profile"] = m.group(1).strip().strip("'\"")
        m = re.search(r"^\s*task_size:\s*(\S+)", text, re.MULTILINE)
        if m: defaults["task_size"] = m.group(1).strip().strip("'\"")
        m = re.search(r"^\s*allow_pro:\s*(\S+)", text, re.MULTILINE)
        if m: defaults["allow_pro"] = m.group(1).strip().strip("'\"").lower() in ("true", "yes", "1")
    except Exception as e:
        log.warning(f"Config parse warning: {e}")
    return defaults

# ─── Dynamic Free Model Discovery ────────────────────────────────────────────
def get_best_free_models() -> dict[str, str]:
    fallbacks = {
        "balanced_free": "google/gemini-2.5-flash:free",
        "reasoning_free": "deepseek/deepseek-r1:free",
        "pro_escalation": "google/gemini-3.1-pro-preview",
    }
    try:
        log.info("☀ Morning Routine — scanning OpenRouter for free models...")
        req = urllib.request.Request(
            f"{OPENROUTER_BASE}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"} if OPENROUTER_API_KEY else {}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8")).get("data", [])

        free = []
        for m in data:
            p = m.get("pricing", {})
            try:
                if float(p.get("prompt", "1")) == 0 and float(p.get("completion", "1")) == 0:
                    free.append(m)
            except (ValueError, TypeError):
                continue

        free.sort(key=lambda x: x.get("context_length", 0), reverse=True)
        if free:
            fallbacks["balanced_free"] = free[0]["id"]
            log.info(f"✓ balanced_free: {free[0]['id']} ({free[0].get('context_length', '?'):,} ctx)")
            reasoning = [m for m in free if any(k in m["id"].lower() for k in ("deepseek", "r1", "reason", "think"))]
            if reasoning:
                reasoning.sort(key=lambda x: x.get("context_length", 0), reverse=True)
                fallbacks["reasoning_free"] = reasoning[0]["id"]
                log.info(f"✓ reasoning_free: {reasoning[0]['id']}")

        log.info("Model roster locked:")
        for role, mid in fallbacks.items():
            log.info(f"  {'💰' if role == 'pro_paid' else '🆓'} {role:18s} → {mid}")
        return fallbacks
    except Exception as e:
        log.warning(f"Dynamic fetch failed, using fallbacks: {e}")
        return fallbacks

MODELS = get_best_free_models()

# ─── Escalation Detection ────────────────────────────────────────────────────
_ESCALATION_KEYWORDS = ["//escalate", "hey hermes, use your pro brain"]
_consecutive_failures = 0
_FAILURE_THRESHOLD = 3

def _check_escalation(messages):
    for msg in messages:
        c = msg.get("content", "")
        if isinstance(c, str) and any(kw in c.lower() for kw in _ESCALATION_KEYWORDS):
            return True
    return False

def _record_failure():
    global _consecutive_failures
    _consecutive_failures += 1
    return _consecutive_failures >= _FAILURE_THRESHOLD

def _reset_failures():
    global _consecutive_failures
    _consecutive_failures = 0

# ─── Routing Engine ──────────────────────────────────────────────────────────
def resolve_route(requested_model, messages):
    """
    3-Tier Escalation Path:
      Tier 1 (Default)    → balanced_free   (best free general model)
      Tier 2 (Reasoning)  → reasoning_free  (best free reasoning model)
      Tier 3 (Escalation) → pro_escalation  (Gemini 3.1 Pro via VertexAI)

    Returns (actual_model, backend_base_url, is_local)
    """
    config = _load_routing_config()
    allow_pro = config["allow_pro"]

    if _check_escalation(messages):
        log.info("🔑 Keyword escalation detected.")
        allow_pro = True
    if _consecutive_failures >= _FAILURE_THRESHOLD:
        log.info(f"🔄 Validation loop escalation ({_consecutive_failures} failures).")
        allow_pro = True

    label = (requested_model or "auto").lower().strip()

    # ── Explicit model labels ──
    if label == "local":
        return LOCAL_OLLAMA_MODEL, LOCAL_OLLAMA_BASE, True
    if label == "balanced_free":
        return MODELS["balanced_free"], OPENROUTER_BASE, False
    if label == "reasoning_free" or label == "reasoning":
        return MODELS["reasoning_free"], OPENROUTER_BASE, False
    if label == "pro" or label == "escalate":
        if allow_pro:
            return MODELS["pro_escalation"], OPENROUTER_BASE, False
        log.warning("⚠️  Pro denied (allow_pro=false). Falling back to reasoning.")
        return MODELS["reasoning_free"], OPENROUTER_BASE, False

    # ── Auto: Tier 1 (Default → balanced_free) ──
    if label == "auto":
        m = MODELS["balanced_free"]
        log.info(f"☁️  auto → {m} [Tier 1: default]")
        return m, OPENROUTER_BASE, False

    # ── Passthrough: unknown model tags go to OpenRouter verbatim ──
    log.info(f"↗️  Passthrough: {requested_model}")
    return requested_model, OPENROUTER_BASE, False

# ─── HTTP Proxy Handler ──────────────────────────────────────────────────────
class RouterHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        """Suppress default request logging — we use our own."""
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, message):
        self._send_json({"error": {"message": message, "type": "router_error", "code": status}}, status)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ── GET Endpoints ──

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        if path == "/health":
            ollama_alive = False
            try:
                urllib.request.urlopen(f"{LOCAL_OLLAMA_BASE}/api/tags", timeout=1)
                ollama_alive = True
            except Exception:
                pass
            self._send_json({
                "status": "ok", "models": MODELS,
                "ollama_alive": ollama_alive,
                "consecutive_failures": _consecutive_failures,
                "config": _load_routing_config(),
            })

        elif path == "/v1/models":
            now = int(time.time())
            self._send_json({"object": "list", "data": [
                {"id": "auto", "object": "model", "created": now, "owned_by": "hermes-router"},
                {"id": "balanced_free", "object": "model", "created": now, "owned_by": "hermes-router"},
                {"id": "reasoning_free", "object": "model", "created": now, "owned_by": "hermes-router"},
                {"id": "pro", "object": "model", "created": now, "owned_by": "hermes-router"},
                {"id": "local", "object": "model", "created": now, "owned_by": "hermes-router"},
            ]})

        else:
            self._send_error(404, f"Unknown endpoint: {path}")

    # ── POST Endpoints ──

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")

        if path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif path == "/admin/refresh-models":
            global MODELS
            MODELS = get_best_free_models()
            self._send_json({"status": "refreshed", "models": MODELS})
        elif path == "/admin/reset-failures":
            _reset_failures()
            self._send_json({"status": "ok", "consecutive_failures": 0})
        else:
            self._send_error(404, f"Unknown endpoint: {path}")

    def _handle_chat_completions(self):
        raw = self._read_body()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON body")
            return

        requested_model = data.get("model", "auto")
        messages = data.get("messages", [])
        is_streaming = data.get("stream", False)

        log.info(f"── Request: model={requested_model}, msgs={len(messages)}, stream={is_streaming}")

        # ── Measure routing decision time ──
        t_route_start = time.monotonic()
        actual_model, backend_base, is_local = resolve_route(requested_model, messages)
        t_route_ms = (time.monotonic() - t_route_start) * 1000
        log.info(f"   Resolved → {actual_model} @ {'LOCAL' if is_local else 'OPENROUTER'} ({t_route_ms:.1f}ms routing)")

        data["model"] = actual_model
        target_url = f"{backend_base}/{'v1/' if is_local else ''}chat/completions"

        headers = {"Content-Type": "application/json"}
        if not is_local:
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
            headers["HTTP-Referer"] = "https://hermes.local"
            headers["X-Title"] = "Hermes Router"

        # ── Measure upstream latency ──
        t_upstream_start = time.monotonic()
        try:
            if is_streaming:
                self._proxy_streaming(target_url, data, headers, actual_model, is_local, messages)
            else:
                self._proxy_sync(target_url, data, headers, actual_model, is_local, messages)
            t_upstream_ms = (time.monotonic() - t_upstream_start) * 1000
            log.info(f"   ✓ Done. Proxy overhead: {t_route_ms:.1f}ms | Upstream: {t_upstream_ms:.0f}ms ({t_upstream_ms/1000:.1f}s)")
        except urllib.error.URLError as e:
            if is_local:
                log.warning("⚡ Ollama offline — cloud fallback.")
                data["model"] = MODELS["balanced_free"]
                fallback_url = f"{OPENROUTER_BASE}/chat/completions"
                fb_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://hermes.local",
                    "X-Title": "Hermes Router",
                }
                try:
                    if is_streaming:
                        self._proxy_streaming(fallback_url, data, fb_headers, MODELS["balanced_free"], False, messages)
                    else:
                        self._proxy_sync(fallback_url, data, fb_headers, MODELS["balanced_free"], False, messages)
                except Exception as e2:
                    self._send_error(502, f"Cloud fallback also failed: {e2}")
            else:
                self._send_error(502, f"Connection to backend failed")
        except Exception as e:
            log.error(f"Proxy error: {e}")
            self._send_error(502, f"Router proxy error: {e}")

    def _proxy_sync(self, url, data, headers, model, is_local, messages):
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                status_code = resp.status
                content = resp.read()
        except urllib.error.HTTPError as e:
            status_code = e.code
            content = e.read()
        except urllib.error.URLError as e:
            raise Exception(f"Connection error: {e.reason}")

        if status_code == 400 and not is_local:
            body_lower = content.decode("utf-8", errors="ignore").lower()
            if any(kw in body_lower for kw in ("context", "too long", "token limit", "context_length")):
                log.warning(f"📏 Context overflow on {model} — escalating.")
                self._escalate_to_pro(data, messages, streaming=False)
                return
            _record_failure()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(content)
            return

        if status_code >= 400:
            _record_failure()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(content)
            return

        _reset_failures()
        result = json.loads(content.decode("utf-8"))
        result["_hermes_route"] = f"{'local' if is_local else 'openrouter'}:{model}"
        self._send_json(result)

    def _proxy_streaming(self, url, data, headers, model, is_local, messages):
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            _record_failure()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
            return
        except urllib.error.URLError as e:
            raise Exception(f"Connection error: {e.reason}")

        _reset_failures()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Hermes-Route", f"{'local' if is_local else 'openrouter'}:{model}")
        self.end_headers()

        for line in resp:
            self.wfile.write(line)
            self.wfile.flush()

    def _escalate_to_pro(self, data, messages, streaming=False):
        config = _load_routing_config()
        if not (config["allow_pro"] or _check_escalation(messages) or _consecutive_failures >= _FAILURE_THRESHOLD):
            self._send_error(503, "Pro escalation denied. Set routing.allow_pro: true or use //escalate.")
            return

        pro_model = MODELS["pro_escalation"]
        log.warning(f"🚀 TIER 3 ESCALATION: {pro_model} (VertexAI)")

        notice = "\n\n--- ESCALATION NOTICE ---\nYou were escalated because a smaller model failed. Be precise."
        modified = []
        sys_found = False
        for msg in messages:
            if msg.get("role") == "system" and not sys_found:
                modified.append({**msg, "content": msg.get("content", "") + notice})
                sys_found = True
            else:
                modified.append(msg)
        if not sys_found:
            modified.insert(0, {"role": "system", "content": notice.strip()})

        data["model"] = pro_model
        data["messages"] = modified

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://hermes.local",
            "X-Title": "Hermes Router (Pro)",
        }

        try:
            if streaming:
                self._proxy_streaming(f"{OPENROUTER_BASE}/chat/completions", data, headers, pro_model, False, messages)
            else:
                self._proxy_sync(f"{OPENROUTER_BASE}/chat/completions", data, headers, pro_model, False, messages)
        except Exception as e:
            self._send_error(502, f"Pro escalation failed: {e}")


# ─── Entry Point ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hermes Router Proxy (stdlib)")
    parser.add_argument("--port", "-p", type=int, default=PROXY_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           HERMES ROUTER PROXY — 3-Tier Escalation           ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoint:  http://{args.host}:{args.port}/v1
║  Health:    http://{args.host}:{args.port}/health
╠══════════════════════════════════════════════════════════════╣
║  Tier 1 (Default)   → {MODELS['balanced_free']}
║  Tier 2 (Reasoning) → {MODELS['reasoning_free']}
║  Tier 3 (Escalate)  → {MODELS['pro_escalation']}
╠══════════════════════════════════════════════════════════════╣
║  Config: {_CONFIG_PATH}
║  Waiting for requests...
╚══════════════════════════════════════════════════════════════╝
""")

    server = HTTPServer((args.host, args.port), RouterHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
