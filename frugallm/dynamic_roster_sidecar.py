#!/usr/bin/env python3
"""
Hermes Dynamic Roster Sidecar
==============================

Lightweight daemon that polls OpenRouter for free models every 5 minutes
and writes the best candidates to the LiteLLM proxy via REST API.

Extracted from router_server.py:
  - _is_reasoning_model()
  - _background_model_fetch()

Strategy: Push to /model/new and /fallback REST APIs.

Run as a launchd daemon:
  ~/.hermes/venv_router/bin/python3 ~/.hermes/skills/dynamic_roster_sidecar.py
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error

# ─── Configuration ───────────────────────────────────────────────────────────
import os
_HERMES_DIR = Path(os.environ.get("FRUGALLM_CONFIG_DIR", Path.cwd() / "config"))
_ENV_PATH = Path.cwd() / ".env"
_DYNAMIC_MODELS_PATH = _HERMES_DIR / "dynamic_models.yaml"
_LITELLM_PID_PATTERN = "litellm"

POLL_INTERVAL = 300  # 5 minutes
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Sidecar] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hermes-roster-sidecar")

# ─── Environment Loading ────────────────────────────────────────────────────
def _load_env():
    """Load .env file into os.environ (same pattern as router_server.py)."""
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
        k = key.strip()
        v = value.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env()

# Use the dedicated management key if available, else fall back to inference key
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_MANAGEMENT_KEY", os.getenv("OPENROUTER_API_KEY", "")
)
# The inference key for the dynamic models (always use the standard key)
OPENROUTER_INFERENCE_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ─── State Tracking ─────────────────────────────────────────────────────────
_current_balanced: list[str] | None = None
_current_reasoning: list[str] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Extracted Logic from router_server.py (VERBATIM)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_reasoning_model(model_data: dict) -> bool:
    """
    Agnostic heuristic to detect reasoning models based on metadata.

    Extracted from router_server.py L103-120 (VERBATIM).
    """
    m_id = model_data.get("id", "").lower()
    m_name = model_data.get("name", "").lower()
    m_desc = model_data.get("description", "").lower()

    search_space = f"{m_id} {m_name} {m_desc}"
    keywords = [
        "reasoning",
        "chain-of-thought",
        "-cot-",
        "thinker",
        "thought process",
    ]

    # Check for direct heuristic keywords
    if any(kw in search_space for kw in keywords):
        return True

    # Generic matching for models that commonly include "think" or "reason" in the raw ID
    if any(kw in m_id for kw in ("-reason", "think", "-o1", "-r1")):
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# YAML File Writer (replaces Management API)
# ═══════════════════════════════════════════════════════════════════════════════

def _push_models_to_db(balanced_ids: list[str], reasoning_ids: list[str]) -> bool:
    """
    Pushes dynamic model configurations to the LiteLLM proxy via REST API.
    Replaces the previous YAML generation approach.
    """
    max_chain_len = int(os.getenv("MAX_DYNAMIC_CHAIN_LEN", "4"))
    balanced_ids = balanced_ids[:max_chain_len]
    reasoning_ids = reasoning_ids[:max_chain_len]

    def _get_local_models() -> list[str]:
        try:
            import yaml
            config_path = _HERMES_DIR / "litellm_config.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                locals_list = []
                for m in config.get("model_list", []):
                    params = m.get("litellm_params", {})
                    api_base = params.get("api_base", "")
                    api_key = params.get("api_key", "")
                    if api_base:
                        is_placeholder_key = not api_key or api_key in ("ollama", "EMPTY", "anything", "dummy", "na")
                        is_local_host = any(host in api_base for host in ("localhost", "127.0.0.1", "0.0.0.0"))
                        if is_placeholder_key or is_local_host:
                            locals_list.append(m)  # Append the whole model object
                if locals_list:
                    return locals_list
        except Exception as e:
            log.warning(f"Failed to load local models from config: {e}")
        return [{"model_name": "gemma-4-12b-gguf", "litellm_params": {"model": "openai//models/gemma-4-12b-it-Q5_K_M.gguf"}}]

    local_models_objs = _get_local_models()
    local_models_names = [m.get("model_name") for m in local_models_objs if m.get("model_name")]

    def _litellm_model(or_id: str) -> str:
        if or_id.startswith("openrouter/") or or_id.startswith("ollama/") or or_id.startswith("openai/") or or_id.startswith("gemini/"):
            return or_id
        return f"openrouter/{or_id}"

    balanced_names = [_litellm_model(b) for b in balanced_ids]
    reasoning_names = [_litellm_model(r) for r in reasoning_ids]

    balanced_fallbacks = balanced_names[1:] + local_models_names if len(balanced_names) > 0 else local_models_names
    balanced_primary = balanced_names[0] if balanced_names else local_models_names[0]

    reasoning_fallbacks = reasoning_names[1:] + local_models_names if len(reasoning_names) > 0 else local_models_names
    reasoning_primary = reasoning_names[0] if reasoning_names else local_models_names[0]

    balanced_strict_fallbacks = [n + "_strict" for n in balanced_names[1:]] + local_models_names if len(balanced_names) > 0 else local_models_names

    groups = []
    
    # 1. Base models (we must push them so fallbacks resolve)
    base_models_to_push = set()
    for b_id in balanced_ids:
        base_models_to_push.add(_litellm_model(b_id))
        base_models_to_push.add(_litellm_model(b_id) + "_strict")
    for r_id in reasoning_ids:
        base_models_to_push.add(_litellm_model(r_id))
    
    for m_name in base_models_to_push:
        groups.append((m_name, m_name.replace("_strict", ""), []))

    # 2. Add balanced aliases
    for alias in ["free_balanced", "auto", "local", "frugal", "smart", "offline", "private", "free"]:
        groups.append((alias, balanced_primary, balanced_fallbacks))
    groups.append(("free_strict_balanced", balanced_primary + "_strict" if balanced_names else local_models_names[0], balanced_strict_fallbacks))

    # 3. Add reasoning aliases
    for alias in ["frugallm", "free_reasoning", "reasoning", "thinker", "reasoner"]:
        groups.append((alias, reasoning_primary, reasoning_fallbacks))

    # 4. Add cloud aliases
    groups.append(("cloud", "gemini/gemini-3.6-flash", []))
    groups.append(("fast", "gemini/gemini-3.5-flash-lite", []))
    groups.append(("lite", "gemini/gemini-3.5-flash-lite", []))
    groups.append(("gemini-pro", "gemini/gemini-3.1-pro-preview", []))
    groups.append(("pro", "gemini/gemini-3.1-pro-preview", []))

    proxy_port = os.getenv("FRUGALLM_PROXY_PORT", "5050")
    proxy_host = os.getenv("FRUGALLM_PROXY_HOST", "127.0.0.1")
    if proxy_port == "4000" and proxy_host == "127.0.0.1":
        proxy_host = "litellm"
    base_url = f"http://{proxy_host}:{proxy_port}"
    headers = {
        "Authorization": "Bearer sk-sidecar-1",
        "Content-Type": "application/json"
    }

    success_all = True
    for alias, primary, fallbacks in groups:
        api_key_env = "os.environ/GOOGLE_API_KEY" if primary.startswith("gemini/") else "os.environ/OPENROUTER_API_KEY"
        provider = "gemini" if primary.startswith("gemini/") else "openrouter"
        
        payload = {
            "model_name": alias,
            "litellm_params": {
                "model": primary,
                "api_key": api_key_env,
                "llm_provider": provider,
                "max_tokens": 8192,
                "timeout": 35,
                "request_timeout": 35,
                "max_retries": 0
            },
            "model_info": {
                "supports_function_calling": True,
                "mode": "chat"
            }
        }
        # We do NOT embed fallbacks in litellm_params anymore

        # Try POST /model/update
        req = urllib.request.Request(
            f"{base_url}/model/update",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    pass
        except urllib.error.HTTPError as e:
            # If update fails, try POST /model/new
            req_new = urllib.request.Request(
                f"{base_url}/model/new",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            try:
                with urllib.request.urlopen(req_new, timeout=30) as resp_new:
                    if resp_new.status != 200:
                        log.error(f"Failed to create {alias}: HTTP {resp_new.status}")
                        success_all = False
            except Exception as inner_e:
                log.error(f"Failed to create {alias}: {inner_e}")
                success_all = False
        except Exception as e:
            log.error(f"Unexpected error updating {alias}: {e}")
            success_all = False

        if fallbacks:
            fallback_payload = {
                "model": alias,
                "fallback_models": fallbacks,
                "fallback_type": "general"
            }
            req_fallback = urllib.request.Request(
                f"{base_url}/fallback",
                data=json.dumps(fallback_payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            try:
                with urllib.request.urlopen(req_fallback, timeout=30) as resp_fb:
                    if resp_fb.status != 200:
                        log.error(f"Failed to register fallbacks for {alias}: HTTP {resp_fb.status}")
                        success_all = False
            except Exception as e:
                log.error(f"Error registering fallbacks for {alias}: {e}")
                success_all = False

    # Additionally push the local models so LiteLLM proxy DB router can resolve them as fallbacks
    for m in local_models_objs:
        if not m.get("model_name"):
            continue
        
        # Inject llm_provider if missing
        params = m.get("litellm_params", {})
        model_str = params.get("model", "")
        if "llm_provider" not in params:
            if "openai/" in model_str:
                params["llm_provider"] = "openai"
            elif "ollama/" in model_str or "ollama_chat/" in model_str:
                params["llm_provider"] = "ollama"
            m["litellm_params"] = params

        req_new = urllib.request.Request(
            f"{base_url}/model/new",
            data=json.dumps(m).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req_new, timeout=30) as resp_new:
                if resp_new.status != 200:
                    log.error(f"Failed to push local model {m['model_name']} to DB: HTTP {resp_new.status}")
        except Exception as e:
            log.error(f"Error pushing local model {m['model_name']} to DB: {e}")

    if success_all:
        log.info(f"✓ Pushed {len(groups)} dynamic models/aliases to DB via REST API.")
    return success_all


def _wait_for_litellm() -> bool:
    """Wait for LiteLLM to become available (up to 300 seconds).
    Uses /v1/models instead of /health because /health probes all backends
    (including unreachable ones like offline Qwen) causing long timeouts.
    """
    # Inside Docker: FRUGALLM_PROXY_PORT=4000, hostname=litellm
    # On host: defaults to 127.0.0.1:5050 (Gatekeeper port)
    proxy_port = os.getenv("FRUGALLM_PROXY_PORT", "5050")
    proxy_host = os.getenv("FRUGALLM_PROXY_HOST", "127.0.0.1")
    # If running in Docker (port 4000), use the Docker service name
    if proxy_port == "4000" and proxy_host == "127.0.0.1":
        proxy_host = "litellm"
    proxy_url = f"http://{proxy_host}:{proxy_port}/v1/models"

    log.info(f"⏳ Waiting for LiteLLM proxy at {proxy_url}...")
    for attempt in range(60):
        try:
            req = urllib.request.Request(
                proxy_url,
                headers={"Authorization": "Bearer sk-sidecar-1"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    log.info("✓ LiteLLM proxy is online.")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(5)

    log.error("✗ LiteLLM proxy did not become available after 300s.")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Main Discovery Loop (adapted from _background_model_fetch)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_and_register():
    """
    Polls OpenRouter for free models, classifies them, and writes the
    best ones to the dynamic_models.yaml include file.

    Adapted from router_server.py L122-176.
    """
    global _current_balanced, _current_reasoning

    # Fallback model IDs (must be free models!)
    FALLBACK_BALANCED = ["google/gemini-2.5-flash:free", "google/gemini-pro"]
    FALLBACK_REASONING = ["google/gemini-2.5-pro:free", "google/gemini-pro"]

    best_balanced = FALLBACK_BALANCED
    best_reasoning = FALLBACK_REASONING

    try:
        log.info("☀ Scanning OpenRouter for the best free models...")

        headers = {}
        if OPENROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

        req = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8")).get("data", [])

        gemini_data = []
        if os.getenv("GOOGLE_API_KEY"):
            log.info("☀ Scanning Google AI Studio for free tier models...")
            gemini_req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models?key={os.getenv('GOOGLE_API_KEY')}")
            try:
                with urllib.request.urlopen(gemini_req, timeout=15) as gemini_resp:
                    gemini_raw = json.loads(gemini_resp.read().decode("utf-8")).get("models", [])
                    for gm in gemini_raw:
                        methods = gm.get("supportedGenerationMethods", [])
                        if "generateContent" not in methods:
                            log.debug(f"Discarding model {gm.get('name')} (unsupported methods: {methods})")
                            continue
                        
                        m_name = gm.get("name", "").replace("models/", "")
                        norm_m = {
                            "id": f"gemini/{m_name}",
                            "name": gm.get("displayName", ""),
                            "description": gm.get("description", ""),
                            "context_length": gm.get("inputTokenLimit", 0),
                            "supported_parameters": ["tools", "response_format"],
                            "pricing": {"prompt": "0", "completion": "0"},
                            "created": int(time.time()),
                            "top_provider": {"max_completion_tokens": gm.get("outputTokenLimit", 8192)}
                        }
                        if gm.get("thinking"):
                            norm_m["description"] += " reasoning"
                        gemini_data.append(norm_m)
            except Exception as e:
                log.warning(f"Google AI Studio fetch failed: {e}")
                
        # Merge both pools
        data.extend(gemini_data)

        # Filter free models that support tool use (or are reasoning models)
        free = []
        for m in data:
            params = m.get("supported_parameters", [])
            if "tools" not in params and not _is_reasoning_model(m):
                continue
                
            p = m.get("pricing", {})
            try:
                if (
                    float(p.get("prompt", "1")) == 0
                    and float(p.get("completion", "1")) == 0
                ):
                    free.append(m)
            except (ValueError, TypeError):
                continue

        # Sort all free models by a ranked composite score:
        # 1. Context length
        # 2. More supported parameters (e.g. tools, reasoning, response_format)
        # 3. Most recently created
        # 4. Larger output context (max_completion_tokens)
        free.sort(
            key=lambda x: (
                x.get("context_length", 0),
                len(x.get("supported_parameters", [])),
                x.get("created", 0),
                (x.get("top_provider") or {}).get("max_completion_tokens") or 0
            ),
            reverse=True
        )

        if free:
            # We filter for 128k+ context for both balanced and reasoning pools
            free_balanced_models = [m for m in free if m.get("context_length", 0) >= 128000]
            if not free_balanced_models:
                free_balanced_models = free  # Fallback to any context length if none >= 128k
                
            best_balanced = [m["id"] for m in free_balanced_models]
            log.info(f"✓ free_balanced candidates: {len(best_balanced)} models")

            # Agnostically search the pool for reasoning models with >=128k context
            reasoning_models = [
                m for m in free
                if _is_reasoning_model(m) and m.get("context_length", 0) >= 128000
            ]

            if reasoning_models:
                best_reasoning = [m["id"] for m in reasoning_models]
                log.info(f"✓ free_reasoning candidates: {best_reasoning}")
            else:
                log.info(
                    "ℹ No dedicated free reasoning model found with >=128k context. "
                    "Falling back reasoning route to balanced pool."
                )
                best_reasoning = best_balanced

        log.info(f"Free pool size: {len(free)} models available.")

    except Exception as e:
        log.warning(f"OpenRouter fetch failed, using fallbacks: {e}")

    # Only update config if the models have changed
    if best_balanced != _current_balanced or best_reasoning != _current_reasoning:
        if _push_models_to_db(best_balanced, best_reasoning):
            _current_balanced = best_balanced
            _current_reasoning = best_reasoning
            log.info("✓ Model roster updated in LiteLLM DB. Proxy will sync on next poll.")
        else:
            log.error("Failed to push dynamic models to database.")
    else:
        log.info("No roster changes needed.")


def main():
    """Main daemon loop."""
    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║     HERMES DYNAMIC ROSTER SIDECAR                           ║
╠══════════════════════════════════════════════════════════════╣
║  Poll Interval:   {POLL_INTERVAL}s ({POLL_INTERVAL // 60} minutes)
║  OpenRouter Key:  {'✓ Present' if OPENROUTER_API_KEY else '✗ MISSING'}
║  Waiting for LiteLLM to come online...
╚══════════════════════════════════════════════════════════════╝
"""
    )

    # Wait for LiteLLM to be ready before starting the loop
    if not _wait_for_litellm():
        log.error("Exiting: LiteLLM proxy is not available.")
        sys.exit(1)

    # Initial discovery
    discover_and_register()

    # Polling loop
    while True:
        try:
            time.sleep(POLL_INTERVAL)
            discover_and_register()
        except KeyboardInterrupt:
            log.info("Shutting down sidecar.")
            break
        except Exception as e:
            log.error(f"Unexpected error in sidecar loop: {e}")
            time.sleep(30)  # Back off on unexpected errors


if __name__ == "__main__":
    main()
