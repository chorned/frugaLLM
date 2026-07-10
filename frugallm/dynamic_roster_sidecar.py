#!/usr/bin/env python3
"""
FrugaLLM Dynamic Roster Sidecar
=================================

Lightweight daemon that polls OpenRouter for free models every 5 minutes
and writes the best candidates to a YAML include file consumed by LiteLLM.

Strategy: Write dynamic models to <config_dir>/dynamic_models.yaml which is
included by the main litellm_config.yaml. After updating the file, send
SIGHUP to LiteLLM to trigger a graceful config reload.

Usage:
  python -m frugallm.dynamic_roster_sidecar

Environment Variables:
  FRUGALLM_CONFIG_DIR       Directory containing litellm_config.yaml (default: ./config)
  OPENROUTER_API_KEY        OpenRouter API key for model inference
  OPENROUTER_MANAGEMENT_KEY Separate key for polling the /models API (optional)
  FRUGALLM_POLL_INTERVAL    Poll interval in seconds (default: 300)
  FRUGALLM_PROXY_PORT       LiteLLM proxy port (default: 4000)
  FRUGALLM_MASTER_KEY       LiteLLM master key (default: sk-frugallm-master)
  FRUGALLM_LOCAL_MODEL      Local Ollama model name (default: llama3.2:latest)
  FRUGALLM_LOCAL_URL        Local Ollama URL (default: http://127.0.0.1:11434)
  FRUGALLM_GPU_MODEL        GPU node model path (default: none — disabled)
  FRUGALLM_GPU_URL          GPU node base URL (default: none — disabled)
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
_CONFIG_DIR = Path(os.getenv("FRUGALLM_CONFIG_DIR", "./config"))
_DYNAMIC_MODELS_PATH = _CONFIG_DIR / "dynamic_models.yaml"

POLL_INTERVAL = int(os.getenv("FRUGALLM_POLL_INTERVAL", "300"))
PROXY_PORT = int(os.getenv("FRUGALLM_PROXY_PORT", "4000"))
MASTER_KEY = os.getenv("FRUGALLM_MASTER_KEY", "sk-frugallm-master")
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Local fallback configuration
LOCAL_MODEL = os.getenv("FRUGALLM_LOCAL_MODEL", "llama3.2:latest")
LOCAL_URL = os.getenv("FRUGALLM_LOCAL_URL", "http://127.0.0.1:11434")

# Optional GPU node configuration
GPU_MODEL = os.getenv("FRUGALLM_GPU_MODEL", "")
GPU_URL = os.getenv("FRUGALLM_GPU_URL", "")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FrugaLLM Sidecar] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frugallm-roster-sidecar")

# ─── Environment Loading ────────────────────────────────────────────────────
def _load_env_file(env_path: Path | None = None):
    """Load .env file into os.environ if it exists."""
    if env_path is None:
        env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
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


_load_env_file()

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
# Model Classification Logic
# ═══════════════════════════════════════════════════════════════════════════════

def _is_reasoning_model(model_data: dict) -> bool:
    """
    Agnostic heuristic to detect reasoning models based on metadata.

    Examines model ID, name, and description for reasoning-related keywords.
    This allows automatic classification without maintaining a hardcoded list.
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
# YAML File Writer
# ═══════════════════════════════════════════════════════════════════════════════

def _write_dynamic_models(balanced_ids: list[str], reasoning_ids: list[str]) -> bool:
    """
    Write the dynamic model definitions to a YAML include file using fallback chaining.

    Each model gets a numbered alias (free_balanced, free_balanced_2, etc.) with
    a linear fallback chain. The last entry in each chain falls back to a local
    backup model.
    """
    def _litellm_model(or_id: str) -> str:
        if or_id.startswith("openrouter/") or or_id.startswith("ollama/") or or_id.startswith("openai/"):
            return or_id
        return f"openrouter/{or_id}"

    yaml_lines = [
        "# ═══════════════════════════════════════════════════════════════════════════════",
        "# AUTO-GENERATED by FrugaLLM Dynamic Roster Sidecar — DO NOT EDIT MANUALLY",
        f"# Last updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "# ═══════════════════════════════════════════════════════════════════════════════",
        "",
        "model_list:"
    ]

    fallbacks = [
        '    - {"auto": ["free_balanced"]}',
        '    - {"reasoning": ["free_reasoning"]}',
        '    - {"local": ["free_balanced"]}'
    ]

    # Generate Balanced Chain
    yaml_lines.append("  # ── Balanced Chain ──")
    for i, b_id in enumerate(balanced_ids):
        model_name = "free_balanced" if i == 0 else f"free_balanced_{i+1}"
        yaml_lines.extend([
            f"  - model_name: {model_name}",
            "    litellm_params:",
            f"      model: {_litellm_model(b_id)}",
            "      api_key: os.environ/OPENROUTER_API_KEY",
            "      timeout: 300",
            "      max_retries: 0",
            ""
        ])
        if i < len(balanced_ids) - 1:
            next_model = f"free_balanced_{i+2}"
            fallbacks.append(f'    - {{"{model_name}": ["{next_model}"]}}')
        else:
            fallbacks.append(f'    - {{"{model_name}": ["free_balanced_backup"]}}')

    # Balanced backup — local Ollama
    yaml_lines.extend([
        "  - model_name: free_balanced_backup",
        "    litellm_params:",
        f"      model: ollama/{LOCAL_MODEL}",
        f"      api_base: {LOCAL_URL}",
        "      timeout: 300",
        "      max_retries: 0",
        ""
    ])

    # Generate Reasoning Chain
    yaml_lines.append("  # ── Reasoning Chain ──")
    for i, r_id in enumerate(reasoning_ids):
        model_name = "free_reasoning" if i == 0 else f"free_reasoning_{i+1}"
        yaml_lines.extend([
            f"  - model_name: {model_name}",
            "    litellm_params:",
            f"      model: {_litellm_model(r_id)}",
            "      api_key: os.environ/OPENROUTER_API_KEY",
            "      timeout: 300",
            "      max_retries: 0",
            ""
        ])
        if i < len(reasoning_ids) - 1:
            next_model = f"free_reasoning_{i+2}"
            fallbacks.append(f'    - {{"{model_name}": ["{next_model}"]}}')
        else:
            fallbacks.append(f'    - {{"{model_name}": ["free_reasoning_backup"]}}')

    # Reasoning backup — GPU node if configured, otherwise local Ollama
    if GPU_MODEL and GPU_URL:
        yaml_lines.extend([
            "  - model_name: free_reasoning_backup",
            "    litellm_params:",
            f"      model: openai/{GPU_MODEL}",
            f"      api_base: {GPU_URL}",
            '      api_key: "na"',
            "      timeout: 300",
            "      max_retries: 0",
            ""
        ])
    else:
        yaml_lines.extend([
            "  - model_name: free_reasoning_backup",
            "    litellm_params:",
            f"      model: ollama/{LOCAL_MODEL}",
            f"      api_base: {LOCAL_URL}",
            "      timeout: 300",
            "      max_retries: 0",
            ""
        ])

    # Add router_settings fallbacks
    yaml_lines.extend([
        "router_settings:",
        "  fallbacks:"
    ])
    yaml_lines.extend(fallbacks)
    yaml_lines.append("")

    yaml_content = "\n".join(yaml_lines)

    try:
        # Atomic write: write to temp, then rename
        tmp_path = _DYNAMIC_MODELS_PATH.with_suffix(".tmp")
        tmp_path.write_text(yaml_content)
        tmp_path.rename(_DYNAMIC_MODELS_PATH)
        log.info(f"✓ Wrote dynamic_models.yaml (balanced={len(balanced_ids)}, reasoning={len(reasoning_ids)})")
        return True
    except Exception as e:
        log.error(f"Failed to write dynamic_models.yaml: {e}")
        return False


def _restart_litellm():
    """
    Restart the LiteLLM proxy to pick up the new config include.
    Uses SIGHUP for a zero-downtime config reload, falling back to process restart.
    """
    try:
        if _signal_litellm():
            log.info("✓ LiteLLM config reloaded via SIGHUP.")
            return True
        else:
            log.warning("SIGHUP failed — LiteLLM process not found. You may need to restart manually.")
            return False
    except Exception as e:
        log.error(f"Restart failed entirely: {e}")
        return False


def _signal_litellm() -> bool:
    """Find LiteLLM PID and send SIGHUP for graceful config reload."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "litellm.*--config"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for pid_str in result.stdout.strip().split("\n"):
                pid = int(pid_str.strip())
                os.kill(pid, signal.SIGHUP)
                log.info(f"✓ Sent SIGHUP to LiteLLM PID {pid}")
            return True
        else:
            log.warning("Could not find LiteLLM process for SIGHUP.")
            return False
    except Exception as e:
        log.error(f"Failed to signal LiteLLM: {e}")
        return False


def _wait_for_litellm() -> bool:
    """Wait for LiteLLM to become available (up to 300 seconds)."""
    log.info("⏳ Waiting for LiteLLM proxy to become available...")
    for attempt in range(60):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PROXY_PORT}/v1/models",
                headers={"Authorization": f"Bearer {MASTER_KEY}"},
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
# Main Discovery Loop
# ═══════════════════════════════════════════════════════════════════════════════

def discover_and_register():
    """
    Polls OpenRouter for free models, classifies them, and writes the
    best ones to the dynamic_models.yaml include file.

    Selection criteria:
      - Must be free (prompt and completion cost = 0)
      - Must support tool use
      - Prefers models with >= 256k context window
      - Separates balanced (general) from reasoning models using heuristics
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

        # Filter free models that support tool use
        free = []
        for m in data:
            params = m.get("supported_parameters", [])
            if "tools" not in params:
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

        # Sort all free models by created to find the newest ones
        free.sort(key=lambda x: x.get("created", 0), reverse=True)

        if free:
            # We filter for 256k+ context for both balanced and reasoning pools
            free_balanced_models = [m for m in free if m.get("context_length", 0) >= 256000]
            if not free_balanced_models:
                free_balanced_models = free  # Fallback to any context length if none >= 256k

            best_balanced = [m["id"] for m in free_balanced_models]
            log.info(f"✓ free_balanced candidates: {len(best_balanced)} models")

            # Agnostically search the pool for reasoning models with >=256k context
            reasoning_models = [
                m for m in free
                if _is_reasoning_model(m) and m.get("context_length", 0) >= 256000
            ]

            if reasoning_models:
                best_reasoning = [m["id"] for m in reasoning_models]
                log.info(f"✓ free_reasoning candidates: {best_reasoning}")
            else:
                log.info(
                    "ℹ No dedicated free reasoning model found with >=256k context. "
                    "Falling back reasoning route to balanced pool."
                )
                best_reasoning = best_balanced

        log.info(f"Free pool size: {len(free)} models available.")

    except Exception as e:
        log.warning(f"OpenRouter fetch failed, using fallbacks: {e}")

    # Only update config and restart if the models have changed
    if best_balanced != _current_balanced or best_reasoning != _current_reasoning:
        if _write_dynamic_models(best_balanced, best_reasoning):
            _current_balanced = best_balanced
            _current_reasoning = best_reasoning
            log.info("🔄 Roster changed — restarting LiteLLM to load new models...")
            _restart_litellm()
            # Wait for LiteLLM to come back online after restart
            time.sleep(5)
            _wait_for_litellm()
        else:
            log.error("Failed to write dynamic models — skipping restart.")
    else:
        log.info("No roster changes needed.")


def main():
    """Main daemon loop."""
    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║     FRUGALLM DYNAMIC ROSTER SIDECAR                         ║
╠══════════════════════════════════════════════════════════════╣
║  Config Target:   {_DYNAMIC_MODELS_PATH}
║  Poll Interval:   {POLL_INTERVAL}s ({POLL_INTERVAL // 60} minutes)
║  Proxy Port:      {PROXY_PORT}
║  OpenRouter Key:  {'✓ Present' if OPENROUTER_API_KEY else '✗ MISSING'}
║  Local Fallback:  {LOCAL_MODEL} @ {LOCAL_URL}
║  GPU Node:        {GPU_URL or 'Not configured'}
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
