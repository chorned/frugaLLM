#!/usr/bin/env python3
"""
Hermes Tiered Router — Cost-Optimized Brain Selection
======================================================

Routes requests across backends with Horizontal Model Rotation.
If a free cloud model returns a 429 (Rate Limit), this script will
instantly bench that model, select the next best free model from the 
cluster, and retry the exact same prompt transparently.

Routing matrix (profile → brain):
  documenter / devops  →  Path A  (balanced free pool)
  engineer + small     →  Path D  (local) → fallback to Path B
  engineer + large     →  Path B  (reasoning free pool)
  any + allow_pro      →  Path C  (pro escalation, last resort)
"""

from __future__ import annotations

import os
import re
import sys
import time
import textwrap
from pathlib import Path
from typing import Optional

import requests
from openai import OpenAI
import openai

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
if not OPENROUTER_API_KEY:
    print("[Hermes Router] WARNING: OPENROUTER_API_KEY is not set.", file=sys.stderr)

# ─── Clients ──────────────────────────────────────────────────────────────────
or_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

LOCAL_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LOCAL_OLLAMA_MODEL = "hermes:latest"

# ─── Dynamic Rotation State ───────────────────────────────────────────────────
_FREE_MODEL_POOL = []  # Holds all discovered free models sorted by context length
_COOLDOWNS = {}        # Dict mapping model_id -> timestamp (float) of expiration

# ─── Config Loader ────────────────────────────────────────────────────────────
def _load_routing_config() -> dict:
    defaults = {"current_profile": "engineer", "task_size": "small", "allow_pro": False}
    if not _CONFIG_PATH.exists():
        return defaults
    try:
        text = _CONFIG_PATH.read_text()
        m_profile = re.search(r"^\s*current_profile:\s*(\S+)", text, re.MULTILINE)
        m_size = re.search(r"^\s*task_size:\s*(\S+)", text, re.MULTILINE)
        m_pro = re.search(r"^\s*allow_pro:\s*(\S+)", text, re.MULTILINE)
        if m_profile: defaults["current_profile"] = m_profile.group(1).strip().strip("'\"")
        if m_size: defaults["task_size"] = m_size.group(1).strip().strip("'\"")
        if m_pro: defaults["allow_pro"] = m_pro.group(1).strip().strip("'\"").lower() in ("true", "yes", "1")
    except Exception as e:
        print(f"[Hermes Router] Config parse warning: {e}", file=sys.stderr)
    return defaults

def _load_soul_prompt() -> str:
    if _SOUL_PATH.exists():
        return _SOUL_PATH.read_text().strip()
    return ""

# ─── Dynamic Free Model Discovery ────────────────────────────────────────────
def get_best_free_models() -> dict[str, str]:
    fallback_models = {
        "balanced_free": "google/gemini-2.5-flash:free",
        "reasoning_free": "deepseek/deepseek-r1:free",
        "pro_paid": "anthropic/claude-3.7-sonnet",
    }
    try:
        print("[Hermes Router] Scanning OpenRouter for today's free model promotions...")
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"} if OPENROUTER_API_KEY else {},
            timeout=10,
        )
        resp.raise_for_status()
        models_data = resp.json().get("data", [])

        free_models = []
        for m in models_data:
            pricing = m.get("pricing", {})
            try:
                if float(pricing.get("prompt", "1")) == 0 and float(pricing.get("completion", "1")) == 0:
                    free_models.append(m)
            except (ValueError, TypeError):
                continue

        free_models.sort(key=lambda x: x.get("context_length", 0), reverse=True)
        
        # Save to global pool for rotation
        global _FREE_MODEL_POOL
        _FREE_MODEL_POOL = free_models

        if free_models:
            best_id = free_models[0]["id"]
            print(f"[Hermes Router] ✓ Dynamic balanced_free: {best_id} ({free_models[0].get('context_length', '?'):,} ctx)")
            fallback_models["balanced_free"] = best_id

            reasoning_candidates = [m for m in free_models if any(kw in m["id"].lower() for kw in ("deepseek", "r1", "reason", "think"))]
            if reasoning_candidates:
                reasoning_candidates.sort(key=lambda x: x.get("context_length", 0), reverse=True)
                reasoning_id = reasoning_candidates[0]["id"]
                print(f"[Hermes Router] ✓ Dynamic reasoning_free: {reasoning_id}")
                fallback_models["reasoning_free"] = reasoning_id

        print(f"[Hermes Router] Model roster locked ({len(_FREE_MODEL_POOL)} free models in cluster pool).")
        return fallback_models
    except Exception as e:
        print(f"[Hermes Router] Dynamic fetch failed: {e}", file=sys.stderr)
        return fallback_models

MODELS = get_best_free_models()

def get_available_free_model(exclude_model: Optional[str] = None, reasoning_only: bool = False) -> Optional[str]:
    """Finds the best free model that is NOT currently benched."""
    now = time.time()
    
    # Housekeeping
    for k in list(_COOLDOWNS.keys()):
        if _COOLDOWNS[k] < now:
            del _COOLDOWNS[k]
            print(f"[Hermes Router] ♻️  Model {k} cooldown expired. Re-adding to pool.")

    for m in _FREE_MODEL_POOL:
        m_id = m["id"]
        if m_id == exclude_model:
            continue
        if m_id in _COOLDOWNS and _COOLDOWNS[m_id] > now:
            continue
        if reasoning_only and not any(kw in m_id.lower() for kw in ("deepseek", "r1", "reason", "think")):
            continue
        return m_id
    return None

# ─── Escalation & Validation ─────────────────────────────────────────────────
_ESCALATION_KEYWORDS = ["//escalate", "hey hermes, use your pro brain"]
_consecutive_failures: int = 0
_FAILURE_ESCALATION_THRESHOLD: int = 3

def _should_escalate_by_keyword(prompt: str) -> bool:
    return any(kw in prompt.lower() for kw in _ESCALATION_KEYWORDS)

def reset_failure_counter():
    global _consecutive_failures
    _consecutive_failures = 0

def record_failure():
    global _consecutive_failures
    _consecutive_failures += 1
    if _consecutive_failures >= _FAILURE_ESCALATION_THRESHOLD:
        print(f"[Hermes Router] ⚠ {_consecutive_failures} consecutive failures — escalation armed.")
        return True
    return False

# ─── Core Router ──────────────────────────────────────────────────────────────
def ask_hermes(prompt: str, profile: Optional[str] = None, task_size: Optional[str] = None, allow_pro: Optional[bool] = None, system_prompt: Optional[str] = None) -> str:
    config = _load_routing_config()
    profile = profile or config["current_profile"]
    task_size = task_size or config["task_size"]
    allow_pro = config["allow_pro"] if allow_pro is None else allow_pro
    system_prompt = system_prompt or _load_soul_prompt()

    if _should_escalate_by_keyword(prompt):
        print("[Hermes Router] 🔑 Keyword escalation detected in prompt.")
        allow_pro = True
    if _consecutive_failures >= _FAILURE_ESCALATION_THRESHOLD:
        print(f"[Hermes Router] 🔄 Validation loop escalation ({_consecutive_failures} failures).")
        allow_pro = True

    # ── 1. LOCAL ROUTING ──
    if profile == "engineer" and task_size == "small":
        print(f"[Hermes Router] 🖥  Routing to Local Ollama ({LOCAL_OLLAMA_MODEL})...")
        try:
            payload = {"model": LOCAL_OLLAMA_MODEL, "prompt": prompt, "stream": False}
            if system_prompt: payload["system"] = system_prompt
            resp = requests.post(LOCAL_OLLAMA_URL, json=payload, timeout=3)
            resp.raise_for_status()
            result = resp.json().get("response", "")
            if result:
                reset_failure_counter()
                return result
        except requests.exceptions.RequestException as e:
            print(f"[Hermes Router] ⚡ Local Ollama failed ({e}). Falling back to cloud...")

    # ── 2. CLOUD ROUTING (WITH ROTATION) ──
    is_reasoning_task = (profile not in ("documenter", "devops"))
    
    # Grab the best available model from the ledger
    current_model = get_available_free_model(reasoning_only=is_reasoning_task)
    if not current_model:
        current_model = MODELS["reasoning_free"] if is_reasoning_task else MODELS["balanced_free"]

    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    max_attempts = 4
    for attempt in range(max_attempts):
        print(f"[Hermes Router] ☁️  Attempt {attempt+1}/{max_attempts}: Cloud Brain ({current_model})...")
        try:
            response = or_client.chat.completions.create(
                model=current_model,
                messages=messages,
            )
            result = response.choices[0].message.content
            if not is_valid_result(result):
                raise ValueError("Free model response failed validation check.")
            reset_failure_counter()
            return result

        except openai.RateLimitError as e:
            # Extract cooldown time directly from the openai exception response headers
            retry_after = 60
            if getattr(e, 'response', None) is not None:
                headers = e.response.headers
                if 'retry-after' in headers:
                    retry_after = int(headers['retry-after'])

            print(f"[Hermes Router] ⏳ Rate limited on {current_model}! Benched for {retry_after}s.")
            _COOLDOWNS[current_model] = time.time() + retry_after
            
            # Pivot to next model
            next_model = get_available_free_model(current_model, reasoning_only=is_reasoning_task)
            if not next_model:
                print(f"[Hermes Router] 💥 All free models in this tier are exhausted!")
                break # Break to Pro Escalation
                
            print(f"[Hermes Router] 🔄 Rotating to next available free model: {next_model}")
            current_model = next_model

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ("context", "too long", "400", "token limit")):
                print(f"[Hermes Router] 📏 Context overflow detected on {current_model}.")
                allow_pro = True
                break # Break out of rotation loop to trigger Pro Escalation immediately
            
            print(f"[Hermes Router] ✗ Free cloud failed unexpectedly: {e}")
            break # Break to Pro Escalation

    # ── 3. PRO ESCALATION ──
    if allow_pro:
        pro_model = MODELS["pro_paid"]
        print(f"[Hermes Router] 🚀 ESCALATING TO PRO: {pro_model} (PAID TIER)")
        try:
            pro_messages = [
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\n--- ESCALATION NOTICE ---\nYou were escalated. Be precise." if system_prompt else "You are escalated. Be precise."
                },
                {"role": "user", "content": prompt},
            ]
            pro_response = or_client.chat.completions.create(model=pro_model, messages=pro_messages)
            reset_failure_counter()
            return pro_response.choices[0].message.content
        except Exception as pro_e:
            return f"[Hermes Router] CRITICAL: Pro model failed.\n  Pro error: {pro_e}"
    else:
        return f"[Hermes Router] Task failed across cluster.\n  Escalation to Pro tier is DENIED by configuration."

def is_valid_result(text: Optional[str]) -> bool:
    if not text or len(text.strip()) < 2: return False
    return True

# ─── Convenience Wrappers ─────────────────────────────────────────────────────
def ask_local(prompt: str, system_prompt: Optional[str] = None) -> str:
    print(f"[Hermes Router] 🖥  Direct local call ({LOCAL_OLLAMA_MODEL})...")
    payload = {"model": LOCAL_OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if system_prompt: payload["system"] = system_prompt
    resp = requests.post(LOCAL_OLLAMA_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("response", "")

def ask_cloud(prompt: str, model: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
    model = model or MODELS["balanced_free"]
    print(f"[Hermes Router] ☁️  Direct cloud call ({model})...")
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = or_client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

# ─── CLI Entry Point ─────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Hermes Tiered Router — Cost-optimized LLM backend selector with Auto-Rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", help="The prompt text (or use --stdin)")
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    parser.add_argument("--profile", "-p", choices=["engineer", "documenter", "devops"], help="Override routing profile")
    parser.add_argument("--size", "-s", choices=["small", "large"], help="Override task size")
    parser.add_argument("--pro", action="store_true", help="Allow pro escalation for this request")
    parser.add_argument("--model", "-m", help="Force a specific model (bypasses routing)")
    parser.add_argument("--models", action="store_true", help="Print the current model roster and exit")

    args = parser.parse_args()

    if args.models:
        print("\n[Hermes Router] Current Model Roster:")
        for role, model_id in MODELS.items():
            marker = "💰" if role == "pro_paid" else "🆓"
            print(f"  {marker} {role:18s} → {model_id}")
        print(f"\n[Hermes Router] {len(_FREE_MODEL_POOL)} models loaded in fallback cluster pool.")
        sys.exit(0)

    if args.stdin: prompt = sys.stdin.read().strip()
    elif args.prompt: prompt = args.prompt
    else: parser.print_help(); sys.exit(1)

    if args.model: result = ask_cloud(prompt, model=args.model)
    else: result = ask_hermes(prompt=prompt, profile=args.profile, task_size=args.size, allow_pro=args.pro)

    print("\n" + "─" * 60)
    print(result)

if __name__ == "__main__":
    main()