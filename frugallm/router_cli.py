#!/usr/bin/env python3
"""
FrugaLLM Router CLI — Intelligent Routing with RouteLLM
=======================================================

Routes prompts using the RouteLLM Matrix Factorization (MF) model to determine
difficulty. 
- Complex prompts: Routed to Google AI Studio via REST API
- Simple prompts: Fallback to local proxy pipeline

Usage:
  python -m frugallm.router_cli "What is the meaning of life?"
  echo "Explain quantum computing" | python -m frugallm.router_cli --stdin
  python -m frugallm.router_cli --models
  python -m frugallm.router_cli --threshold 0.5 "Prompt"
"""

import argparse
import asyncio
import json
import os
import sys
import logging
import aiohttp

# 4. RouteLLM Initialization Quirks: Mock OPENAI_API_KEY if not set
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("router_cli")

PROXY_URL = os.getenv("FRUGALLM_PROXY_URL", "http://127.0.0.1:5050")
PROXY_API_KEY = os.getenv("FRUGALLM_MASTER_KEY", "sk-sidecar-1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_THRESHOLD = float(os.getenv("ROUTELLM_THRESHOLD", "0.1159"))

# Baseline pricing for Claude 3.5 Sonnet (for telemetry cost comparison)
BASELINE_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
BASELINE_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000

try:
    from routellm.routers.routers import get_router
except ImportError:
    get_router = None

# Global aiohttp session for connection pooling in production proxy environments
_GLOBAL_SESSION = None

async def get_session():
    """
    Ensure the ClientSession is initialized safely inside a running asyncio event loop.
    """
    global _GLOBAL_SESSION
    if _GLOBAL_SESSION is None or _GLOBAL_SESSION.closed:
        _GLOBAL_SESSION = aiohttp.ClientSession()
    return _GLOBAL_SESSION

async def close_session():
    global _GLOBAL_SESSION
    if _GLOBAL_SESSION is not None and not _GLOBAL_SESSION.closed:
        await _GLOBAL_SESSION.close()

async def check_proxy_health():
    """Pings the proxy to retrieve its active model roster and health stats."""
    headers = {"Authorization": f"Bearer {PROXY_API_KEY}"}
    session = await get_session()
    try:
        async with session.get(f"{PROXY_URL}/health", headers=headers, timeout=30) as resp:
            data = await resp.json()
            print("\n[FrugaLLM CLI] LiteLLM Gateway Health Report:")
            print(f"  Status: {data.get('status', 'Unknown')}")

            if "healthy_endpoints" in data:
                print(f"\n  Healthy Endpoints: {len(data.get('healthy_endpoints', []))}")
                for ep in data.get("healthy_endpoints", []):
                    model = ep.get("model", "unknown")
                    print(f"    ✓ {model}")

            if "unhealthy_endpoints" in data:
                unhealthy = data.get("unhealthy_endpoints", [])
                if unhealthy:
                    print(f"\n  Unhealthy Endpoints: {len(unhealthy)}")
                    for ep in unhealthy:
                        model = ep.get("model", "unknown")
                        print(f"    ✗ {model}")

    except Exception as e:
        print(f"[FrugaLLM CLI] Error: LiteLLM gateway is offline at {PROXY_URL} ({e})")
        print("Make sure LiteLLM is running!")
    finally:
        await close_session()
    sys.exit(0)

def format_gemini_payload(prompt_str: str) -> dict:
    """
    Translates standard OpenAI messages array (if valid JSON) into Gemini's format.
    Merges consecutive identical roles to prevent 400 errors.
    """
    payload = {"contents": []}
    
    try:
        messages = json.loads(prompt_str)
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict) and "role" in messages[0]:
            system_instruction_text = []
            merged_messages = []
            
            # Extract system messages and merge consecutive user/assistant roles
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system":
                    system_instruction_text.append(content)
                elif role in ("user", "assistant"):
                    if merged_messages and merged_messages[-1]["role"] == role:
                        # Merge consecutive messages with the same role
                        merged_messages[-1]["content"] += "\n\n" + content
                    else:
                        merged_messages.append({"role": role, "content": content})
            
            # Build the payload from merged messages
            for msg in merged_messages:
                gemini_role = "user" if msg["role"] == "user" else "model"
                payload["contents"].append({
                    "role": gemini_role,
                    "parts": [{"text": msg["content"]}]
                })
                    
            if system_instruction_text:
                # Correct Gemini system_instruction schema: {"parts": [{"text": "..."}]}
                payload["system_instruction"] = {
                    "parts": [{"text": "\n\n".join(system_instruction_text)}]
                }
            
            if not payload["contents"]: # fallback if no user/assistant messages
                payload["contents"].append({"parts": [{"text": prompt_str}]})
                
            return payload
    except json.JSONDecodeError:
        pass # Not a JSON list, fall through to raw string
        
    # Raw string behavior
    payload["contents"].append({
        "role": "user",
        "parts": [{"text": prompt_str}]
    })
    return payload

from pathlib import Path

async def ask_strong_model(prompt: str, target_model: str) -> bool:
    """
    Executes a direct POST request to Google AI Studio for the strong model asynchronously.
    Resolves the Gemini model name from the FRUGALLM_STRONG_MODEL env var (set by sidecar
    or operator), translating the 'gemini/' prefix to the native 'models/' prefix.
    Returns True if successful, False if rate limited or server error.
    """
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY is not set. Falling back to proxy pipeline.")
        return False

    strong_model_name = "models/gemini-flash-latest"  # Baseline fallback
    env_model = os.environ.get("FRUGALLM_STRONG_MODEL", "")
    if env_model:
        if env_model.startswith("gemini/"):
            strong_model_name = env_model.replace("gemini/", "models/")
        elif env_model.startswith("models/"):
            strong_model_name = env_model
        else:
            strong_model_name = f"models/{env_model}"

    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/{strong_model_name}:generateContent"
    
    payload = format_gemini_payload(prompt)
    
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    
    log.info(f"Sending prompt to STRONG model ({strong_model_name}) via REST API...")
    session = await get_session()
    try:
        async with session.post(gemini_url, headers=headers, json=payload, timeout=60) as response:
            
            # Catch 400 (Bad Request), 429 (Rate Limit), and 5xx (Server Errors)
            if response.status in (400, 429) or response.status >= 500:
                error_body = await response.text()
                log.warning(f"Strong model failed with HTTP {response.status}. Falling back to proxy. Body: {error_body}")
                return False
                
            response.raise_for_status()
            data = await response.json()
            
            candidates = data.get("candidates", [])
            if not candidates:
                log.error("Google AI Studio response contained no candidates (potential safety block). Falling back to proxy.")
                return False
            
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")
            
            if finish_reason == "SAFETY":
                log.warning("Google AI Studio rejected the prompt due to SAFETY. Falling back to proxy.")
                return False
                
            try:
                text = candidate["content"]["parts"][0]["text"]
                log.info("Success! Model: gemini-flash-latest (Google AI Studio REST)")
                print("─" * 60)
                print(text)
                return True
            except (KeyError, IndexError):
                log.error("Unexpected response structure from Google AI Studio. Falling back to proxy.")
                return False
                
    except aiohttp.ClientError as e:
        log.error(f"Request to Google AI Studio failed: {e}. Falling back to proxy.")
        return False
    except asyncio.TimeoutError:
        log.error("Request to Google AI Studio timed out. Falling back to proxy.")
        return False

async def ask_proxy(prompt: str, target_model: str):
    """Packages the terminal prompt and sends it to the LiteLLM gateway."""
    
    try:
        parsed_prompt = json.loads(prompt)
        if isinstance(parsed_prompt, list) and all(isinstance(m, dict) and "role" in m for m in parsed_prompt):
            messages = parsed_prompt
        else:
            messages = [{"role": "user", "content": prompt}]
    except json.JSONDecodeError:
        messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "metadata": {
            "source": "frugallm-cli",
            "autonomy_tier": "user_guided",
            "tags": ["cli", target_model],
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PROXY_API_KEY}",
    }

    session = await get_session()
    try:
        log.info(f"Sending prompt to WEAK model via LiteLLM gateway (Target: {target_model})...")
        async with session.post(f"{PROXY_URL}/v1/chat/completions", json=payload, headers=headers, timeout=300) as resp:
            if resp.status >= 400:
                error_msg = await resp.text()
                log.error(f"Gateway returned an error: HTTP {resp.status}")
                log.error(error_msg)
                return

            result = await resp.json()

            text = result["choices"][0]["message"]["content"]
            model_used = result.get("model", "unknown")
            usage = result.get("usage", {})

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            baseline_cost = (
                input_tokens * BASELINE_INPUT_COST_PER_TOKEN
                + output_tokens * BASELINE_OUTPUT_COST_PER_TOKEN
            )

            log.info(
                f"Success! Model: {model_used} "
                f"(tokens: {input_tokens}→{output_tokens}, "
                f"baseline cost: ${baseline_cost:.6f})"
            )
            print("─" * 60)
            print(text)

    except aiohttp.ClientError as e:
        log.error(f"Could not connect to gateway at {PROXY_URL}: {e}")
        log.error("Make sure LiteLLM is running!")
    except asyncio.TimeoutError:
        log.error("Request to LiteLLM gateway timed out.")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        raise

async def evaluate_prompt_and_route(prompt: str, threshold: float, default_weak_target: str):
    """
    Evaluates prompt using RouteLLM and routes to appropriate model.
    """
    if get_router is None:
        log.warning("RouteLLM is not installed. Routing directly to weak proxy pipeline.")
        await ask_proxy(prompt, default_weak_target)
        return
        
    try:
        router = get_router("mf")
        
        # RouteLLM Context Fix: Extract and concatenate all raw historical context 
        # (ignoring system instructions to avoid noise, but preserving the conversation flow).
        eval_prompt = prompt
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list):
                # Concatenate all user and assistant messages for full context
                messages = [m.get("content", "") for m in parsed if m.get("role") in ("user", "assistant")]
                eval_prompt = "\n\n".join(messages)
        except json.JSONDecodeError:
            pass

        if len(eval_prompt) > 64000:
            log.info(f"Prompt length ({len(eval_prompt)} chars) exceeds 64,000. Bypassing RouteLLM and classifying as COMPLEX.")
            score = 1.0
        else:
            score = router.calculate_strong_win_rate(eval_prompt)
            log.info(f"RouteLLM (mf) evaluated prompt difficulty score: {score:.4f} (Threshold: {threshold})")
        
        if score > threshold:
            log.info(f"Prompt is COMPLEX (score > threshold). Attempting strong model...")
            success = await ask_strong_model(prompt, default_weak_target)
            if not success:
                log.info("Strong model failed. Bypassing RouteLLM recommendation and falling back to weak model...")
                await ask_proxy(prompt, default_weak_target)
        else:
            log.info(f"Prompt is SIMPLE (score <= threshold). Routing to weak model pipeline...")
            await ask_proxy(prompt, default_weak_target)
            
    except Exception as e:
        log.error(f"Failed to evaluate prompt via RouteLLM: {e}. Falling back to weak model.")
        await ask_proxy(prompt, default_weak_target)

async def amain():
    parser = argparse.ArgumentParser(
        description="FrugaLLM CLI (Intelligent Routing via RouteLLM)"
    )
    parser.add_argument("prompt", nargs="?", help="The prompt text (or use --stdin)")
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="RouteLLM threshold for strong model fallback")
    parser.add_argument(
        "--profile",
        "-p",
        choices=["engineer", "documenter", "devops"],
        help="Maps to reasoning or auto pools",
    )
    parser.add_argument(
        "--thinker", "--reasoner", action="store_true", help="Use deep-reasoning pool (thinker)"
    )
    parser.add_argument(
        "--offline", "--private", action="store_true", help="Force 100%% local CPU execution (offline)"
    )
    parser.add_argument(
        "--cloud", action="store_true", help="Route directly to paid Gemini 3.6 Flash (cloud)"
    )
    parser.add_argument(
        "--fast", "--lite", action="store_true", help="Route to Gemini 3.5 Flash-Lite (fast)"
    )
    parser.add_argument(
        "--free", action="store_true", help="Route to top free OpenRouter model (free)"
    )
    parser.add_argument(
        "--pro", action="store_true", help="Force escalation to the paid Pro tier"
    )
    parser.add_argument(
        "--local", action="store_true", help="Force local Ollama execution (legacy alias for --offline)"
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Force a specific passthrough or pseudo-model alias (e.g. frugal, thinker, cloud)",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Print the gateway's active model roster",
    )
    args = parser.parse_args()

    if args.models:
        await check_proxy_health()
        return

    if args.stdin:
        prompt = sys.stdin.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        sys.exit(1)

    if not prompt:
        print("[FrugaLLM CLI] Error: empty prompt.", file=sys.stderr)
        sys.exit(1)

    # ── Map CLI arguments to the gateway's weak model aliases ──
    target_model = "thinker"  # Default execution uses thinker
    if args.profile in ("engineer", "documenter"):
        target_model = "reasoning"
    elif args.profile == "devops":
        target_model = "auto"
    if args.thinker:
        target_model = "thinker"
    if args.local or args.offline:
        target_model = "offline"
    if args.cloud:
        target_model = "cloud"
    if args.fast:
        target_model = "fast"
    if args.free:
        target_model = "free"
    if args.pro:
        target_model = "pro"
    if args.model:
        target_model = args.model

    try:
        await evaluate_prompt_and_route(prompt, args.threshold, target_model)
    finally:
        await close_session()

if __name__ == "__main__":
    asyncio.run(amain())
