#!/usr/bin/env python3
"""
FrugaLLM Router CLI — Lightweight LiteLLM Wrapper
===================================================

A lightweight CLI wrapper that delegates 100% of its routing, caching,
failover, and telemetry to the LiteLLM gateway.

Takes your terminal input, formats it, and POSTs it to the local proxy.
Telemetry (Langfuse, Prometheus, etc.) is handled natively by LiteLLM —
this script only injects metadata into the standard OpenAI payload.

Usage:
  python -m frugallm.router_cli "What is the meaning of life?"
  echo "Explain quantum computing" | python -m frugallm.router_cli --stdin
  python -m frugallm.router_cli --models

Environment Variables:
  FRUGALLM_PROXY_URL    LiteLLM proxy URL (default: http://127.0.0.1:4000)
  FRUGALLM_MASTER_KEY   LiteLLM master key (default: sk-frugallm-master)
"""

import argparse
import json
import os
import sys
import urllib.request
from urllib.error import URLError, HTTPError

PROXY_URL = os.getenv("FRUGALLM_PROXY_URL", "http://127.0.0.1:5050")
PROXY_API_KEY = os.getenv("FRUGALLM_MASTER_KEY", "sk-frugallm-master")

# Baseline pricing for Claude 3.5 Sonnet (for telemetry cost comparison)
BASELINE_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
BASELINE_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000


def check_proxy_health():
    """Pings the proxy to retrieve its active model roster and health stats."""
    try:
        req = urllib.request.Request(
            f"{PROXY_URL}/health",
            headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("\n[FrugaLLM CLI] LiteLLM Gateway Health Report:")
            print(f"  Status: {data.get('status', 'Unknown')}")

            # LiteLLM returns health info in a different format
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
    sys.exit(0)


def ask_proxy(prompt: str, target_model: str):
    """Packages the terminal prompt and sends it to the LiteLLM gateway."""

    # Calculate baseline cost metadata for the telemetry dashboard.
    # Actual cost tracking is done by LiteLLM + Langfuse natively.
    payload = {
        "model": target_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # LiteLLM forwards metadata to Langfuse automatically
        "metadata": {
            "source": "frugallm-cli",
            "autonomy_tier": "user_guided",
            "tags": ["cli", target_model],
        },
    }

    req = urllib.request.Request(
        f"{PROXY_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PROXY_API_KEY}",
        },
        method="POST",
    )

    try:
        print(f"[FrugaLLM CLI] Sending to LiteLLM gateway (Target: {target_model})...")
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))

            # Extract response text
            text = result["choices"][0]["message"]["content"]
            model_used = result.get("model", "unknown")
            usage = result.get("usage", {})

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Calculate the baseline flagship cost for telemetry comparison
            baseline_cost = (
                input_tokens * BASELINE_INPUT_COST_PER_TOKEN
                + output_tokens * BASELINE_OUTPUT_COST_PER_TOKEN
            )

            print(
                f"[FrugaLLM CLI] Success! Model: {model_used} "
                f"(tokens: {input_tokens}→{output_tokens}, "
                f"baseline cost: ${baseline_cost:.6f})"
            )
            print("─" * 60)
            print(text)

    except HTTPError as e:
        error_msg = e.read().decode("utf-8")
        print(f"[FrugaLLM CLI] Gateway returned an error: HTTP {e.code}")
        print(error_msg)
    except URLError as e:
        print(
            f"[FrugaLLM CLI] Could not connect to gateway at {PROXY_URL}: {e.reason}"
        )
        print("Make sure LiteLLM is running!")
    except Exception as e:
        print(f"[FrugaLLM CLI] Unexpected error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="FrugaLLM CLI (Delegates to LiteLLM gateway)"
    )
    parser.add_argument("prompt", nargs="?", help="The prompt text (or use --stdin)")
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    parser.add_argument(
        "--profile",
        "-p",
        choices=["engineer", "documenter", "devops"],
        help="Maps to reasoning or auto pools",
    )
    parser.add_argument(
        "--pro", action="store_true", help="Force escalation to the paid Pro tier"
    )
    parser.add_argument(
        "--local", action="store_true", help="Force local Ollama execution"
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Force a specific passthrough model (e.g. anthropic/claude-3.5-sonnet)",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Print the gateway's active model roster",
    )
    args = parser.parse_args()

    if args.models:
        check_proxy_health()

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

    # ── Map CLI arguments to the gateway's model aliases ──
    target_model = "auto"
    if args.profile == "engineer":
        target_model = "reasoning"
    if args.local:
        target_model = "local"
    if args.pro:
        target_model = "pro"
    if args.model:
        target_model = args.model

    ask_proxy(prompt, target_model)


if __name__ == "__main__":
    main()
