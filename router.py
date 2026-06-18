#!/usr/bin/env python3
"""
Hermes Router CLI — Dumb Wrapper
======================================================
This is a lightweight CLI wrapper that delegates 100% of its
routing, caching, and failover logic to the router_server.py proxy.

It takes your terminal input, formats it, and POSTs it to localhost:5050.
"""

import argparse
import json
import sys
import urllib.request
from urllib.error import URLError, HTTPError

PROXY_URL = "http://127.0.0.1:5050"

def check_proxy_health():
    """Pings the proxy to retrieve its active model roster and health stats."""
    try:
        req = urllib.request.Request(f"{PROXY_URL}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("\n[Hermes Router CLI] Proxy Health Report:")
            print(f"  Status: {data.get('status', 'Unknown')}")
            print("\n  Active Roster:")
            for k, v in data.get('models', {}).items():
                print(f"    {k:15s} -> {v}")
            print("\n  Stats:")
            print(f"    Cache Size: {data.get('cache_size', 0)}")
            print(f"    Blacklisted Tools: {len(data.get('tool_blacklist', {}))}")
            print(f"    Cooldowns Active: {len(data.get('cooldowns', {}))}")
            print(f"    Consecutive Fails: {data.get('consecutive_failures', 0)}")
    except Exception as e:
        print(f"[Hermes Router CLI] Error: Proxy is offline at {PROXY_URL} ({e})")
        print("Make sure router_server.py is running via launchd!")
    sys.exit(0)

def ask_proxy(prompt: str, target_model: str):
    """Packages the terminal prompt and sends it to our intelligent proxy."""
    payload = {
        "model": target_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    req = urllib.request.Request(
        f"{PROXY_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        print(f"[Hermes Router CLI] Sending to proxy (Target: {target_model})...")
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            # Extract response text and the custom header/field we append in the proxy
            text = result["choices"][0]["message"]["content"]
            route = result.get("_hermes_route", "unknown")
            print(f"[Hermes Router CLI] Success! Routed via: {route}\n" + "─" * 60)
            print(text)
            
    except HTTPError as e:
        print(f"[Hermes Router CLI] Proxy returned an error: HTTP {e.code}")
        print(e.read().decode("utf-8"))
    except URLError as e:
        print(f"[Hermes Router CLI] Could not connect to proxy at {PROXY_URL}: {e.reason}")
        print("Make sure router_server.py is running!")

def main():
    parser = argparse.ArgumentParser(description="Hermes Router CLI (Delegates to localhost:5050)")
    parser.add_argument("prompt", nargs="?", help="The prompt text (or use --stdin)")
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    parser.add_argument("--profile", "-p", choices=["engineer", "documenter", "devops"], help="Maps to reasoning or auto pools")
    parser.add_argument("--pro", action="store_true", help="Force escalation to the paid Pro tier")
    parser.add_argument("--local", action="store_true", help="Force local Ollama execution")
    parser.add_argument("--model", "-m", help="Force a specific passthrough model (e.g. anthropic/claude-3.5-sonnet)")
    parser.add_argument("--models", action="store_true", help="Print the proxy's active model roster")
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
        print("[Hermes Router CLI] Error: empty prompt.", file=sys.stderr)
        sys.exit(1)

    # ── Map CLI arguments to the proxy's abstract routing tags ──
    target_model = "auto"
    if args.profile == "engineer":
        target_model = "reasoning"  # Tells the proxy to fetch from the DeepSeek/R1 pool
    if args.local:
        target_model = "local"      # Tells the proxy to hit Ollama
    if args.pro:
        target_model = "pro"        # Tells the proxy to jump straight to the Escalation Ladder
    if args.model:
        target_model = args.model   # Direct passthrough to a specific model string

    ask_proxy(prompt, target_model)

if __name__ == "__main__":
    main()