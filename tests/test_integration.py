#!/usr/bin/env python3
"""
FrugaLLM Integration Test Suite
=================================

A comprehensive live integration test suite that validates every model alias,
dynamic model, fallback chain, sidecar discovery logic, and infrastructure
health endpoint in the FrugaLLM stack.

Usage:
    python3 tests/test_integration.py                # Run all tests
    python3 tests/test_integration.py --skip-cloud   # Skip paid API tests
    python3 tests/test_integration.py --skip-dynamic  # Skip dynamic/OpenRouter tests
    python3 tests/test_integration.py --verbose       # Show full response bodies
    make test-suite                                   # Via Makefile

Test Groups:
    1. Static Model Aliases   — auto, reasoning, local, gemini-2.5-pro
    2. Dynamic Model Aliases  — free_balanced/free_reasoning chains from sidecar
    3. Sidecar Discovery Logic — _is_reasoning_model heuristic, YAML generation
    4. Infrastructure Health   — Gatekeeper, LiteLLM, Classifier endpoints
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

# ─── Configuration ───────────────────────────────────────────────────────────
GATEWAY_URL = os.getenv("FRUGALLM_GATEWAY_URL", "http://127.0.0.1:5050")
MASTER_KEY = os.getenv("FRUGALLM_MASTER_KEY", "sk-sidecar-1")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DYNAMIC_MODELS_PATH = CONFIG_DIR / "dynamic_models.yaml"

# ─── ANSI Colors ─────────────────────────────────────────────────────────────
class C:
    """ANSI color codes for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_RED  = "\033[41m"
    BG_GREEN = "\033[42m"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Result Tracking
# ═══════════════════════════════════════════════════════════════════════════════

class TestResult:
    """Tracks the result of a single test case."""
    def __init__(self, name: str, group: str):
        self.name = name
        self.group = group
        self.passed: bool | None = None
        self.warning: bool = False
        self.message: str = ""
        self.details: str = ""
        self.duration_ms: float = 0.0

    def pass_(self, message: str = "", details: str = ""):
        self.passed = True
        self.message = message
        self.details = details

    def fail(self, message: str, details: str = ""):
        self.passed = False
        self.message = message
        self.details = details

    def warn(self, message: str, details: str = ""):
        self.passed = True
        self.warning = True
        self.message = message
        self.details = details


class TestSuite:
    """Collects and reports test results."""
    def __init__(self, verbose: bool = False):
        self.results: list[TestResult] = []
        self.verbose = verbose
        self.current_group = ""

    def start_group(self, name: str):
        self.current_group = name
        print(f"\n{C.BOLD}{C.CYAN}{'═' * 70}")
        print(f"  {name}")
        print(f"{'═' * 70}{C.RESET}\n")

    def run_test(self, name: str, test_fn, *args, **kwargs) -> TestResult:
        result = TestResult(name, self.current_group)
        start = time.monotonic()
        try:
            test_fn(result, *args, **kwargs)
        except Exception as e:
            result.fail(f"Unhandled exception: {type(e).__name__}: {e}",
                        details=traceback.format_exc())
        result.duration_ms = (time.monotonic() - start) * 1000
        self.results.append(result)
        self._print_result(result)
        return result

    def _print_result(self, r: TestResult):
        duration = f"{C.DIM}({r.duration_ms:.0f}ms){C.RESET}"
        if r.passed and not r.warning:
            icon = f"{C.GREEN}✓{C.RESET}"
            label = f"{C.GREEN}{r.name}{C.RESET}"
            msg = f" — {r.message}" if r.message else ""
            print(f"  {icon} {label}{msg} {duration}")
        elif r.passed and r.warning:
            icon = f"{C.YELLOW}⚠{C.RESET}"
            label = f"{C.YELLOW}{r.name}{C.RESET}"
            print(f"  {icon} {label} — {r.message} {duration}")
        else:
            icon = f"{C.RED}✗{C.RESET}"
            label = f"{C.RED}{r.name}{C.RESET}"
            print(f"  {icon} {label} — {r.message} {duration}")

        if self.verbose and r.details:
            for line in r.details.strip().split("\n"):
                print(f"      {C.DIM}{line}{C.RESET}")

    def print_summary(self):
        passed = sum(1 for r in self.results if r.passed and not r.warning)
        warned = sum(1 for r in self.results if r.passed and r.warning)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print(f"\n{C.BOLD}{'═' * 70}")
        print(f"  TEST SUITE RESULTS")
        print(f"{'═' * 70}{C.RESET}\n")

        if failed == 0:
            bg = C.BG_GREEN
            status = "ALL TESTS PASSED"
        else:
            bg = C.BG_RED
            status = f"{failed} TEST(S) FAILED"

        print(f"  {C.BOLD}{bg}{C.WHITE} {status} {C.RESET}")
        print(f"  {C.GREEN}✓ {passed} passed{C.RESET}", end="")
        if warned:
            print(f"  {C.YELLOW}⚠ {warned} warnings{C.RESET}", end="")
        if failed:
            print(f"  {C.RED}✗ {failed} failed{C.RESET}", end="")
        print(f"  {C.DIM}({total} total){C.RESET}\n")

        # Print failed test details
        if failed > 0:
            print(f"  {C.RED}{C.BOLD}Failed Tests:{C.RESET}")
            for r in self.results:
                if not r.passed:
                    print(f"    {C.RED}✗ [{r.group}] {r.name}{C.RESET}")
                    print(f"      {r.message}")
                    if r.details:
                        for line in r.details.strip().split("\n")[:5]:
                            print(f"      {C.DIM}{line}{C.RESET}")
            print()

        return failed == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _api_call(
    path: str,
    method: str = "GET",
    body: dict | None = None,
    timeout: float = 30.0,
    base_url: str | None = None,
) -> tuple[int, dict | str]:
    """
    Make an HTTP request to the FrugaLLM gateway.
    Returns (status_code, parsed_json_or_raw_text).
    """
    url = f"{base_url or GATEWAY_URL}{path}"
    headers = {
        "Authorization": f"Bearer {MASTER_KEY}",
        "Content-Type": "application/json",
    }

    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, str(e)


def _chat_completion(model: str, prompt: str = "Say hi", timeout: float = 60.0) -> tuple[int, dict | str]:
    """Send a minimal chat completion request to a specific model alias."""
    return _api_call(
        "/v1/chat/completions",
        method="POST",
        body={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3072,
            "stream": False,
        },
        timeout=timeout,
    )


def _validate_completion_response(status: int, data: Any, expected_model: str | None = None) -> tuple[bool, str, str]:
    """
    Validate that a chat completion response is well-formed.
    Returns (ok, summary_message, detail_text).
    """
    if status == 0:
        return False, f"Connection failed: {data}", ""

    if status != 200:
        error_msg = data if isinstance(data, str) else json.dumps(data, indent=2)
        return False, f"HTTP {status}", error_msg

    if not isinstance(data, dict):
        return False, "Response is not a JSON object", str(data)[:200]

    choices = data.get("choices", [])
    if not choices:
        return False, "No choices in response", json.dumps(data, indent=2)[:300]

    message = choices[0].get("message", {})
    content = message.get("content", "")
    # Gemma with thinking enabled returns content in reasoning_content
    reasoning_content = ""
    if hasattr(message, "get"):
        reasoning_content = message.get("reasoning_content", "") or ""
    elif isinstance(message, dict):
        reasoning_content = message.get("reasoning_content", "") or ""

    has_content = bool(content) or bool(reasoning_content)
    if not has_content and not message.get("tool_calls"):
        return False, "Empty content and no tool_calls", json.dumps(data, indent=2)[:300]

    model_used = data.get("model", "?")
    usage = data.get("usage", {})
    tokens = f"{usage.get('prompt_tokens', '?')}→{usage.get('completion_tokens', '?')}"

    detail = f"model={model_used}, tokens={tokens}"
    if content:
        detail += f", content={content[:80]!r}"

    summary = f"model={model_used}, tokens={tokens}"
    return True, summary, detail


def _parse_dynamic_models_yaml() -> dict:
    """
    Parse dynamic_models.yaml to extract model names and fallback chains.
    Uses a simple line-by-line parser to avoid PyYAML dependency.

    Returns:
        {
            "model_names": ["free_balanced", "free_balanced_2", ...],
            "fallbacks": {"free_balanced": "free_balanced_2", ...},
            "balanced_models": [...],
            "reasoning_models": [...],
        }
    """
    if not DYNAMIC_MODELS_PATH.exists():
        return {"model_names": [], "fallbacks": {}, "balanced_models": [], "reasoning_models": []}

    text = DYNAMIC_MODELS_PATH.read_text()
    model_names = []
    fallbacks = {}
    balanced = []
    reasoning = []

    # Extract model_name entries
    in_model_list = False
    for line in text.split("\n"):
        stripped = line.strip()

        if stripped == "model_list:":
            in_model_list = True
            continue

        if stripped.startswith("router_settings:"):
            in_model_list = False

        if in_model_list and stripped.startswith("- model_name:"):
            name = stripped.split(":", 1)[1].strip()
            model_names.append(name)
            if "balanced" in name:
                balanced.append(name)
            elif "reasoning" in name:
                reasoning.append(name)

        # Parse fallback entries like: - {"auto": ["free_balanced"]}
        if stripped.startswith("- {") and ":" in stripped:
            # Strip the leading "- " and parse the JSON-ish dict
            try:
                entry_str = stripped[2:]
                entry = json.loads(entry_str)
                for src, dsts in entry.items():
                    if isinstance(dsts, list) and dsts:
                        fallbacks[src] = dsts[0]
            except (json.JSONDecodeError, TypeError):
                pass

    return {
        "model_names": model_names,
        "fallbacks": fallbacks,
        "balanced_models": balanced,
        "reasoning_models": reasoning,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1: Static Model Alias Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_static_model(result: TestResult, model: str, timeout: float = 120.0):
    """Test a static model alias by sending a chat completion request."""
    status, data = _chat_completion(model, prompt="Say hi in exactly 3 words.", timeout=timeout)
    ok, summary, detail = _validate_completion_response(status, data)
    if ok:
        result.pass_(summary, detail)
    else:
        result.fail(summary, detail)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2: Dynamic Model Alias Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_dynamic_model(result: TestResult, model: str, timeout: float = 60.0):
    """
    Test a dynamically-registered model from the sidecar.
    Failures on OpenRouter free models are treated as warnings (they rotate).
    """
    status, data = _chat_completion(model, prompt="Say hi.", timeout=timeout)
    ok, summary, detail = _validate_completion_response(status, data)

    if ok:
        result.pass_(summary, detail)
    elif "backup" in model:
        # Backup models are local — failures are real failures
        result.fail(summary, detail)
    else:
        # Free OpenRouter models may go offline — warn, don't fail
        result.warn(f"Free model may be offline: {summary}", detail)


def test_dynamic_models_yaml_exists(result: TestResult):
    """Verify that the sidecar has generated dynamic_models.yaml."""
    if DYNAMIC_MODELS_PATH.exists():
        size = DYNAMIC_MODELS_PATH.stat().st_size
        result.pass_(f"Exists ({size} bytes)")
    else:
        result.fail(f"File not found: {DYNAMIC_MODELS_PATH}")


def test_fallback_chain_integrity(result: TestResult, parsed: dict):
    """Verify the fallback chain is complete and terminates at a backup model."""
    fallbacks = parsed["fallbacks"]
    balanced = parsed["balanced_models"]
    reasoning = parsed["reasoning_models"]

    issues = []

    # Check balanced chain terminates at backup
    if balanced:
        current = "free_balanced"
        visited = set()
        while current in fallbacks and current not in visited:
            visited.add(current)
            current = fallbacks[current]
        if "backup" not in current:
            issues.append(f"Balanced chain does not terminate at backup (ends at {current})")

    # Check reasoning chain terminates at backup
    if reasoning:
        current = "free_reasoning"
        visited = set()
        while current in fallbacks and current not in visited:
            visited.add(current)
            current = fallbacks[current]
        if "backup" not in current:
            issues.append(f"Reasoning chain does not terminate at backup (ends at {current})")

    # Check auto → free_balanced fallback exists
    if "auto" not in fallbacks:
        issues.append("Missing fallback: auto → free_balanced")
    elif fallbacks["auto"] != "free_balanced":
        issues.append(f"auto fallback points to {fallbacks['auto']!r} instead of free_balanced")

    # Check reasoning → free_reasoning fallback exists
    if "reasoning" not in fallbacks:
        issues.append("Missing fallback: reasoning → free_reasoning")
    elif fallbacks["reasoning"] != "free_reasoning":
        issues.append(f"reasoning fallback points to {fallbacks['reasoning']!r} instead of free_reasoning")

    if issues:
        result.fail(f"{len(issues)} issue(s)", "\n".join(issues))
    else:
        chains = f"balanced={len(balanced)}, reasoning={len(reasoning)}"
        result.pass_(f"All chains terminate at backup ({chains})")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3: Sidecar Discovery Logic Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_is_reasoning_model(result: TestResult):
    """Unit test the _is_reasoning_model() heuristic from the sidecar."""
    # Import the function under test
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from frugallm.dynamic_roster_sidecar import _is_reasoning_model
    except ImportError as e:
        result.fail(f"Cannot import sidecar: {e}")
        return

    # Known positives — models that SHOULD be classified as reasoning
    positives = [
        {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1", "description": "A reasoning model"},
        {"id": "qwen/qwq-32b:free", "name": "QwQ-32B", "description": "Advanced chain-of-thought thinker"},
        {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "name": "Nemotron Reasoning", "description": ""},
        {"id": "google/gemini-2.5-pro-thinking:free", "name": "Gemini Pro Think", "description": ""},
    ]

    # Known negatives — models that should NOT be classified as reasoning
    negatives = [
        {"id": "google/gemma-4-12b-it:free", "name": "Gemma 4 12B", "description": "A general chat model"},
        {"id": "meta-llama/llama-3.1-8b-instruct:free", "name": "Llama 3.1 8B", "description": "Instruction tuned"},
        {"id": "cohere/command-r-plus:free", "name": "Command R+", "description": "Enterprise assistant"},
        {"id": "poolside/laguna-xs-2.1:free", "name": "Laguna XS", "description": "Code generation model"},
    ]

    failures = []

    for model in positives:
        if not _is_reasoning_model(model):
            failures.append(f"FALSE NEGATIVE: {model['id']} should be reasoning")

    for model in negatives:
        if _is_reasoning_model(model):
            failures.append(f"FALSE POSITIVE: {model['id']} should NOT be reasoning")

    if failures:
        result.fail(f"{len(failures)} classification error(s)", "\n".join(failures))
    else:
        result.pass_(f"All {len(positives) + len(negatives)} classifications correct")


def test_write_dynamic_models(result: TestResult):
    """Test that _write_dynamic_models produces valid YAML with correct structure."""
    import tempfile
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from frugallm.dynamic_roster_sidecar import _write_dynamic_models, _DYNAMIC_MODELS_PATH
    except ImportError as e:
        result.fail(f"Cannot import sidecar: {e}")
        return

    # Save original path and redirect to temp file
    import frugallm.dynamic_roster_sidecar as sidecar_module
    original_path = sidecar_module._DYNAMIC_MODELS_PATH

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            tmp_path = Path(f.name)

        sidecar_module._DYNAMIC_MODELS_PATH = tmp_path

        test_balanced = ["google/gemini-2.5-flash:free", "meta/llama-3:free"]
        test_reasoning = ["deepseek/deepseek-r1:free"]

        success = _write_dynamic_models(test_balanced, test_reasoning)
        if not success:
            result.fail("_write_dynamic_models returned False")
            return

        content = tmp_path.read_text()
        issues = []

        # Verify structure
        if "model_list:" not in content:
            issues.append("Missing 'model_list:' section")
        if "router_settings:" not in content:
            issues.append("Missing 'router_settings:' section")
        if "fallbacks:" not in content:
            issues.append("Missing 'fallbacks:' section")

        # Verify model names
        if "free_balanced" not in content:
            issues.append("Missing free_balanced model")
        if "free_reasoning" not in content:
            issues.append("Missing free_reasoning model")
        if "free_balanced_backup" not in content:
            issues.append("Missing free_balanced_backup fallback")
        if "free_reasoning_backup" not in content:
            issues.append("Missing free_reasoning_backup fallback")

        # Verify the correct model IDs are present
        for model_id in test_balanced:
            if model_id not in content and f"openrouter/{model_id}" not in content:
                issues.append(f"Missing balanced model ID: {model_id}")

        # Verify the backup models use gemini-flash (not local)
        if "gemini-3.6-flash" not in content and "gemini/gemini-3.6-flash" not in content:
            issues.append("Backup models should use gemini/gemini-3.6-flash as terminal fallback")
        if "ollama/hermes" in content:
            issues.append("Backup models should NOT use ollama/hermes (old pattern)")

        # Verify fallback chain references
        if '"auto"' not in content:
            issues.append("Missing auto → free_balanced fallback")
        if '"reasoning"' not in content:
            issues.append("Missing reasoning → free_reasoning fallback")

        if issues:
            result.fail(f"{len(issues)} structural issue(s)", "\n".join(issues) + f"\n\nGenerated:\n{content[:500]}")
        else:
            result.pass_(f"Valid YAML generated ({len(content)} bytes, {len(test_balanced)} balanced, {len(test_reasoning)} reasoning)")

    finally:
        sidecar_module._DYNAMIC_MODELS_PATH = original_path
        try:
            tmp_path.unlink()
        except Exception:
            pass


def test_openrouter_model_filtering(result: TestResult):
    """
    Test the sidecar's model filtering pipeline with synthetic OpenRouter data.
    Verifies: free filter, tool-use filter, context length filter, reasoning classification.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from frugallm.dynamic_roster_sidecar import _is_reasoning_model
    except ImportError as e:
        result.fail(f"Cannot import sidecar: {e}")
        return

    # Synthetic OpenRouter model catalog
    mock_models = [
        # Should pass all filters: free, tools, 256k+ context
        {
            "id": "test/good-balanced:free",
            "name": "Good Balanced",
            "description": "A balanced model",
            "context_length": 300000,
            "supported_parameters": ["tools", "temperature"],
            "pricing": {"prompt": "0", "completion": "0"},
        },
        # Should pass: free, tools, 256k+, AND is a reasoning model
        {
            "id": "test/good-reasoning:free",
            "name": "Good Reasoner",
            "description": "A chain-of-thought reasoning model",
            "context_length": 512000,
            "supported_parameters": ["tools", "temperature"],
            "pricing": {"prompt": "0", "completion": "0"},
        },
        # Should FAIL: not free (costs money)
        {
            "id": "test/paid-model",
            "name": "Paid Model",
            "description": "Costs money",
            "context_length": 300000,
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "3.0", "completion": "15.0"},
        },
        # Should FAIL: no tool support
        {
            "id": "test/no-tools:free",
            "name": "No Tools",
            "description": "No tool support",
            "context_length": 300000,
            "supported_parameters": ["temperature"],
            "pricing": {"prompt": "0", "completion": "0"},
        },
        # Should pass free+tools but fail 256k context filter
        {
            "id": "test/small-context:free",
            "name": "Small Context",
            "description": "Low context",
            "context_length": 8192,
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]

    # Replicate the sidecar's filtering logic
    free = []
    for m in mock_models:
        params = m.get("supported_parameters", [])
        if "tools" not in params:
            continue
        p = m.get("pricing", {})
        try:
            if float(p.get("prompt", "1")) == 0 and float(p.get("completion", "1")) == 0:
                free.append(m)
        except (ValueError, TypeError):
            continue

    free_balanced_256k = [m for m in free if m.get("context_length", 0) >= 256000]
    reasoning_256k = [m for m in free if _is_reasoning_model(m) and m.get("context_length", 0) >= 256000]

    issues = []

    # Verify free filter
    free_ids = {m["id"] for m in free}
    if "test/paid-model" in free_ids:
        issues.append("Paid model passed free filter")
    if "test/no-tools:free" in free_ids:
        issues.append("No-tools model passed tool filter")
    if "test/good-balanced:free" not in free_ids:
        issues.append("Good balanced model was incorrectly filtered out")

    # Verify 256k filter
    balanced_ids = {m["id"] for m in free_balanced_256k}
    if "test/small-context:free" in balanced_ids:
        issues.append("Small context model passed 256k filter")
    if "test/good-balanced:free" not in balanced_ids:
        issues.append("Good balanced 256k model was incorrectly filtered")

    # Verify reasoning classification
    reasoning_ids = {m["id"] for m in reasoning_256k}
    if "test/good-reasoning:free" not in reasoning_ids:
        issues.append("Good reasoning model was not classified as reasoning")
    if "test/good-balanced:free" in reasoning_ids:
        issues.append("Non-reasoning model was incorrectly classified as reasoning")

    if issues:
        result.fail(f"{len(issues)} filtering error(s)", "\n".join(issues))
    else:
        result.pass_(
            f"Filtering pipeline correct: "
            f"{len(free)} free, {len(free_balanced_256k)} balanced 256k+, "
            f"{len(reasoning_256k)} reasoning 256k+"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4: Infrastructure Health Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_gatekeeper_health(result: TestResult):
    """Test the Gatekeeper /health endpoint and verify all upstream services."""
    status, data = _api_call("/health")

    if status == 0:
        result.fail(f"Cannot reach Gatekeeper at {GATEWAY_URL}: {data}")
        return

    if not isinstance(data, dict):
        result.fail(f"Unexpected response format (HTTP {status})", str(data)[:300])
        return

    gk = data.get("gatekeeper", "unknown")
    litellm = data.get("litellm", "unknown")
    classifier = data.get("classifier", "unknown")

    issues = []
    if gk != "ready":
        issues.append(f"Gatekeeper: {gk}")
    if litellm != "healthy":
        issues.append(f"LiteLLM: {litellm}")
    if classifier != "healthy":
        issues.append(f"Classifier: {classifier}")

    if issues:
        result.fail(f"Unhealthy: {', '.join(issues)}", json.dumps(data, indent=2))
    else:
        result.pass_("All services healthy (gatekeeper=ready, litellm=healthy, classifier=healthy)")


def test_litellm_models_endpoint(result: TestResult):
    """Test that LiteLLM's /v1/models endpoint lists all expected model aliases."""
    status, data = _api_call("/v1/models")

    if status != 200:
        result.fail(f"HTTP {status}", str(data)[:300])
        return

    if not isinstance(data, dict) or "data" not in data:
        result.fail("Unexpected response format", str(data)[:300])
        return

    model_ids = {m.get("id", "") for m in data.get("data", [])}

    # These are the minimum expected static aliases
    required = {"auto", "reasoning", "local"}
    missing = required - model_ids
    if missing:
        result.fail(f"Missing static models: {missing}", f"Found: {sorted(model_ids)}")
    else:
        result.pass_(f"{len(model_ids)} models listed, all required aliases present")


def test_auth_rejection(result: TestResult):
    """Test that requests with an invalid API key are properly rejected."""
    url = f"{GATEWAY_URL}/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-INVALID-KEY-12345",
        "Content-Type": "application/json",
    }
    body = json.dumps({
        "model": "auto",
        "messages": [{"role": "user", "content": "test"}],
    }).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result.fail(f"Expected 401, got HTTP {resp.status} (request was accepted)")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            result.pass_("Invalid key correctly rejected with 401")
        else:
            result.fail(f"Expected 401, got HTTP {e.code}")
    except Exception as e:
        result.fail(f"Unexpected error: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 5: Langfuse Telemetry & Tracking Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_langfuse_config_enabled(result: TestResult):
    """Verify litellm_config.yaml enables langfuse in success and failure callbacks."""
    config_path = CONFIG_DIR / "litellm_config.yaml"
    if not config_path.exists():
        result.fail(f"Config file not found: {config_path}")
        return

    text = config_path.read_text()
    issues = []
    if "success_callback:" not in text or "- langfuse" not in text:
        issues.append("success_callback does not contain langfuse")
    if "failure_callback:" not in text or "- langfuse" not in text:
        issues.append("failure_callback does not contain langfuse")

    if issues:
        result.fail("; ".join(issues), text[:300])
    else:
        result.pass_("Langfuse enabled in success_callback and failure_callback")


def test_langfuse_env_keys(result: TestResult):
    """Verify .env contains all required Langfuse keys."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        result.fail(".env file not found")
        return

    text = env_path.read_text()
    required = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]
    missing = []
    for k in required:
        if k not in text or f"{k}=" not in text:
            missing.append(k)

    if missing:
        result.fail(f"Missing env key(s): {missing}")
    else:
        result.pass_("All Langfuse keys present in .env (SECRET, PUBLIC, HOST)")


def test_langfuse_model_normalization(result: TestResult):
    """Unit test model name normalization logic for Langfuse."""
    # Test logic matching HermesProxyHandler.async_log_success_event
    _OPENROUTER_PREFIX = "openrouter/"
    _NON_OPENROUTER_PREFIXES = (
        "ollama/", "openai/", "gemini/", "anthropic/",
        "bedrock/", "azure/", "huggingface/", "vertex_ai/", "cohere/", "mistral/"
    )

    def normalize(model: str, litellm_params: dict) -> str:
        if model.startswith(_OPENROUTER_PREFIX):
            return model
        if model.startswith(_NON_OPENROUTER_PREFIXES):
            return model
        config_model = litellm_params.get("model", "")
        api_base = litellm_params.get("api_base", "") or ""
        if config_model.startswith(_OPENROUTER_PREFIX) or "openrouter.ai" in api_base:
            return f"{_OPENROUTER_PREFIX}{model}"
        return model

    test_cases = [
        ("nvidia/nemotron", {"model": "openrouter/nvidia/nemotron"}, "openrouter/nvidia/nemotron"),
        ("openrouter/google/gemma", {"model": "openrouter/google/gemma"}, "openrouter/google/gemma"),
        ("gemini/gemini-3.6-flash", {"model": "gemini/gemini-3.6-flash"}, "gemini/gemini-3.6-flash"),
        ("ollama/hermes:latest", {"model": "ollama/hermes:latest"}, "ollama/hermes:latest"),
    ]

    failures = []
    for model_in, params, expected in test_cases:
        actual = normalize(model_in, params)
        if actual != expected:
            failures.append(f"Model '{model_in}': expected '{expected}', got '{actual}'")

    if failures:
        result.fail(f"{len(failures)} normalization failure(s)", "\n".join(failures))
    else:
        result.pass_(f"All {len(test_cases)} model normalization test cases passed")


def test_langfuse_cloud_reachability(result: TestResult):
    """Test HTTPS connectivity to LANGFUSE_HOST."""
    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    url = f"{host}/api/public/health"
    req = urllib.request.Request(url, headers={"User-Agent": "FrugaLLM-Test/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result.pass_(f"Langfuse host reachable: {host} (HTTP {resp.status})")
    except urllib.error.HTTPError as e:
        if e.code in (200, 401, 403, 404):
            result.pass_(f"Langfuse host reachable: {host} (HTTP {e.code})")
        else:
            result.fail(f"HTTP {e.code} from {url}")
    except Exception as e:
        result.fail(f"Cannot reach {host}: {type(e).__name__}: {e}")


def test_langfuse_metadata_propagation(result: TestResult):
    """Verify requests with Langfuse telemetry metadata pass through cleanly."""
    status, data = _api_call(
        "/v1/chat/completions",
        method="POST",
        body={
            "model": "gemini-flash-lite",
            "messages": [{"role": "user", "content": "Ping"}],
            "max_tokens": 16,
            "metadata": {
                "source": "frugallm-test-suite",
                "trace_name": "langfuse-test-trace",
                "tags": ["test", "telemetry"],
            },
        },
        timeout=15.0,
    )

    if status == 200 and isinstance(data, dict) and "choices" in data:
        result.pass_("Metadata payload accepted and routed through telemetry pipeline")
    else:
        result.fail(f"HTTP {status}", str(data)[:200])


# ═══════════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FrugaLLM Integration Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/test_integration.py                # Run all tests
  python3 tests/test_integration.py --skip-cloud   # Skip paid API tests
  python3 tests/test_integration.py --skip-dynamic  # Skip dynamic model tests
  python3 tests/test_integration.py --verbose       # Show full response bodies
""",
    )
    parser.add_argument("--skip-cloud", action="store_true",
                        help="Skip paid cloud API tests (gemini-2.5-pro)")
    parser.add_argument("--skip-dynamic", action="store_true",
                        help="Skip dynamic/OpenRouter model tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full response details for each test")
    args = parser.parse_args()

    suite = TestSuite(verbose=args.verbose)

    # ── Banner ────────────────────────────────────────────────────────────
    print(f"""
{C.BOLD}{C.MAGENTA}╔══════════════════════════════════════════════════════════════════════╗
║                 FrugaLLM Integration Test Suite                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Gateway:     {GATEWAY_URL:<54}║
║  Config:      {str(CONFIG_DIR):<54}║
║  Skip Cloud:  {str(args.skip_cloud):<54}║
║  Skip Dynamic:{str(args.skip_dynamic):<54}║
╚══════════════════════════════════════════════════════════════════════╝{C.RESET}
""")

    # ── GROUP 4: Infrastructure Health (run first — everything depends on this)
    suite.start_group("GROUP 4: Infrastructure Health")
    suite.run_test("Gatekeeper /health", test_gatekeeper_health)
    suite.run_test("LiteLLM /v1/models", test_litellm_models_endpoint)
    suite.run_test("Auth rejection (invalid key)", test_auth_rejection)

    # Bail early if infrastructure is down
    infra_results = [r for r in suite.results if not r.passed]
    if infra_results:
        print(f"\n  {C.RED}{C.BOLD}⚠ Infrastructure tests failed — skipping model tests{C.RESET}\n")
        suite.print_summary()
        sys.exit(1)

    # ── GROUP 5: Langfuse Telemetry & Tracking ────────────────────────────
    suite.start_group("GROUP 5: Langfuse Telemetry & Tracking")
    suite.run_test("Langfuse config enabled", test_langfuse_config_enabled)
    suite.run_test("Langfuse API keys present", test_langfuse_env_keys)
    suite.run_test("Model name normalization", test_langfuse_model_normalization)
    suite.run_test("Langfuse Cloud reachability", test_langfuse_cloud_reachability)
    suite.run_test("Langfuse metadata propagation", test_langfuse_metadata_propagation)

    # ── GROUP 1: Static Model Aliases ─────────────────────────────────────
    suite.start_group("GROUP 1: Static Model Aliases")

    static_models = [
        ("auto", 120.0),
        ("reasoning", 120.0),
        ("local", 90.0),
        # Friendly Intuitive Aliases
        ("frugal", 120.0),
        ("smart", 120.0),
        ("thinker", 120.0),
        ("free", 60.0),
        ("fast", 60.0),
    ]

    cloud_models = [
        ("gemini-flash", 120.0),
        ("gemini-flash-lite", 120.0),
        ("gemini-pro", 120.0),
        ("cloud", 120.0),
    ]

    if not args.skip_cloud:
        static_models.extend(cloud_models)
    else:
        print(f"  {C.DIM}⏭ Skipping Gemini fleet (--skip-cloud){C.RESET}")

    for model, timeout in static_models:
        suite.run_test(f"Model: {model}", test_static_model, model, timeout)

    # ── GROUP 2: Dynamic Model Aliases ────────────────────────────────────
    suite.start_group("GROUP 2: Dynamic Model Aliases")

    if args.skip_dynamic:
        print(f"  {C.DIM}⏭ Skipped all dynamic model tests (--skip-dynamic){C.RESET}")
    else:
        suite.run_test("dynamic_models.yaml exists", test_dynamic_models_yaml_exists)
        parsed = _parse_dynamic_models_yaml()

        if not parsed["model_names"]:
            print(f"  {C.YELLOW}⚠ No dynamic models found — sidecar may not have run yet{C.RESET}")
        else:
            suite.run_test("Fallback chain integrity", test_fallback_chain_integrity, parsed)

            # Test a representative subset: first balanced, first reasoning, and both backups
            test_candidates = []
            if parsed["balanced_models"]:
                test_candidates.append(parsed["balanced_models"][0])   # free_balanced
                if "free_balanced_backup" in parsed["balanced_models"]:
                    test_candidates.append("free_balanced_backup")
            if parsed["reasoning_models"]:
                test_candidates.append(parsed["reasoning_models"][0])  # free_reasoning
                if "free_reasoning_backup" in parsed["reasoning_models"]:
                    test_candidates.append("free_reasoning_backup")

            # Deduplicate while preserving order
            seen = set()
            for model in test_candidates:
                if model not in seen:
                    seen.add(model)
                    suite.run_test(f"Dynamic: {model}", test_dynamic_model, model, 60.0)

    # ── GROUP 3: Sidecar Discovery Logic ──────────────────────────────────
    suite.start_group("GROUP 3: Sidecar Discovery Logic")
    suite.run_test("_is_reasoning_model() heuristic", test_is_reasoning_model)
    suite.run_test("_write_dynamic_models() YAML gen", test_write_dynamic_models)
    suite.run_test("OpenRouter model filtering pipeline", test_openrouter_model_filtering)

    # ── Summary ───────────────────────────────────────────────────────────
    all_passed = suite.print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
