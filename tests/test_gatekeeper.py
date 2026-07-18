#!/usr/bin/env python3
"""
FrugaLLM Gatekeeper — Unit Tests
==================================

Tests all three components in isolation without requiring a running LLM.
Run with:
    python3 -m pytest tests/test_gatekeeper.py -v
    # or directly:
    python3 tests/test_gatekeeper.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

# Ensure the module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frugallm.gatekeeper import GatekeeperConfig, ToolCallValidator, GatekeeperMiddleware, _fatal_error_response


# ═════════════════════════════════════════════════════════════════════════════
# Test: GatekeeperConfig
# ═════════════════════════════════════════════════════════════════════════════

class TestGatekeeperConfig(unittest.TestCase):
    """Test environment variable parsing and defaults."""

    def test_defaults(self):
        """Config should have sane defaults when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = GatekeeperConfig()
            self.assertFalse(cfg.enabled)
            self.assertEqual(cfg.max_retries, 3)
            self.assertEqual(cfg.timeout_seconds, 30)
            self.assertEqual(cfg.target_model, "reasoning")

    def test_enabled_true(self):
        """Various truthy values should enable the gatekeeper."""
        for val in ("true", "True", "TRUE", "1", "yes"):
            with patch.dict(os.environ, {"GATEKEEPER_ENABLED": val}):
                cfg = GatekeeperConfig()
                self.assertTrue(cfg.enabled, f"Failed for value: {val!r}")

    def test_enabled_false(self):
        """Non-truthy values should keep the gatekeeper disabled."""
        for val in ("false", "0", "no", "nope", ""):
            with patch.dict(os.environ, {"GATEKEEPER_ENABLED": val}):
                cfg = GatekeeperConfig()
                self.assertFalse(cfg.enabled, f"Failed for value: {val!r}")

    def test_custom_values(self):
        """Custom env values should override defaults."""
        env = {
            "GATEKEEPER_ENABLED": "true",
            "GATEKEEPER_MAX_RETRIES": "5",
            "GATEKEEPER_TIMEOUT_SECONDS": "60",
            "GATEKEEPER_TARGET_MODEL": "  Local-Gemma  ",
        }
        with patch.dict(os.environ, env):
            cfg = GatekeeperConfig()
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.max_retries, 5)
            self.assertEqual(cfg.timeout_seconds, 60)
            self.assertEqual(cfg.target_model, "local-gemma")


# ═════════════════════════════════════════════════════════════════════════════
# Test: ToolCallValidator — Tier 1 (Syntax)
# ═════════════════════════════════════════════════════════════════════════════

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_server_logs",
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Number of lines"},
                    "server": {"type": "string", "description": "Server name"},
                },
                "required": ["lines"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                },
                "required": ["service_name"],
            },
        },
    },
]


class TestValidatorTier1(unittest.TestCase):
    """Test JSON extraction from raw text (Tier 1: Syntax Validation)."""

    def setUp(self):
        self.validator = ToolCallValidator(SAMPLE_TOOLS)

    def test_extract_json_from_markdown_fence(self):
        """JSON inside ```json ... ``` blocks should be extracted."""
        text = 'Here is the call:\n```json\n{"name": "check_server_logs", "arguments": {"lines": 50}}\n```'
        result = self.validator._extract_json(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(parsed["name"], "check_server_logs")

    def test_extract_json_from_bare_fence(self):
        """JSON inside bare ``` blocks should also be extracted."""
        text = '```\n{"name": "check_server_logs", "arguments": {"lines": 20}}\n```'
        result = self.validator._extract_json(text)
        self.assertIsNotNone(result)

    def test_extract_json_from_raw_text(self):
        """JSON embedded in conversational text should be extracted."""
        text = 'I will check the logs now: {"name": "check_server_logs", "arguments": {"lines": 100}} That should work.'
        result = self.validator._extract_json(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(parsed["arguments"]["lines"], 100)

    def test_extract_json_nested_braces(self):
        """Nested JSON objects should be handled correctly."""
        text = '{"name": "check_server_logs", "arguments": {"lines": 50, "filter": {"level": "error"}}}'
        result = self.validator._extract_json(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(parsed["arguments"]["filter"]["level"], "error")

    def test_extract_json_with_escaped_strings(self):
        """JSON with escaped quotes inside strings should be handled."""
        text = '{"name": "check_server_logs", "arguments": {"lines": 50, "pattern": "he said \\"hello\\""}}'
        result = self.validator._extract_json(text)
        self.assertIsNotNone(result)

    def test_no_json_in_text(self):
        """Text without JSON should return None."""
        text = "I'll check the server logs for you and report back."
        result = self.validator._extract_json(text)
        self.assertIsNone(result)

    def test_incomplete_json(self):
        """Unbalanced braces should return None."""
        text = '{"name": "check_server_logs", "arguments": {"lines": 50}'
        result = self.validator._extract_json(text)
        self.assertIsNone(result)


# ═════════════════════════════════════════════════════════════════════════════
# Test: ToolCallValidator — Tier 2 (Schema)
# ═════════════════════════════════════════════════════════════════════════════

class TestValidatorTier2(unittest.TestCase):
    """Test tool name and argument schema validation (Tier 2)."""

    def setUp(self):
        self.validator = ToolCallValidator(SAMPLE_TOOLS)

    def test_valid_tool_call(self):
        """A correct tool call should pass validation."""
        valid, err = self.validator._validate_schema(
            "check_server_logs", {"lines": 50}
        )
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_valid_tool_with_optional_args(self):
        """Optional arguments should not cause failure."""
        valid, err = self.validator._validate_schema(
            "check_server_logs", {"lines": 50, "server": "prod-01"}
        )
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_unknown_tool_name(self):
        """A tool name not in the registry should fail."""
        valid, err = self.validator._validate_schema(
            "nonexistent_tool", {"arg": "value"}
        )
        self.assertFalse(valid)
        self.assertIn("not in your registry", err)

    def test_missing_required_argument(self):
        """Missing a required argument should fail."""
        valid, err = self.validator._validate_schema(
            "check_server_logs", {"server": "prod-01"}  # missing 'lines'
        )
        self.assertFalse(valid)
        self.assertIn("lines", err.lower())

    def test_wrong_argument_type(self):
        """Wrong type for an argument should fail."""
        valid, err = self.validator._validate_schema(
            "check_server_logs", {"lines": "fifty"}  # should be integer
        )
        self.assertFalse(valid)


# ═════════════════════════════════════════════════════════════════════════════
# Test: ToolCallValidator — Full Response Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestValidatorFullResponse(unittest.TestCase):
    """Test end-to-end response validation (both Tier 1 + Tier 2)."""

    def setUp(self):
        self.validator = ToolCallValidator(SAMPLE_TOOLS)

    def _make_native_response(self, tool_calls):
        """Helper to construct a native OpenAI-format response."""
        return {
            "id": "test",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _make_text_response(self, content):
        """Helper to construct a text-only response."""
        return {
            "id": "test",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def test_native_tool_calls_pass(self):
        """Native tool_calls with correct schema should pass."""
        resp = self._make_native_response([{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "check_server_logs",
                "arguments": json.dumps({"lines": 50}),
            },
        }])
        valid, err, result = self.validator.validate_response(resp)
        self.assertTrue(valid)
        self.assertIsNone(err)
        self.assertIsNotNone(result)

    def test_native_tool_calls_bad_args(self):
        """Native tool_calls with invalid JSON arguments should fail."""
        resp = self._make_native_response([{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "check_server_logs",
                "arguments": "not valid json {{{",
            },
        }])
        valid, err, _ = self.validator.validate_response(resp)
        self.assertFalse(valid)
        self.assertIn("invalid JSON arguments", err)

    def test_text_with_valid_json(self):
        """Text content containing valid tool call JSON should pass."""
        resp = self._make_text_response(
            '{"name": "check_server_logs", "arguments": {"lines": 50}}'
        )
        valid, err, result = self.validator.validate_response(resp)
        self.assertTrue(valid)
        self.assertIsNone(err)
        # Should construct a proper tool_calls response
        tc = result["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["function"]["name"], "check_server_logs")

    def test_text_hallucination_triggers_tier15(self):
        """Conversational text with action intent should trigger Tier 1.5 reprimand."""
        resp = self._make_text_response(
            "I'll check the server logs for you to find the last 50 lines."
        )
        valid, err, _ = self.validator.validate_response(resp)
        self.assertFalse(valid)
        self.assertIn("CRITICAL VALIDATION ERROR", err)

    def test_text_with_wrong_tool_name(self):
        """Text with JSON calling a non-existent tool should fail."""
        resp = self._make_text_response(
            '{"name": "delete_everything", "arguments": {}}'
        )
        valid, err, _ = self.validator.validate_response(resp)
        self.assertFalse(valid)
        self.assertIn("not in your registry", err)

    def test_empty_response(self):
        """Empty response should fail."""
        resp = self._make_text_response("")
        valid, err, _ = self.validator.validate_response(resp)
        self.assertFalse(valid)

    def test_no_choices(self):
        """Response with no choices should fail."""
        resp = {"id": "test", "model": "test", "choices": []}
        valid, err, _ = self.validator.validate_response(resp)
        self.assertFalse(valid)


# ═════════════════════════════════════════════════════════════════════════════
# Test: Tier 1.5 — Heuristic Intent Detection ("Empty Promises")
# ═════════════════════════════════════════════════════════════════════════════

class TestValidatorTier15(unittest.TestCase):
    """Test heuristic intent detection for 'Empty Promises'."""

    def setUp(self):
        self.validator = ToolCallValidator(SAMPLE_TOOLS)

    # ── Positive matches: should trigger Tier 1.5 ────────────────────────

    def test_i_will_create(self):
        """'I will create a ticket' should trigger empty promise detection."""
        is_promise, msg = self.validator._detect_empty_promise(
            "I will create a ticket for this issue now."
        )
        self.assertTrue(is_promise)
        self.assertIn("CRITICAL VALIDATION ERROR", msg)

    def test_i_ll_run(self):
        """'I'll run the script' should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "I'll run the deployment script immediately."
        )
        self.assertTrue(is_promise)

    def test_let_me_execute(self):
        """'Let me execute the query' should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "Let me execute the database query for you."
        )
        self.assertTrue(is_promise)

    def test_first_i_will_check(self):
        """'First, I will check the logs' should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "First, I will check the server logs to identify the issue."
        )
        self.assertTrue(is_promise)

    def test_then_i_will_restart(self):
        """'Then I will restart the service' should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "Then I will restart the nginx service."
        )
        self.assertTrue(is_promise)

    def test_next_let_me_update(self):
        """'Next, let me update the config' should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "Next, let me update the configuration file."
        )
        self.assertTrue(is_promise)

    def test_now_i_will_deploy(self):
        """'Now I will deploy the changes' should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "Now I will deploy the latest changes to production."
        )
        self.assertTrue(is_promise)

    def test_i_am_going_to_query(self):
        """'I am going to query the database' should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "I am going to query the database to find the record."
        )
        self.assertTrue(is_promise)

    def test_gerund_creating_ticket(self):
        """'creating ticket' (gerund pattern) should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "Okay, creating ticket for the deployment issue."
        )
        self.assertTrue(is_promise)

    def test_gerund_running_script(self):
        """'running script' (gerund pattern) should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "Running the backup script as requested."
        )
        self.assertTrue(is_promise)

    def test_gerund_querying_database(self):
        """'querying database' (gerund pattern) should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "Querying the production database for that record now."
        )
        self.assertTrue(is_promise)

    def test_i_shall_invoke(self):
        """'I shall invoke the function' should trigger."""
        is_promise, msg = self.validator._detect_empty_promise(
            "I shall invoke the restart_service function."
        )
        self.assertTrue(is_promise)

    # ── Negative matches: should NOT trigger Tier 1.5 ────────────────────

    def test_benign_help_offer(self):
        """'I can help you think about this' should NOT trigger."""
        is_promise, _ = self.validator._detect_empty_promise(
            "I can help you think about the best approach here."
        )
        self.assertFalse(is_promise)

    def test_benign_question(self):
        """'Would you like me to...' should NOT trigger."""
        is_promise, _ = self.validator._detect_empty_promise(
            "Would you like me to look into this further?"
        )
        self.assertFalse(is_promise)

    def test_benign_explanation(self):
        """Explanatory text without action intent should NOT trigger."""
        is_promise, _ = self.validator._detect_empty_promise(
            "The server logs typically contain error messages and timestamps."
        )
        self.assertFalse(is_promise)

    def test_benign_past_tense(self):
        """Past tense descriptions should NOT trigger."""
        is_promise, _ = self.validator._detect_empty_promise(
            "The system previously created a backup automatically."
        )
        self.assertFalse(is_promise)

    def test_benign_no_verb(self):
        """Text without action verbs should NOT trigger."""
        is_promise, _ = self.validator._detect_empty_promise(
            "Here is the configuration file you requested."
        )
        self.assertFalse(is_promise)

    def test_benign_pure_json(self):
        """Pure JSON (no prose) should NOT trigger (Tier 1 would catch it)."""
        is_promise, _ = self.validator._detect_empty_promise(
            '{"name": "check_server_logs", "arguments": {"lines": 50}}'
        )
        self.assertFalse(is_promise)


# ═════════════════════════════════════════════════════════════════════════════
# Test: GatekeeperConfig.should_intercept
# ═════════════════════════════════════════════════════════════════════════════

class TestShouldIntercept(unittest.TestCase):
    """Test the should_intercept logic on GatekeeperConfig."""

    def setUp(self):
        env = {
            "GATEKEEPER_ENABLED": "true",
            "GATEKEEPER_MAX_RETRIES": "3",
            "GATEKEEPER_TIMEOUT_SECONDS": "10",
            "GATEKEEPER_TARGET_MODEL": "reasoning",
        }
        with patch.dict(os.environ, env):
            self.config = GatekeeperConfig()

    def test_should_intercept_all_conditions(self):
        """should_intercept returns True when all conditions met."""
        self.assertTrue(self.config.should_intercept("reasoning", [{"type": "function"}]))

    def test_should_not_intercept_disabled(self):
        """should_intercept returns False when disabled."""
        self.config.enabled = False
        self.assertFalse(self.config.should_intercept("reasoning", [{"type": "function"}]))

    def test_should_not_intercept_wrong_model(self):
        """should_intercept returns False for non-target model."""
        self.assertTrue(self.config.enabled)
        self.assertFalse(self.config.should_intercept("auto", [{"type": "function"}]))

    def test_should_not_intercept_no_tools(self):
        """should_intercept returns False when no tools are present."""
        self.assertFalse(self.config.should_intercept("reasoning", []))


# ═════════════════════════════════════════════════════════════════════════════
# Test: Module-Level Helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestModuleLevelHelpers(unittest.TestCase):
    """Test the module-level helper functions."""

    def test_fatal_error_response_format(self):
        """Fatal error response should have the correct OpenAI format."""
        result = _fatal_error_response("test-model", "Test error")

        self.assertEqual(result["object"], "chat.completion")
        self.assertTrue(result["_gatekeeper_fatal"])
        choices = result["choices"]
        self.assertEqual(len(choices), 1)
        msg = choices[0]["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertIsNone(msg["content"])
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "__frugallm_gatekeeper_fatal_error__")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args["message"], "Test error")

    def test_middleware_class_importable(self):
        """GatekeeperMiddleware should be importable from gatekeeper module."""
        self.assertTrue(callable(GatekeeperMiddleware))


# ═════════════════════════════════════════════════════════════════════════════
# Test: Full Response Validation with Tier 1.5 Integration
# ═════════════════════════════════════════════════════════════════════════════

class TestFullResponseTier15Integration(unittest.TestCase):
    """Test that Tier 1.5 integrates correctly into the full validation pipeline."""

    def setUp(self):
        self.validator = ToolCallValidator(SAMPLE_TOOLS)

    def _make_text_response(self, content):
        return {
            "id": "test",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def test_empty_promise_gets_critical_reprimand(self):
        """A text response with action intent (no JSON) should get the Tier 1.5 reprimand."""
        resp = self._make_text_response(
            "I will check the server logs to diagnose the issue."
        )
        valid, err, _ = self.validator.validate_response(resp)
        self.assertFalse(valid)
        self.assertIn("CRITICAL VALIDATION ERROR", err)
        self.assertIn("did not execute the required tool call", err)

    def test_generic_text_gets_syntax_error(self):
        """Plain text without action intent should get the generic Tier 1 error."""
        resp = self._make_text_response(
            "The weather is nice today and the sky is blue."
        )
        valid, err, _ = self.validator.validate_response(resp)
        self.assertFalse(valid)
        self.assertIn("SYSTEM ERROR", err)
        self.assertNotIn("CRITICAL VALIDATION ERROR", err)

    def test_json_in_text_skips_tier15(self):
        """Text with valid JSON should skip Tier 1.5 entirely (Tier 1 succeeds)."""
        resp = self._make_text_response(
            'I will create the ticket: {"name": "check_server_logs", "arguments": {"lines": 50}}'
        )
        valid, err, result = self.validator.validate_response(resp)
        self.assertTrue(valid)
        self.assertIsNone(err)


# ═════════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
