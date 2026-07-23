#!/usr/bin/env python3
"""
FrugaLLM Gatekeeper — Unified Async ASGI Validator & Retry Engine
====================================================================

A deterministic middleware layer that intercepts LLM responses flowing through
LiteLLM and validates them for proper tool call formatting. When small local
models (like Qwen and Gemma) "hallucinate" tool use — promising to run a tool
in conversational prose instead of outputting the required structured JSON —
the Gatekeeper catches the failure, injects explicit system feedback, and
retries transparently.

Validation Tiers:
    Tier 1   — Syntax: Extract JSON from raw text (markdown fences, balanced braces)
    Tier 1.5 — Heuristic Intent Detection: Catch "Empty Promises" where the model
               states intent to act but never outputs the tool call JSON block.
    Tier 2   — Schema: Validate tool name + arguments against declared JSON Schema

Architecture:
    GatekeeperConfig     — Environment variable reader with sane defaults
    ToolCallValidator    — Three-tier pure-Python validator (Syntax + Heuristic + Schema)
    GatekeeperMiddleware — ASGI middleware: sole retry engine ("Clone and Trash")

Loaded by LiteLLM via litellm_config.yaml:
    middleware:
      - gatekeeper.GatekeeperMiddleware

Uses only Python stdlib (json, re, copy, asyncio) for the core loop. The
jsonschema library is used for rigorous Tier 2 validation if available, with
a graceful fallback to basic key-presence checks.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import time
import uuid

log = logging.getLogger("hermes-gatekeeper")

# ─── Optional: jsonschema for rigorous Tier 2 validation ──────────────────────
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    log.warning("jsonschema not installed; Tier 2 falls back to basic key-presence checks.")


# ═════════════════════════════════════════════════════════════════════════════
# Component A: Configuration
# ═════════════════════════════════════════════════════════════════════════════

class GatekeeperConfig:
    """
    Reads Gatekeeper configuration from environment variables.

    Variables (all optional, with safe defaults):
        GATEKEEPER_ENABLED           (bool,  default: false)
        GATEKEEPER_MAX_RETRIES       (int,   default: 3)
        GATEKEEPER_TIMEOUT_SECONDS   (int,   default: 30)
        GATEKEEPER_TARGET_MODEL      (str,   default: "reasoning")
    """

    def __init__(self):
        self.reload()

    def reload(self):
        """Re-read config from environment. Safe to call at any time."""
        self.enabled = os.getenv(
            "GATEKEEPER_ENABLED", "false"
        ).lower() in ("true", "1", "yes")
        self.max_retries = int(os.getenv("GATEKEEPER_MAX_RETRIES", "3"))
        self.timeout_seconds = int(os.getenv("GATEKEEPER_TIMEOUT_SECONDS", "30"))
        self.target_model = os.getenv(
            "GATEKEEPER_TARGET_MODEL", "reasoning"
        ).lower().strip()

    def should_intercept(self, requested_model: str, tools: list) -> bool:
        """
        Check whether this request should be intercepted.

        All three conditions must be true:
            1. GATEKEEPER_ENABLED is true
            2. requested model matches GATEKEEPER_TARGET_MODEL
            3. tools array is non-empty
        """
        if not self.enabled:
            return False
        if requested_model.lower().strip() != self.target_model:
            return False
        if not tools:
            return False
        return True

    def __repr__(self):
        return (
            f"GatekeeperConfig(enabled={self.enabled}, "
            f"target={self.target_model!r}, "
            f"retries={self.max_retries}, "
            f"timeout={self.timeout_seconds}s)"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Component B: The Three-Tier Referee (Validator)
# ═════════════════════════════════════════════════════════════════════════════

class ToolCallValidator:
    """
    Pure-Python, deterministic validator for LLM tool call output.

    Tier 1 — Syntax Validation:
        Extracts JSON from raw text (markdown fences, balanced braces)
        and verifies it parses without error.

    Tier 1.5 — Heuristic Intent Detection ("Empty Promises"):
        When no JSON is found, scans the raw text for high-confidence
        action intents (sequential markers + execution verbs). Catches
        models that describe what they will do instead of doing it.

    Tier 2 — Schema/Tool Validation:
        Checks that the tool name exists in the registry and that the
        arguments satisfy the JSON Schema from the tools payload.
    """

    # ── Tier 1.5: Compiled Regex Patterns ─────────────────────────────────
    # Dual-gate: requires BOTH an intent/sequential marker AND an execution
    # verb in the same phrase to avoid false positives on benign text.

    _INTENT_MARKERS = re.compile(
        r"(?:"
        r"I(?:'ll| will| am going to| shall)"
        r"|[Ll]et me"
        r"|[Ff]irst(?:,?\s*I(?:'ll| will))?"
        r"|[Tt]hen(?:,?\s*I(?:'ll| will))?"
        r"|[Nn]ext(?:,?\s*(?:I(?:'ll| will)|[Ll]et me))?"
        r"|[Nn]ow(?:,?\s*I(?:'ll| will))?"
        r")"
        r"\s+"
        r"(?:"
        r"creat|run|execut|query|call|invok|fetch|updat|delet"
        r"|restart|deploy|check|search|send|post|submit|generat"
        r"|build|process|launch|trigger|schedul|dispatch|perform"
        r")\w*",
        re.IGNORECASE,
    )

    _ACTION_GERUNDS = re.compile(
        r"(?:"
        r"creating|running|executing|querying|calling|invoking|fetching"
        r"|updating|deleting|restarting|deploying|checking|searching|sending"
        r"|posting|submitting|generating|building|processing|launching"
        r"|triggering|scheduling|dispatching|performing"
        r")"
        r"\s+(?:the\s+|a\s+|an\s+)?"
        r"\w+",
        re.IGNORECASE,
    )

    _EMPTY_PROMISE_REPRIMAND = (
        "CRITICAL VALIDATION ERROR: You stated intent to perform an action but did not "
        "execute the required tool call. Stop conversational prose. "
        "Output ONLY the JSON block immediately."
    )

    def __init__(self, tools: list):
        self.tool_registry: dict[str, dict] = {}
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name")
                if name:
                    self.tool_registry[name] = func.get("parameters", {})
            elif "name" in tool:
                # Legacy "functions" format already normalized
                self.tool_registry[tool["name"]] = tool.get("parameters", {})

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_response(self, response_data: dict) -> tuple[bool, str | None, dict | None]:
        """
        Validate an LLM response for proper tool call formatting.

        Returns:
            (is_valid, error_message_or_none, validated_response_or_none)
        """
        choices = response_data.get("choices", [])
        if not choices:
            return (
                False,
                "[SYSTEM ERROR: Validation failed. No choices in LLM response.]",
                None,
            )

        message = choices[0].get("message", {})
        native_tool_calls = message.get("tool_calls", [])

        # ── Path A: Native tool_calls present → Tier 2 only ──────────────
        if native_tool_calls:
            return self._validate_native_tool_calls(native_tool_calls, response_data)

        # ── Path B: Text content → Tier 1 → Tier 1.5 → Tier 2 ───────────
        content = message.get("content") or ""
        if not content.strip():
            return (
                False,
                "[SYSTEM ERROR: Validation failed. Empty response with no tool "
                "calls. You must output a valid JSON tool call.]",
                None,
            )

        return self._validate_text_content(content, response_data)

    # ── Path A: Native Tool Calls ─────────────────────────────────────────

    def _validate_native_tool_calls(
        self, tool_calls: list, response_data: dict
    ) -> tuple[bool, str | None, dict | None]:
        """Validate tool_calls that the LLM returned in structured format."""
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_raw = func.get("arguments", "{}")

            # Parse arguments string
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                return (
                    False,
                    f"[SYSTEM ERROR: Validation failed. Tool call '{name}' has "
                    f"invalid JSON arguments. Ensure brackets and quotes are "
                    f"properly escaped.]",
                    None,
                )

            # Tier 2
            valid, err = self._validate_schema(name, args)
            if not valid:
                return False, err, None

        return True, None, response_data

    # ── Path B: Text Content → Extract + Validate ─────────────────────────

    def _validate_text_content(
        self, content: str, response_data: dict
    ) -> tuple[bool, str | None, dict | None]:
        """Extract JSON from text content, validate, and construct response."""

        # ── Tier 1.5: Heuristic Intent Detection ─────────────────────
        # Evaluate heuristic BEFORE structural JSON validation.
        # Do not short-circuit just because a stray { is found.
        is_promise, reprimand = self._detect_empty_promise(content)

        # Tier 1: Syntax — extract JSON
        extracted = self._extract_json(content)
        if extracted is None:
            if is_promise:
                return False, reprimand, None

            # Generic Tier 1 syntax failure
            return (
                False,
                "[SYSTEM ERROR: Validation failed. You must output valid JSON. "
                "Do not include conversational filler. Ensure brackets and "
                "quotes are properly escaped.]",
                None,
            )

        # Tier 1: Parse
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as e:
            if is_promise:
                return False, reprimand, None
            return (
                False,
                f"[SYSTEM ERROR: Validation failed. Extracted text is not valid "
                f"JSON: {e}. Output only the JSON tool call.]",
                None,
            )

        # Normalize to {name, arguments}
        tool_call_data = self._normalize_tool_call(parsed)
        if tool_call_data is None:
            if is_promise:
                return False, reprimand, None
            return (
                False,
                "[SYSTEM ERROR: Validation failed. JSON does not contain a "
                'tool call. Expected format: {"name": "tool_name", '
                '"arguments": {...}}. Try again.]',
                None,
            )

        name = tool_call_data.get("name", "")
        args = tool_call_data.get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return (
                    False,
                    f"[SYSTEM ERROR: Validation failed. Tool '{name}' "
                    f"arguments are not valid JSON.]",
                    None,
                )

        # Tier 2: Schema
        valid, err = self._validate_schema(name, args)
        if not valid:
            return False, err, None

        # Construct proper OpenAI tool_calls response
        constructed = self._construct_tool_call_response(name, args, response_data)
        return True, None, constructed

    # ── Tier 1: JSON Extraction ───────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """
        Extract JSON from raw text.

        Strategy:
            1. Try markdown code fences (```json ... ``` or ``` ... ```)
            2. Fall back to outermost balanced-brace extraction
        """
        # Strategy 1: Markdown fenced code blocks (grab the last one)
        md_matches = list(re.finditer(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL))
        if md_matches:
            candidate = md_matches[-1].group(1).strip()
            if candidate.startswith("{"):
                return candidate

        # Strategy 2: Last balanced brace block
        start = text.rfind("{")
        if start == -1:
            return None

        depth = 0
        blocks = []
        current_start = -1
        in_string = False
        escape_next = False

        for i, c in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if c == "\\" and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue

            if c == "{":
                if depth == 0:
                    current_start = i
                depth += 1
            elif c == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and current_start != -1:
                        blocks.append(text[current_start:i+1])
                        current_start = -1

        if blocks:
            return blocks[-1]
            
        return None

    # ── Tier 1.5: Heuristic Intent Detection ──────────────────────────────

    def _detect_empty_promise(self, content: str) -> tuple[bool, str | None]:
        """
        Tier 1.5: Scan text for high-confidence action intents without tool calls.

        Uses a multi-gate approach:
            Gate 1 — Explicit programmatic tool names (e.g., `skill_view`, "skill_view")
            Gate 2 — Intent markers with execution verbs (e.g., "I will create", "Let me run")
            Gate 3 — Action gerunds in operational context (e.g., "creating ticket", "running script")

        Returns:
            (is_empty_promise, reprimand_or_none)
        """
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return False, None

        reprimand = self._EMPTY_PROMISE_REPRIMAND
        if self.tool_registry:
            avail_tools = ", ".join(f"'{t}'" for t in list(self.tool_registry.keys())[:5])
            reprimand = (
                f"{self._EMPTY_PROMISE_REPRIMAND} Available tools: [{avail_tools}]. "
                "Do not state intent in text — output the structured tool call immediately."
            )

        # Gate 1: Check for explicit programmatic tool names mentioned in prose.
        # We match names enclosed in backticks or quotes.
        if self.tool_registry:
            names = "|".join(re.escape(name) for name in self.tool_registry.keys())
            # Match `tool_name` or "tool_name" or 'tool_name'
            tool_name_pattern = re.compile(rf"([`'\"])({names})\1", re.IGNORECASE)
            if tool_name_pattern.search(content):
                return True, reprimand

        # Gates 2 & 3: Broad sequential verbs & gerunds
        if self._INTENT_MARKERS.search(content) or self._ACTION_GERUNDS.search(content):
            return True, reprimand

        return False, None

    # ── Normalization ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_tool_call(parsed: dict) -> dict | None:
        """
        Normalize various tool call JSON formats to {"name": ..., "arguments": ...}.

        Handles:
            {"name": "x", "arguments": {...}}
            {"function": {"name": "x", "arguments": {...}}}
            {"tool_calls": [{"function": {"name": "x", ...}}]}
        """
        if "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
            # Direct format: accept "parameters" as an alias for "arguments"
            if "parameters" in parsed and "arguments" not in parsed:
                parsed["arguments"] = parsed.pop("parameters")
            return parsed

        if "function" in parsed and isinstance(parsed["function"], dict):
            return parsed["function"]

        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
            if parsed["tool_calls"]:
                tc = parsed["tool_calls"][0]
                if "function" in tc and isinstance(tc["function"], dict):
                    return tc["function"]
                if "name" in tc:
                    return tc

        return None

    # ── Tier 2: Schema Validation ─────────────────────────────────────────

    def _validate_schema(self, name: str, args: dict) -> tuple[bool, str | None]:
        """Validate tool name exists and arguments match the declared schema."""
        if name not in self.tool_registry:
            available = ", ".join(sorted(self.tool_registry.keys())) or "(none)"
            return (
                False,
                f"[SYSTEM ERROR: Validation failed. You attempted to call "
                f"tool '{name}' which is not in your registry. Available "
                f"tools: {available}. Try again.]",
            )

        schema = self.tool_registry[name]
        if not schema:
            return True, None  # No schema → accept any arguments

        # Rigorous validation via jsonschema
        if HAS_JSONSCHEMA:
            try:
                jsonschema.validate(instance=args, schema=schema)
                return True, None
            except jsonschema.ValidationError as e:
                required = schema.get("required", [])
                return (
                    False,
                    f"[SYSTEM ERROR: Validation failed. Tool '{name}' "
                    f"arguments invalid: {e.message}. Required fields: "
                    f"{json.dumps(required)}. Try again.]",
                )
            except jsonschema.SchemaError:
                pass  # Schema itself is broken → fall through to basic check

        # Fallback: basic key-presence and type checking
        return self._basic_schema_check(name, args, schema)

    def _basic_schema_check(
        self, name: str, args: dict, schema: dict
    ) -> tuple[bool, str | None]:
        """Stdlib-only fallback for Tier 2 when jsonschema is unavailable."""
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in args:
                return (
                    False,
                    f"[SYSTEM ERROR: Validation failed. Tool '{name}' "
                    f"missing required argument '{field}'. Try again.]",
                )

        _TYPE_MAP = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for key, value in args.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type:
                    expected_cls = _TYPE_MAP.get(expected_type)
                    if expected_cls and not isinstance(value, expected_cls):
                        return (
                            False,
                            f"[SYSTEM ERROR: Validation failed. Tool '{name}' "
                            f"argument '{key}' has wrong type. Expected "
                            f"{expected_type}, got {type(value).__name__}.]",
                        )

        return True, None

    # ── Response Construction ─────────────────────────────────────────────

    @staticmethod
    def _construct_tool_call_response(
        name: str, args: dict, original_response: dict
    ) -> dict:
        """
        Construct a proper OpenAI-format response with tool_calls
        from JSON that was extracted from raw text content.
        """
        call_id = f"call_gk_{uuid.uuid4().hex[:12]}"

        return {
            "id": f"chatcmpl-gk-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": original_response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": (
                                        json.dumps(args)
                                        if isinstance(args, dict)
                                        else args
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": original_response.get(
                "usage",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            ),
            "_gatekeeper": True,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Module-Level Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _extract_content(response_data: dict) -> str:
    """Extract text content from an LLM response for replay in context."""
    choices = response_data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if content:
            return content
        # Failed native tool_calls → serialize for context replay
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return json.dumps(tool_calls)
    return str(response_data)[:500]


def _fatal_error_response(model: str, message: str) -> dict:
    """
    Construct the fatal error mock tool call.

    Uses a unique function name (__frugallm_gatekeeper_fatal_error__)
    to ensure zero namespace collisions with user-defined tools.
    """
    return {
        "id": f"chatcmpl-gk-fatal-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_gk_fatal_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": "__frugallm_gatekeeper_fatal_error__",
                                "arguments": json.dumps(
                                    {"message": message}
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "_gatekeeper": True,
        "_gatekeeper_fatal": True,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Component C: The ASGI Middleware — Sole Retry Engine ("Clone and Trash")
# ═════════════════════════════════════════════════════════════════════════════

class GatekeeperMiddleware:
    """
    ASGI Middleware that wraps LiteLLM to enforce Gatekeeper validation.

    This is the sole retry engine. It intercepts chat/completions requests,
    forces stream=False for validation, and internally loops the underlying
    ASGI app if tool call hallucinations are detected.

    Context Management ("Clone and Trash"):
        On each retry, the previous hallucinated response + system reprimand
        are sliced out of volatile_messages before appending the new failure
        context. This prevents VRAM explosion and context contamination.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if not (scope["method"] == "POST" and "chat/completions" in scope.get("path", "")):
            return await self.app(scope, receive, send)

        # ── Read body and reconstruct receive stream ─────────────────────
        more_body = True
        body = b""
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        try:
            req_json = json.loads(body)
        except Exception:
            req_json = {}

        if not gatekeeper_config.should_intercept(
            req_json.get("model", ""), req_json.get("tools", [])
        ):
            # Not intercepting — pass through with reconstructed stream
            async def receive_passthrough():
                return {"type": "http.request", "body": body, "more_body": False}
            return await self.app(scope, receive_passthrough, send)

        # ── GATEKEEPER INTERCEPTION ──────────────────────────────────────
        log.info(f"🛡️ Gatekeeper intercepting ASGI request for {req_json.get('model')}")

        original_messages = copy.deepcopy(req_json.get("messages", []))
        volatile_messages = copy.deepcopy(original_messages)
        tools = req_json.get("tools", [])
        validator = ToolCallValidator(tools)

        # Force stream=False to validate JSON response
        was_streaming = req_json.get("stream", False)
        if was_streaming:
            log.info("🛡️ Gatekeeper disabling 'stream' to validate response.")
            req_json["stream"] = False

        max_retries = gatekeeper_config.max_retries
        timeout_seconds = gatekeeper_config.timeout_seconds
        prev_retry_start_idx: int | None = None  # Track retry artifact position

        for attempt in range(1, max_retries + 1):
            log.info(f"🛡️ Gatekeeper attempt {attempt}/{max_retries}")

            req_json["messages"] = volatile_messages
            new_body = json.dumps(req_json).encode("utf-8")

            # Update Content-Length for the mutated body
            headers = dict(scope.get("headers", []))
            headers[b"content-length"] = str(len(new_body)).encode("utf-8")

            # Create a fresh scope copy for this attempt
            attempt_scope = scope.copy()
            attempt_scope["headers"] = [(k, v) for k, v in headers.items()]

            # Reconstruct receive for this attempt
            async def attempt_receive(_body=new_body):
                return {"type": "http.request", "body": _body, "more_body": False}

            response_body = b""
            response_headers = []
            response_status = 200

            # Capture send to intercept the LLM response
            async def attempt_send(message):
                nonlocal response_body, response_headers, response_status
                if message["type"] == "http.response.start":
                    response_status = message["status"]
                    response_headers = message.get("headers", [])
                elif message["type"] == "http.response.body":
                    response_body += message.get("body", b"")

            try:
                await asyncio.wait_for(
                    self.app(attempt_scope, attempt_receive, attempt_send),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                log.error(f"🛡️ Gatekeeper timeout ({timeout_seconds}s) on attempt {attempt}")
                fatal_resp = _fatal_error_response(
                    req_json.get("model", "unknown"),
                    f"LLM timeout after {timeout_seconds}s.",
                )
                fatal_resp["_gatekeeper_attempts"] = attempt
                await self._send_json_response(send, fatal_resp, 200, was_streaming)
                return

            # Non-200 → upstream error, pass through directly
            if response_status != 200:
                await send({"type": "http.response.start", "status": response_status, "headers": response_headers})
                await send({"type": "http.response.body", "body": response_body})
                return

            try:
                resp_json = json.loads(response_body)
            except Exception as e:
                log.warning(f"🛡️ Gatekeeper failed to parse response JSON (likely hit max tokens): {e}")
                
                # If Hermes requested a stream, wrap the truncated content in an SSE chunk
                if was_streaming:
                    raw_text = response_body.decode("utf-8", errors="ignore")
                    sse_payload = (
                        f"data: {json.dumps({'choices': [{'delta': {'content': raw_text}, 'finish_reason': 'length'}]})}\n\n"
                        "data: [DONE]\n\n"
                    ).encode("utf-8")
                    
                    headers = [
                        (b"content-type", b"text/event-stream"),
                        (b"content-length", str(len(sse_payload)).encode("utf-8")),
                    ]
                    await send({"type": "http.response.start", "status": response_status, "headers": headers})
                    await send({"type": "http.response.body", "body": sse_payload})
                    return
                
                # Non-streaming passthrough fallback
                await send({"type": "http.response.start", "status": response_status, "headers": response_headers})
                await send({"type": "http.response.body", "body": response_body})
                return

            # ── Validate the response ────────────────────────────────────
            is_valid, error_msg, validated_result = await asyncio.to_thread(validator.validate_response, resp_json)

            if is_valid:
                log.info(f"✅ Gatekeeper validation PASSED on attempt {attempt}.")
                validated_result["_gatekeeper_attempts"] = attempt
                await self._send_json_response(send, validated_result, 200, was_streaming)
                return

            # ── Validation failed — prepare retry ────────────────────────
            log.warning(f"🛡️ Gatekeeper FAILED on attempt {attempt}: {error_msg}")

            if attempt >= max_retries:
                break  # Exhausted — fall through to fatal response

            # ── Context Management: Clone and Trash ──────────────────────
            # On retry 2+, remove previous hallucination + error feedback
            # to prevent context window exhaustion and VRAM explosion.
            if prev_retry_start_idx is not None:
                del volatile_messages[prev_retry_start_idx:]
                log.debug(
                    f"🛡️ Removed previous retry artifacts from index "
                    f"{prev_retry_start_idx}"
                )

            # Record where we're inserting new retry artifacts
            prev_retry_start_idx = len(volatile_messages)

            # Append the LLM's hallucinated response for context
            hallucinated_content = _extract_content(resp_json)
            volatile_messages.append({"role": "assistant", "content": hallucinated_content})

            # Append the targeted system reprimand
            volatile_messages.append({"role": "system", "content": error_msg})

            log.debug(
                f"🛡️ Injected error feedback. Volatile context: "
                f"{len(volatile_messages)} messages "
                f"(+2 from original {prev_retry_start_idx})"
            )

        # ── Exhausted retries ────────────────────────────────────────────
        log.error(f"💥 Gatekeeper exhausted all {max_retries} retries")
        fatal_resp = _fatal_error_response(
            req_json.get("model", "unknown"),
            "Max retries exceeded catching tool hallucinations.",
        )
        fatal_resp["_gatekeeper_attempts"] = max_retries
        await self._send_json_response(send, fatal_resp, 200, was_streaming)

    @staticmethod
    async def _send_json_response(send, data: dict, status: int, was_streaming: bool = False):
        """Send a JSON response with properly calculated content-length, or SSE chunks."""
        if was_streaming:
            body = f"data: {json.dumps(data)}\n\n".encode("utf-8")
            headers = [
                (b"content-type", b"text/event-stream; charset=utf-8"),
                (b"transfer-encoding", b"chunked"),
            ]
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": body, "more_body": True})
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})
        else:
            body = json.dumps(data).encode("utf-8")
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("utf-8")),
            ]
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": body})


# ═════════════════════════════════════════════════════════════════════════════
# Module-Level Singletons
# ═════════════════════════════════════════════════════════════════════════════
gatekeeper_config = GatekeeperConfig()

log.info(f"🛡️ Gatekeeper module loaded: {gatekeeper_config}")
