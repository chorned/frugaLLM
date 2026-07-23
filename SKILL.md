---
name: frugallm-gateway
description: Manage FrugaLLM 3.0 Gateway Stack — zero-cost AI routing proxy with ONNX neural empty promise classifier, FastAPI Gatekeeper, automatic free model discovery, Gemini fleet, and Langfuse telemetry
skill_type: procedure
applies_to: [gateway, litellm, routing, devops, llm-proxy, docker, gatekeeper, langfuse, Gemini]
triggers:
  - "FrugaLLM gateway issue"
  - "gateway timeout"
  - "restart proxy"
  - "check dynamic models"
  - "free model routing"
  - "LiteLLM proxy"
  - "FrugaLLM container stack"
  - "run integration test suite"
  - "test frugaLLM"
version: 3.0.0
---

# FrugaLLM 3.0 — Gateway Management Skill

This skill covers the operation, troubleshooting, testing, and maintenance of the FrugaLLM 3.0 Gateway Stack — a self-healing LLM proxy stack that automatically discovers and routes through free models on OpenRouter, with zero-shot neural empty-promise classification, paid Gemini 3.6 Flash terminal fallback, 100% Langfuse telemetry tracking, and intuitive pseudo-model aliases for agents and users.

---

## Architecture Overview

FrugaLLM 3.0 runs as a multi-service Docker Compose stack:

1. **🛡️ Gatekeeper Gateway (`:5050`)** — FastAPI reverse proxy entrypoint. Intercepts chat completions, inspects output for tool call integrity, and manages the internal retry loop when empty promises occur.
2. **🧠 Micro-Classifier (`:8000`, Internal)** — CPU-optimized ONNX zero-shot NLI classifier (`cross-encoder/nli-deberta-v3-small`). Semantically detects intent-to-act vs standard text.
3. **⚡ LiteLLM Proxy (`:4000`, Internal)** — Model routing core. Manages free model selection, fallbacks, and Langfuse telemetry hooks.
4. **🔄 Dynamic Roster Sidecar** — Background daemon that scans OpenRouter every 5 minutes and writes candidate chains to `config/dynamic_models.yaml`.
5. **🐘 PostgreSQL DB (`:5432`)** — Spend logging and usage tracking.

```
Clients (Hermes / Apps)
   │ (port 5050)
   ▼
[ 🛡️ Gatekeeper ] ──(classify text)──▶ [ 🧠 Micro-Classifier (:8000) ]
   │                                           │ (returns is_empty_promise)
   │ (forward / internal retry) ◄──────────────┘
   ▼
[ ⚡ LiteLLM Router (:4000) ] ──(telemetry)──▶ [ 📊 Langfuse Cloud ]
   │
   ├─► OpenRouter Free Roster (Balanced / Reasoning)
   ├─► Local Gemma / Ollama
   └─► Paid Gemini Fleet Terminal Fallback (Gemini 3.6 Flash)
```

---

## Model Aliases & Routing Roster

Main API Base URL: `http://localhost:5050/v1`

### Intuitive Pseudo-Model Aliases (Recommended)

| Intuitive Alias | Legacy Alias | Routing Strategy / Fallback Chain | Primary Use Case |
|---|---|---|---|
| **`frugal`** / **`smart`** | `auto` | Local Gemma 4 12B → OpenRouter Free → Paid Gemini 3.6 Flash | General tasks, coding, Q&A (default) |
| **`thinker`** / **`reasoner`** | `reasoning` | Local Gemma 4 12B (CoT) → Free Reasoning → Paid Gemini 3.6 Flash | Complex architecture, math, multi-step planning |
| **`offline`** / **`private`** | `local` | Ollama `hermes:latest` (100% local CPU) | Sensitive data, offline work |
| **`free`** | `free_balanced` | Top OpenRouter free model (dynamic 5-min pool) | Zero-cost cloud execution |
| **`cloud`** | `gemini-flash` | Paid Gemini 3.6 Flash ($1.50 / $7.50 per 1M) | Guaranteed uptime, large context |
| **`fast`** / **`lite`** | `gemini-flash-lite` | Paid Gemini 3.5 Flash-Lite ($0.30 / $2.50 per 1M) | Low latency, high-volume subagent tasks |
| `gemini-pro` | `gemini-pro` | Gemini 2.5 Pro (legacy alias) | Deep reasoning fallback |

> **Backwards Compatibility**: All legacy aliases (`auto`, `reasoning`, `local`, `gemini-flash`, `gemini-pro`) remain 100% supported.
>
> **Best Practice for Stateful / Multi-turn Sessions**: Dynamic free models (`frugal`, `free`) cap dynamic fallback chains to 2 hops before dropping to `gemini-flash`. For long-running, multi-file agentic tasks with complex tool calling, use **`cloud`** (`gemini-flash`) or **`thinker`** to avoid context shifts across different model providers mid-session.

---

## Router CLI & Test Commands

### 1. FrugaLLM Router CLI

The interactive CLI wrapper communicates directly with the Gateway:

```bash
# General query (uses frugal / auto alias)
python -m frugallm.router_cli "Explain quantum computing"

# Use the deep reasoning / thinker alias
python -m frugallm.router_cli --thinker "Design a microservice architecture"

# Route to paid Gemini cloud tier directly
python -m frugallm.router_cli --cloud "Summarize this paper"

# Force offline CPU execution
python -m frugallm.router_cli --offline "Process local logs"

# Inspect gateway health & active roster
python -m frugallm.router_cli --models
```

### 2. Integration Test Suite

The stack includes a comprehensive, dependency-free test runner covering 23+ assertion check points across 5 test groups:

```bash
# Run full integration test suite (includes live cloud & dynamic model tests)
make test-suite

# Or run directly via Python
python3 tests/test_integration.py

# CLI Options:
python3 tests/test_integration.py --skip-cloud    # Skip paid API tests (Gemini fleet)
python3 tests/test_integration.py --skip-dynamic   # Skip OpenRouter dynamic model tests
python3 tests/test_integration.py --verbose        # Show detailed response bodies
```

---

## Telemetry & Langfuse Tracking

FrugaLLM integrates with **Langfuse Cloud** for 100% endpoint usage tracking:
- **`success_callback` & `failure_callback`**: Hooks registered in `config/litellm_config.yaml`.
- **Model Name Normalizer**: `custom_callbacks.py` prepends `openrouter/` to OpenRouter model logs, preventing duplicate entries in Langfuse dashboards.
- **Metadata Forwarding**: Custom client tags, trace IDs, and autonomous tier annotations are forwarded automatically.

Verify telemetry in tests via Group 5 of `make test-suite`.

---

## Management Commands

```bash
# Start container stack
docker compose up -d

# Rebuild and start after changes
docker compose up -d --build

# View logs for all services
docker compose logs -f

# View logs for specific service
docker compose logs -f gatekeeper
docker compose logs -f litellm
docker compose logs -f sidecar
```

---

## Health Checks & Debugging

```bash
# Gatekeeper Gateway Health
curl -s http://localhost:5050/health | python3 -m json.tool

# Test chat completion with new 'frugal' alias
curl -X POST http://localhost:5050/v1/chat/completions \
  -H "Authorization: Bearer sk-sidecar-1" \
  -H "Content-Type: application/json" \
  -d '{"model": "frugal", "messages": [{"role": "user", "content": "Hello"}]}'

# Inspect dynamic models written by sidecar
cat config/dynamic_models.yaml
```
