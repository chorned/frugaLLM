# FrugaLLM 3.0 — Architecture

## System Overview

FrugaLLM 3.0 is a **containerized, microservice-based AI gateway stack** that sits between client applications (Hermes, AI Agents, CLI tools) and model backends:

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR APPLICATIONS                        │
│   (AI Agents, Hermes, CLI Tools, Any OpenAI Client)             │
└──────────────────────────────┬──────────────────────────────────┘
                               │ OpenAI API (POST /v1/chat/completions)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 1: GATEKEEPER GATEWAY (:5050)                │
│  FastAPI Reverse Proxy & Internal Retry Engine                  │
│                                                                 │
│  - Intercepts chat completions & streams                       │
│  - Validates output text for "empty promise" hallucinations     │
│  - Manages internal retry loops + system reprimands             │
└──────────────┬─────────────────────────────────┬────────────────┘
               │ 1. Forward Request              │ 2. Check Text
               ▼                                 ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│   LAYER 2: LiteLLM ROUTER    │  │   LAYER 2: MICRO-CLASSIFIER  │
│   (Internal Port :4000)      │  │   (Internal Port :8000)      │
│                              │  │                              │
│  - Dynamic Free Roster       │  │  - CPU ONNX Runtime          │
│  - Fallback Chains           │  │  - DeBERTa-v3 NLI Neural Model│
│  - Custom Callbacks          │  │  - Zero-shot Intent Detection│
│  - Response Cache & DB Logging│ │                              │
└──────────────┬───────────────┘  └──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: MODEL BACKENDS                       │
│                                                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │  OpenRouter     │  │  Local Ollama   │  │  GPU Node        │   │
│  │  (Free Models)  │  │  (Fallback)     │  │  (Optional)      │   │
│  │                 │  │                 │  │                  │   │
│  │  Balanced Chain │  │  llama3.2       │  │  Custom GGUF     │   │
│  │  Reasoning Chain│  │  or any model   │  │  via Tailscale   │   │
│  └────────────────┘  └────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Request Flow

1. **Client** sends a standard OpenAI API request to `http://localhost:5050/v1/chat/completions` (the Gatekeeper entrypoint).
2. **Gatekeeper Gateway** receives the request and forwards it to the internal LiteLLM Router (`litellm:4000`).
3. **LiteLLM Router** applies low-level callback hooks (Anti-Hijack persona protection, Gemini thought signatures, reasoning extraction) and routes to OpenRouter or local backends via linear fallback chains.
4. **Gatekeeper Inspection**:
   - If response has structured `tool_calls` → passed through immediately.
   - If response is text-only → sent to the **Micro-Classifier** (`classifier:8000`).
5. **Auto-Retry Loop**: If the classifier detects an "empty promise" (intent to act without tool JSON), Gatekeeper injects a system reprimand and retries upstream internally up to `GATEKEEPER_MAX_RETRIES` times.
6. **Response Delivery**: Validated response is returned to the client (re-wrapped as SSE stream chunks if streaming was requested).

---

## Component Details

### 🛡️ Gatekeeper Gateway (`gatekeeper/`)
- Entrypoint microservice exposed on host port `5050`.
- FastAPI reverse proxy built with connection pooling.
- Keeps client applications (Hermes) clean of intermediate failure retries.

### 🧠 Micro-Classifier (`classifier/`)
- Internal microservice on port `8000`.
- Runs `cross-encoder/nli-deberta-v3-small` ONNX model.
- Zero-shot natural language inference for semantic empty-promise detection with sub-millisecond CPU latency.

### ⚡ LiteLLM Router (`config/litellm_config.yaml`)
- Internal model router on port `4000`.
- Manages free model selection, fallbacks, and PostgreSQL spend logging.

### 🔄 Dynamic Roster Sidecar (`frugallm/dynamic_roster_sidecar.py`)
- Background container scanning OpenRouter every 5 minutes.
- Auto-updates `config/dynamic_models.yaml` with the latest free model roster.

