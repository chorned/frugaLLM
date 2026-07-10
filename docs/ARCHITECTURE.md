# FrugaLLM 2.0 — Architecture

## System Overview

FrugaLLM is a **three-layer architecture** that sits between your applications and LLM providers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR APPLICATIONS                        │
│   (AI Agents, CLI Tools, Any OpenAI-Compatible Client)          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ OpenAI API (POST /v1/chat/completions)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1: LiteLLM PROXY                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Router      │  │  Middleware   │  │  Response Cache      │   │
│  │   (Fallback   │  │  (Anti-Hijack │  │  (In-Memory, 5 min) │   │
│  │    Chains)    │  │   Thought Sig │  │                      │   │
│  │              │  │   Reasoning)  │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  Config: config/litellm_config.yaml + config/dynamic_models.yaml │
│  Port:   4000 (configurable via FRUGALLM_PROXY_PORT)             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 2: DYNAMIC ROSTER SIDECAR                 │
│                                                                  │
│  Polls OpenRouter /models API every 5 minutes.                   │
│  Filters for: free pricing + tool support + 256k+ context.       │
│  Classifies: balanced vs. reasoning (heuristic-based).           │
│  Writes: config/dynamic_models.yaml with fallback chains.        │
│  Signals: SIGHUP to LiteLLM for zero-downtime reload.           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: MODEL BACKENDS                       │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │  OpenRouter     │  │  Local Ollama   │  │  GPU Node        │   │
│  │  (Free Models)  │  │  (Fallback)     │  │  (Optional)      │   │
│  │                 │  │                 │  │                   │   │
│  │  Balanced Chain │  │  llama3.2       │  │  Custom GGUF     │   │
│  │  Reasoning Chain│  │  or any model   │  │  via Tailscale   │   │
│  └────────────────┘  └────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Request Flow

1. **Client** sends a standard OpenAI API request to `http://localhost:4000/v1/chat/completions`
2. **LiteLLM Proxy** receives the request and applies middleware hooks:
   - Anti-Hijack: Appends an override to the system message
   - Gemini Thought Signatures: Ensures `thought_signature` fields are present
3. **Router** resolves the model alias (`auto`, `reasoning`, `local`, etc.) to an actual model
4. **Fallback Chain** executes: if the primary model fails, the next one in the chain is tried
5. **Response** is returned to the client, with reasoning fields extracted and surfaced

## Fallback Chain Design

The sidecar generates a **linear fallback chain** for each model category:

```
auto → free_balanced → free_balanced_2 → ... → free_balanced_N → free_balanced_backup (Ollama)
reasoning → free_reasoning → free_reasoning_2 → ... → free_reasoning_N → free_reasoning_backup
```

This means:
- If `auto` fails → tries the first free balanced model
- If that fails → tries the next one in the chain
- If ALL cloud models fail → falls back to local Ollama
- The chain is rebuilt every 5 minutes based on the current OpenRouter roster

## Component Details

### LiteLLM Proxy (`config/litellm_config.yaml`)

The proxy is the central routing layer. It provides:
- **OpenAI-compatible API** on port 4000
- **Model aliasing** (`auto`, `reasoning`, `local`)
- **Router settings** (retry strategy, timeouts, allowed failures)
- **Telemetry** (Langfuse, Prometheus, PostgreSQL)
- **Response caching** (in-memory, 5-minute TTL)

### Dynamic Roster Sidecar (`frugallm/dynamic_roster_sidecar.py`)

The sidecar is a daemon that runs alongside the proxy:
- **Polls** OpenRouter `/api/v1/models` every 5 minutes
- **Filters** for free models with tool support and large context windows
- **Classifies** models as balanced or reasoning using keyword heuristics
- **Writes** `config/dynamic_models.yaml` with numbered aliases and fallback chains
- **Signals** LiteLLM via SIGHUP for zero-downtime config reload

### Custom Callbacks (`frugallm/custom_callbacks.py`)

Three middleware hooks that modify requests and responses:
- **Anti-Hijack**: Pre-call hook that defeats upstream persona injection
- **Gemini Thought Signatures**: Pre-call hook that mocks missing fields
- **Reasoning Extractor**: Post-call hook that surfaces hidden reasoning

### Router CLI (`frugallm/router_cli.py`)

A lightweight CLI tool for quick queries against the gateway.

## Legacy Architecture

The `legacy/router_server.py` file is the original monolithic implementation that predates the LiteLLM-based architecture. It includes all routing, caching, middleware, and model discovery in a single 750-line Python file using only stdlib. It's included for reference and as a fallback for environments where LiteLLM cannot be installed.
