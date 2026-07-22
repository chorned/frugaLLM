# FrugaLLM 3.0 — Configuration Reference

## Configuration Files

| File | Purpose | Auto-Generated? |
|------|---------|-----------------|
| `docker-compose.yml` | Container stack definition | No — edit freely |
| `config/litellm_config.yaml` | Main LiteLLM proxy configuration | No — edit freely |
| `config/dynamic_models.yaml` | Dynamic model roster from OpenRouter | Yes — by sidecar |
| `.env` | API keys and runtime configuration | No — edit freely |

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key for model inference. Get one at [openrouter.ai/keys](https://openrouter.ai/keys) |
| `POSTGRES_PASSWORD` | PostgreSQL database password for spend logging |

---

### FrugaLLM 3.0 Stack Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_URL` | `http://litellm:4000` | Gatekeeper → LiteLLM internal URL |
| `CLASSIFIER_URL` | `http://classifier:8000` | Gatekeeper → Classifier internal URL |
| `GATEKEEPER_MAX_RETRIES` | `3` | Maximum internal retries on empty promises |
| `GATEKEEPER_TIMEOUT` | `300` | Request timeout for Gatekeeper (seconds) |
| `FRUGALLM_MASTER_KEY` | `sk-sidecar-1` | Master authentication key for proxy |
| `FRUGALLM_POLL_INTERVAL` | `300` | Sidecar poll interval in seconds |

---

### Local & Fallback Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FRUGALLM_LOCAL_MODEL` | `llama3.2:latest` | Ollama model name for local fallback |
| `FRUGALLM_LOCAL_URL` | `http://127.0.0.1:11434` | Ollama base URL |

---

### Telemetry Settings (Optional)

| Variable | Description |
|----------|-------------|
| `LANGFUSE_SECRET_KEY` | Langfuse secret key for tracing |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_HOST` | Langfuse host URL (e.g., `https://us.cloud.langfuse.com`) |
| `DATABASE_URL` | PostgreSQL connection string for spend logging |

