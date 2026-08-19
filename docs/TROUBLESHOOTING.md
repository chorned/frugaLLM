# FrugaLLM 3.0 — Troubleshooting

## Common Issues

### Container Stack Not Starting

**Symptom:** `docker compose up -d` fails or containers exit immediately.

**Fix:**
```bash
# Check container status
docker compose ps

# Inspect logs of failing container
docker compose logs -f gatekeeper
docker compose logs -f litellm
docker compose logs -f classifier
```

---

### Gatekeeper 502 / Upstream Connection Failed

**Symptom:** Client receives HTTP 502 or "Upstream connection failed".

**Fix:**
1. Verify LiteLLM and Classifier containers are healthy:
   ```bash
   docker compose ps
   ```
2. The Gatekeeper container routes internally to `http://litellm:4000` and `http://classifier:8000`. Ensure all containers are attached to `frugallm-net`.

---

### ModuleNotFoundError: No module named 'prisma'

**Symptom:** LiteLLM container logs error on PostgreSQL spend logging initialization.

**Fix:**
```bash
# Verify POSTGRES_PASSWORD is set in .env
grep POSTGRES_PASSWORD .env
```

---

### Ollama Timeout Issues

**Symptom:** `litellm.Timeout: Connection timed out.`

**Fix:** Increase timeout for Ollama endpoints in `config/litellm_config.yaml`:

```yaml
- model_name: local
  litellm_params:
    model: ollama/llama3.2:latest
    api_base: http://127.0.0.1:11434
    timeout: 60
    max_retries: 0
```

---

### Dynamic Models Not Updating

**Symptom:** Sidecar container stuck or free model roster empty.

**Fix:**
1. Check sidecar container logs:
   ```bash
   docker compose logs -f sidecar
   ```
2. Verify `OPENROUTER_API_KEY` is set in `.env`.
3. Verify `dynamic_models.yaml` is present in `config/`:
   ```bash
   cat config/dynamic_models.yaml
   ```

---

## Debugging Commands

```bash
# Check Gatekeeper gateway health (Port 5050)
curl -s http://localhost:5050/health | python3 -m json.tool

# Test chat completion
curl -X POST http://localhost:5050/v1/chat/completions \
  -H "Authorization: Bearer sk-sidecar-1" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Quick check"}]}'

# View dynamic model roster
cat config/dynamic_models.yaml

# Tail container logs
docker compose logs -f
```

