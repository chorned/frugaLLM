# FrugaLLM 2.0 — Troubleshooting

## Common Issues

### Proxy Not Starting

**Symptom:** `litellm` command not found or import errors.

**Fix:**
```bash
# Ensure you're in the right venv
source venv/bin/activate
pip install -r requirements.txt

# Verify litellm is installed
python -m litellm --version
```

---

### ModuleNotFoundError: No module named 'prisma'

**Symptom:** Gateway returns 500 internal_server_error on all requests when PostgreSQL logging is enabled.

**Fix:**
```bash
pip install prisma
prisma generate
```

> **Note:** This only happens if you enable PostgreSQL spend logging via `DATABASE_URL`. If you don't need spend logging, remove or comment out the `DATABASE_URL` from your `.env`.

---

### Ollama Timeout Issues

**Symptom:** `litellm.Timeout: Connection timed out. Timeout passed=10.0`

**Fix:** Increase timeout for Ollama endpoints in `config/litellm_config.yaml`:

```yaml
- model_name: local
  litellm_params:
    model: ollama/llama3.2:latest
    api_base: http://127.0.0.1:11434
    timeout: 60  # Was 10, increase to 60
    max_retries: 0
```

---

### Dynamic Models Not Updating

**Symptom:** Sidecar stuck in "Waiting for LiteLLM to come online..."

**Fix:**
1. Check sidecar logs: `tail -f /tmp/frugallm-sidecar.log`
2. Verify the proxy is running: `curl http://localhost:4000/health`
3. Check that `OPENROUTER_API_KEY` is set in your `.env`
4. Verify `dynamic_models.yaml` is being included:
   ```bash
   head config/dynamic_models.yaml
   ```

---

### SIGHUP Not Reloading Config

**Symptom:** Sidecar says "SIGHUP failed" or "Could not find LiteLLM process"

**Fix:** The sidecar looks for LiteLLM processes matching `litellm.*--config`. If you're running LiteLLM differently:
```bash
# Check what processes are running
pgrep -f litellm

# Manually reload
kill -HUP $(pgrep -f "litellm.*--config")
```

If SIGHUP isn't supported by your LiteLLM version, restart the proxy manually:
```bash
make stop && make start-bg
```

---

### GPU Node Unreachable (Tailscale)

**Symptom:** Requests to the GPU node time out.

**macOS Fix:** macOS Sequoia's "Local Network Privacy" blocks LAN IPs. Use Tailscale IPs instead of raw LAN IPs:
```bash
# Bad:  FRUGALLM_GPU_URL=http://192.168.1.100:8080/v1
# Good: FRUGALLM_GPU_URL=http://100.x.y.z:8080/v1  (use your Tailscale IP)
```

---

### Rate Limiting on Free Models

**Symptom:** Models returning 429 errors frequently.

**Fix:** This is expected behavior — free models have rate limits. FrugaLLM handles this automatically:
1. The failing model is benched for the `Retry-After` duration
2. The next model in the fallback chain picks up
3. The sidecar refreshes the roster every 5 minutes

To reduce rate limit hits:
- Enable response caching (on by default, 5-minute TTL)
- Use a separate `OPENROUTER_MANAGEMENT_KEY` for the sidecar's model polling

---

### Anti-Hijack Middleware Issues

**Symptom:** The model seems to be ignoring your system prompt.

**Fix:** The anti-hijack middleware is designed to override upstream persona injection. If it's interfering with your use case, you can disable it by:
1. Removing the `callbacks` line from `config/litellm_config.yaml`
2. Or creating a custom callback that doesn't include `_enforce_anti_hijack`

---

## Debugging Commands

```bash
# Check proxy health
curl -s -H "Authorization: Bearer sk-frugallm-master" http://localhost:4000/health | python3 -m json.tool

# List active models
curl -s -H "Authorization: Bearer sk-frugallm-master" http://localhost:4000/v1/models | python3 -m json.tool

# Test a specific model
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-frugallm-master" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Quick check"}]}'

# Check dynamic models
cat config/dynamic_models.yaml

# View proxy logs
tail -f /tmp/frugallm-proxy.log

# View sidecar logs
tail -f /tmp/frugallm-sidecar.log
```
