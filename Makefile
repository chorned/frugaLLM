# FrugaLLM 2.0 — Makefile
# Convenience commands for managing the FrugaLLM stack.

.PHONY: help install start stop sidecar logs status test clean

PROXY_PORT ?= 4000
MASTER_KEY ?= sk-frugallm-master
CONFIG     ?= config/litellm_config.yaml
PYTHON     ?= python3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

start: ## Start the LiteLLM proxy
	@echo "Starting FrugaLLM proxy on port $(PROXY_PORT)..."
	$(PYTHON) -m litellm --config $(CONFIG) --port $(PROXY_PORT)

start-bg: ## Start the LiteLLM proxy in the background
	@echo "Starting FrugaLLM proxy in the background..."
	nohup $(PYTHON) -m litellm --config $(CONFIG) --port $(PROXY_PORT) > /tmp/frugallm-proxy.log 2>&1 &
	@echo "PID: $$!"
	@echo "Logs: /tmp/frugallm-proxy.log"

sidecar: ## Start the dynamic roster sidecar
	@echo "Starting FrugaLLM sidecar..."
	FRUGALLM_CONFIG_DIR=config $(PYTHON) -m frugallm.dynamic_roster_sidecar

sidecar-bg: ## Start the sidecar in the background
	@echo "Starting FrugaLLM sidecar in the background..."
	nohup FRUGALLM_CONFIG_DIR=config $(PYTHON) -m frugallm.dynamic_roster_sidecar > /tmp/frugallm-sidecar.log 2>&1 &
	@echo "PID: $$!"
	@echo "Logs: /tmp/frugallm-sidecar.log"

stop: ## Stop all FrugaLLM processes
	@echo "Stopping FrugaLLM processes..."
	-pkill -f "litellm.*--config" 2>/dev/null || true
	-pkill -f "frugallm.dynamic_roster_sidecar" 2>/dev/null || true
	@echo "Done."

logs: ## Tail proxy and sidecar logs
	@echo "=== Proxy Log ===" && tail -20 /tmp/frugallm-proxy.log 2>/dev/null || echo "(not found)"
	@echo ""
	@echo "=== Sidecar Log ===" && tail -20 /tmp/frugallm-sidecar.log 2>/dev/null || echo "(not found)"

status: ## Check if the proxy is running and healthy
	@echo "Checking FrugaLLM proxy health..."
	@curl -sf -H "Authorization: Bearer $(MASTER_KEY)" http://localhost:$(PROXY_PORT)/health | python3 -m json.tool 2>/dev/null \
		|| echo "❌ Proxy is not responding on port $(PROXY_PORT)"
	@echo ""
	@echo "Active models:"
	@curl -sf -H "Authorization: Bearer $(MASTER_KEY)" http://localhost:$(PROXY_PORT)/v1/models | python3 -m json.tool 2>/dev/null \
		|| echo "❌ Cannot fetch models"

test: ## Send a test request to the proxy
	@echo "Sending test request to FrugaLLM..."
	@curl -sf -X POST http://localhost:$(PROXY_PORT)/v1/chat/completions \
		-H "Authorization: Bearer $(MASTER_KEY)" \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}]}' \
		| python3 -m json.tool

clean: ## Remove generated files and caches
	rm -f config/dynamic_models.yaml.tmp
	rm -rf __pycache__ frugallm/__pycache__
	@echo "Cleaned."

docker-up: ## Start the Docker Compose stack
	docker compose up -d

docker-down: ## Stop the Docker Compose stack
	docker compose down

docker-full: ## Start with PostgreSQL spend logging
	docker compose --profile full up -d
