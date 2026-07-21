# FrugaLLM 3.0 — Makefile
# Convenience commands for managing the FrugaLLM container stack.

.PHONY: help up down build logs status test clean

GATEWAY_PORT ?= 5050
MASTER_KEY   ?= sk-frugallm-master

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start the FrugaLLM container stack
	docker compose up -d

down: ## Stop the FrugaLLM container stack
	docker compose down

build: ## Rebuild container images and start stack
	docker compose up -d --build

logs: ## Tail container logs
	docker compose logs -f

status: ## Check Gatekeeper health and container status
	@echo "=== Container Status ==="
	@docker compose ps
	@echo ""
	@echo "=== Gatekeeper Gateway Health (Port $(GATEWAY_PORT)) ==="
	@curl -sf http://localhost:$(GATEWAY_PORT)/health | python3 -m json.tool 2>/dev/null \
		|| echo "❌ Gatekeeper is not responding on port $(GATEWAY_PORT)"

test: ## Send a test request to the Gatekeeper gateway
	@echo "Sending test request to FrugaLLM Gatekeeper..."
	@curl -sf -X POST http://localhost:$(GATEWAY_PORT)/v1/chat/completions \
		-H "Authorization: Bearer $(MASTER_KEY)" \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}]}' \
		| python3 -m json.tool

clean: ## Clean Python caches and temp files
	rm -f config/dynamic_models.yaml.tmp
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleaned."

