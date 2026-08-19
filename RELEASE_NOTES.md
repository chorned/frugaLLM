# FrugaLLM 3.0.1 — Database Migration Release Notes

## Overview

This update replaces the legacy file-based dynamic model routing system with a robust, production-grade **PostgreSQL-backed Prisma database architecture**. By leveraging LiteLLM's native database integration, FrugaLLM now supports zero-downtime model roster updates, unified configuration management, and centralized spend logging.

## Key Changes

### 1. Database-Backed Routing
- **Prisma Database Integration:** LiteLLM now connects to a PostgreSQL database (`frugallm-postgres`) to store and manage the dynamic model roster, replacing the legacy `config/dynamic_models.yaml` file.
- **LiteLLM Admin API:** The Dynamic Roster Sidecar (`frugallm/dynamic_roster_sidecar.py`) now directly queries and updates the LiteLLM database via the `/model/new` and `/model/update` Admin API endpoints instead of regenerating a YAML configuration file.
- **Google AI Studio Support:** The Dynamic Roster Sidecar now actively scans Google AI Studio for high-quality free tier models (e.g. Gemini 2.5 Flash, Gemini Pro) and pools them with OpenRouter's free tier, significantly expanding the candidate pool for routing.

### 2. Infrastructure Updates
- **New Container:** A `postgres` container is now included in the `docker-compose.yml` stack, configured automatically with persistent volumes and health checks.
- **Port Exposure:** The internal LiteLLM proxy is now exposed on host port `4000` to allow the Sidecar (and other tools) to interface with the Admin API.
- **Environment Variables:** The `.env` file structure has been updated to include PostgreSQL credentials (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`) and `DATABASE_URL`.

### 3. Stability & Pipeline Improvements
- **Gatekeeper Crash Fixes:** Addressed internal unhandled exceptions and infinite timeout loops in the FastAPI reverse proxy when the classifier encountered malformed requests.
- **Router CLI Cleanup:** Removed stale functions in `router_cli.py`.
- **Pipeline Reliability:** Fixed critical test sandbox authentication issues by configuring the pipeline proxy environment dynamically.

## Deprecations
- **`config/dynamic_models.yaml`** is fully deprecated and has been removed from the stack.
- **File-based Fallbacks:** The router no longer attempts to hot-reload external YAML model files; all routing state is maintained dynamically in the database.
