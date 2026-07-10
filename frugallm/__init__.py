"""
FrugaLLM 2.0 — Zero-Cost AI Routing with Automatic Free Model Discovery
=========================================================================

A self-healing LLM proxy that automatically discovers and routes through
the best available free models on OpenRouter, with local Ollama fallback,
intelligent caching, and middleware hooks.

Components:
  - custom_callbacks:         Anti-hijack, thought signatures, reasoning extractor
  - dynamic_roster_sidecar:   OpenRouter free model scanner daemon
  - router_cli:               Lightweight CLI wrapper for the gateway
"""

__version__ = "2.0.0"
__project__ = "FrugaLLM"
