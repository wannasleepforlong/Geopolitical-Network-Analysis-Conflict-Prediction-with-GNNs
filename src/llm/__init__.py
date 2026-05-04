"""
src.llm package
================
Swappable LLM clients: Mistral (default) and Gemini (optional).

Modules:
    mistral_client  – Mistral AI chat completions
    gemini_client   – Google Gemini REST wrapper
    factory         – LLMFactory.create(provider)
"""
from .factory import LLMFactory
from .mistral_client import MistralClient
from .gemini_client import GeminiClient

__all__ = [
    "LLMFactory",
    "MistralClient",
    "GeminiClient",
]
