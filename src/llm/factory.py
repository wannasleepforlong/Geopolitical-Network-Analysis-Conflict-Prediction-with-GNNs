"""
LLM Client Factory
==================
Load API keys from .env and return the appropriate client.
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from .mistral_client import MistralClient
from .gemini_client import GeminiClient


class LLMFactory:
    """
    Usage:
        llm = LLMFactory.create()          # default = Mistral
        llm = LLMFactory.create("gemini")  # explicit
    """

    _loaded = False

    @classmethod
    def _ensure_env(cls):
        if not cls._loaded:
            load_dotenv()
            cls._loaded = True

    @classmethod
    def create(cls, provider: Optional[str] = None, **kwargs):
        """
        Parameters
        ----------
        provider : "mistral" | "gemini" | None
            None defaults to MISTRAL if key present, else GEMINI.
        """
        cls._ensure_env()
        provider = (provider or os.getenv("LLM_PROVIDER", "mistral")).lower()

        if provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment or .env")
            return MistralClient(api_key=api_key, **kwargs)

        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment or .env")
            return GeminiClient(api_key=api_key, **kwargs)

        raise ValueError(f"Unknown LLM provider: {provider}")

    @classmethod
    def available(cls) -> list[str]:
        """Return list of providers with keys present."""
        cls._ensure_env()
        avail = []
        if os.getenv("MISTRAL_API_KEY"):
            avail.append("mistral")
        if os.getenv("GEMINI_API_KEY"):
            avail.append("gemini")
        return avail
