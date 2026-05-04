"""
Mistral AI API Client
=======================
Default LLM provider.  Uses Mistral's chat completion endpoint.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


class MistralClient:
    """
    Wrapper around Mistral AI API.
    Set MISTRAL_API_KEY in environment or .env file.
    """

    BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-large-latest"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY env var.")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Send messages and return assistant text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Any:
        """
        Send messages and attempt to parse the assistant reply as JSON.
        Falls back to raw string if JSON parse fails.
        """
        # Force JSON mode if available
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def summarize(self, text: str, max_words: int = 100) -> str:
        prompt = f"Summarize the following in under {max_words} words:\n\n{text}"
        return self.chat([{"role": "user", "content": prompt}], temperature=0.3)

    def generate_risk_report(self, pair_data: Dict[str, Any]) -> str:
        prompt = (
            f"Generate a short geopolitical risk report for the country pair "
            f"{pair_data.get('source')} → {pair_data.get('target')}.\n"
            f"Risk score: {pair_data.get('risk', 0):.2f}\n"
            f"Conflict events: {pair_data.get('conflict_count', 0)}\n"
            f"Cooperation events: {pair_data.get('coop_count', 0)}\n"
            f"Average tone: {pair_data.get('avg_tone', 0):.1f}\n\n"
            f"Write 2-3 concise sentences in markdown bullet format."
        )
        return self.chat([{"role": "user", "content": prompt}], temperature=0.3)
