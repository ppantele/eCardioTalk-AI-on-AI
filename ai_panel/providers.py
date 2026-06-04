"""
Unified provider wrappers for Anthropic (Claude), OpenAI (ChatGPT), and
Google (Gemini).

Each provider exposes two methods:
  qualitative(system, user) -> (text: str, usage: dict)
  likert(system, user)      -> (text: str, usage: dict)

usage dicts always contain {"input_tokens": int, "output_tokens": int}.

Pass mock=True to any provider to get deterministic fake responses with
zero API calls and zero cost — useful for end-to-end plumbing tests.
"""

import random
import re
import threading
import time
from abc import ABC, abstractmethod

import config


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Thread-safe token-bucket rate limiter (requests per minute)."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm   # minimum seconds between acquisitions
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.time()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
            self._next_allowed = time.time() + self._interval


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ProviderBase(ABC):
    name: str

    @abstractmethod
    def qualitative(self, system: str, user: str) -> tuple[str, dict]:
        ...

    @abstractmethod
    def likert(self, system: str, user: str) -> tuple[str, dict]:
        ...


# ---------------------------------------------------------------------------
# Mock provider (zero cost, used by --mock flag)
# ---------------------------------------------------------------------------

class MockProvider(ProviderBase):
    def __init__(self, name: str) -> None:
        self.name = name

    def qualitative(self, system: str, user: str) -> tuple[str, dict]:
        text = (
            f"[MOCK — {self.name}] This is a placeholder qualitative answer. "
            "In a real run this would be a thoughtful, detailed AI perspective "
            "on cardiovascular AI from the perspective of this model. "
            "The answer would typically span 3-5 paragraphs covering clinical, "
            "technical, ethical, and societal dimensions."
        )
        usage = {"input_tokens": 150, "output_tokens": 80}
        return text, usage

    def likert(self, system: str, user: str) -> tuple[str, dict]:
        # Simulate a realistic distribution centred around 7
        value = max(0, min(10, int(random.gauss(7.0, 1.8))))
        usage = {"input_tokens": 100, "output_tokens": 2}
        return str(value), usage


# ---------------------------------------------------------------------------
# Anthropic — Claude
# ---------------------------------------------------------------------------

class ClaudeProvider(ProviderBase):
    name = "claude"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        provider_name: str = "claude",
        mock: bool = False,
    ) -> None:
        if mock:
            raise ValueError("Use MockProvider for mock mode.")
        import anthropic  # noqa: PLC0415
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or config.MODELS["claude"]
        self.name = provider_name  # instance attribute (e.g. "claude_likert")

        # Opus 4.7+ deprecated the temperature parameter; Sonnet and below support it.
        self._use_temperature = "opus" not in self._model

        # Opus tier: 50 RPM hard limit → stay at 40. Sonnet tier: much higher.
        rpm = 40 if "opus" in self._model else 80
        self._rate_limiter = _RateLimiter(rpm=rpm)

    def _call(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int,
        temperature: float,
        retries: int = config.MAX_RETRIES,
    ) -> tuple[str, dict]:
        import anthropic  # noqa: PLC0415

        system_blocks = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        for attempt in range(retries):
            try:
                self._rate_limiter.acquire()
                create_kwargs: dict = dict(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system_blocks,
                    messages=[{"role": "user", "content": user_text}],
                )
                if self._use_temperature:
                    create_kwargs["temperature"] = temperature
                resp = self._client.messages.create(**create_kwargs)
                text = resp.content[0].text
                usage = {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                }
                return text, usage
            except anthropic.RateLimitError:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500 and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError("ClaudeProvider: max retries exceeded")  # unreachable

    def qualitative(self, system: str, user: str) -> tuple[str, dict]:
        return self._call(
            system, user,
            max_tokens=config.QUALITATIVE_MAX_TOKENS,
            temperature=config.QUALITATIVE_TEMPERATURE,
        )

    def likert(self, system: str, user: str) -> tuple[str, dict]:
        temp = config.LIKERT_TEMPERATURES.get(self.name, config.LIKERT_TEMPERATURE)
        return self._call(
            system, user,
            max_tokens=config.LIKERT_MAX_TOKENS,
            temperature=temp,
        )


# ---------------------------------------------------------------------------
# OpenAI — ChatGPT
# ---------------------------------------------------------------------------

class OpenAIProvider(ProviderBase):
    name = "openai"

    def __init__(self, api_key: str, mock: bool = False) -> None:
        if mock:
            raise ValueError("Use MockProvider for mock mode.")
        from openai import OpenAI  # noqa: PLC0415
        self._client = OpenAI(api_key=api_key)
        self._model = config.MODELS["openai"]
        self._rate_limiter = _RateLimiter(rpm=200)

    def _call(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int,
        temperature: float,
        retries: int = config.MAX_RETRIES,
    ) -> tuple[str, dict]:
        import openai  # noqa: PLC0415

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

        # gpt-5.5 and O-series reasoning models use max_completion_tokens,
        # not max_tokens, and may not support temperature. We start with both
        # and drop temperature automatically on the first rejection.
        use_temperature = True

        for attempt in range(retries):
            try:
                self._rate_limiter.acquire()
                kwargs: dict = {
                    "model": self._model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
                if use_temperature:
                    kwargs["temperature"] = temperature
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = {
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                }
                return text, usage
            except openai.BadRequestError as exc:
                if "temperature" in str(exc).lower() and use_temperature:
                    use_temperature = False  # retry without temperature
                    continue
                raise
            except openai.RateLimitError:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except openai.APIStatusError as exc:
                if exc.status_code >= 500 and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError("OpenAIProvider: max retries exceeded")

    def qualitative(self, system: str, user: str) -> tuple[str, dict]:
        return self._call(
            system, user,
            max_tokens=config.QUALITATIVE_MAX_TOKENS,
            temperature=config.QUALITATIVE_TEMPERATURE,
        )

    def likert(self, system: str, user: str) -> tuple[str, dict]:
        return self._call(
            system, user,
            max_tokens=config.LIKERT_MAX_TOKENS,
            temperature=config.LIKERT_TEMPERATURES.get("openai", config.LIKERT_TEMPERATURE),
        )


# ---------------------------------------------------------------------------
# Google — Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(ProviderBase):
    name = "gemini"

    def __init__(self, api_key: str, mock: bool = False) -> None:
        if mock:
            raise ValueError("Use MockProvider for mock mode.")
        from google import genai  # noqa: PLC0415
        self._client = genai.Client(api_key=api_key)
        self._model = config.MODELS["gemini"]
        self._rate_limiter = _RateLimiter(rpm=500)

    def _call(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int,
        temperature: float,
        retries: int = config.MAX_RETRIES,
        disable_thinking: bool = False,
    ) -> tuple[str, dict]:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        # gemini-3.5-flash is a thinking model; temperature=1.0 triggers full
        # thinking mode which returns resp.text=None for Likert calls.
        # Disabling thinking forces a direct, fast response.
        thinking_cfg = types.ThinkingConfig(thinking_budget=0) if disable_thinking else None

        cfg = types.GenerateContentConfig(
            system_instruction=system_text,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=thinking_cfg,
        )

        for attempt in range(retries):
            try:
                self._rate_limiter.acquire()
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=user_text,
                    config=cfg,
                )
                text = resp.text or ""
                usage_meta = resp.usage_metadata
                usage = {
                    "input_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
                }
                return text, usage
            except Exception as exc:  # Google SDK raises various error types
                is_rate_limit = "429" in str(exc) or "quota" in str(exc).lower()
                is_server_err = "500" in str(exc) or "503" in str(exc)
                if (is_rate_limit or is_server_err) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError("GeminiProvider: max retries exceeded")

    def qualitative(self, system: str, user: str) -> tuple[str, dict]:
        return self._call(
            system, user,
            max_tokens=config.QUALITATIVE_MAX_TOKENS,
            temperature=config.QUALITATIVE_TEMPERATURE,
            disable_thinking=True,   # thinking consumes the token budget before visible text
        )

    def likert(self, system: str, user: str) -> tuple[str, dict]:
        return self._call(
            system, user,
            max_tokens=config.LIKERT_MAX_TOKENS,
            temperature=config.LIKERT_TEMPERATURES.get("gemini", config.LIKERT_TEMPERATURE),
            disable_thinking=True,   # prevents thinking-mode null responses
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_providers(keys: dict[str, str], mock: bool = False) -> dict[str, ProviderBase]:
    """
    Return a dict of initialised providers keyed by name.

    Keys returned: "claude" (qualitative), "claude_likert" (Sonnet, for Likert),
    "openai", "gemini".

    If mock=True every provider is replaced with a MockProvider at zero cost.
    """
    if mock:
        return {
            "claude":        MockProvider("claude"),
            "claude_likert": MockProvider("claude_likert"),
            "openai":        MockProvider("openai"),
            "gemini":        MockProvider("gemini"),
        }
    return {
        "claude": ClaudeProvider(
            api_key=keys["claude"],
            model=config.MODELS["claude"],
            provider_name="claude",
        ),
        "claude_likert": ClaudeProvider(
            api_key=keys["claude"],
            model=config.MODELS["claude_likert"],
            provider_name="claude_likert",
        ),
        "openai": OpenAIProvider(api_key=keys["openai"]),
        "gemini": GeminiProvider(api_key=keys["gemini"]),
    }
