"""OpenAI-compatible LLM client with retry logic."""

import time
import logging
from openai import OpenAI
from src.config import APIConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around OpenAI-compatible API."""

    def __init__(self, config: APIConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = None, max_tokens: int = None) -> str:
        """Send a chat completion request with retry logic."""
        # Defensive: ensure prompts are strings
        if not isinstance(system_prompt, str):
            system_prompt = str(system_prompt)
        if not isinstance(user_prompt, str):
            user_prompt = str(user_prompt)

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                    max_tokens=tokens,
                )
                self.total_calls += 1
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens

                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts: {e}")

    def get_usage_stats(self) -> dict:
        """Return token usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
