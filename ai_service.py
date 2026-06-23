"""Small, testable OpenRouter transport used by ZubeAnalystOS."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat_completion(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int | tuple[int, int] = (10, 55),
    hard_timeout: int = 70,
    http_client: Any = requests,
) -> str:
    """Return assistant text with both network-phase and wall-clock deadlines."""
    if not api_key:
        raise RuntimeError("An OpenRouter API key is required.")
    def make_request():
        response = http_client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "ZubeAnalystOS",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openrouter")
    future = executor.submit(make_request)
    try:
        response = future.result(timeout=hard_timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        raise requests.Timeout(
            f"OpenRouter did not finish within {hard_timeout} seconds. Please try again."
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenRouter returned an unexpected response format.") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter returned an empty response.")
    return content.strip()
