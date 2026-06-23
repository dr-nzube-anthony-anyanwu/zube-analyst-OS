import os
import time

import pytest
import requests
from dotenv import load_dotenv

from ai_service import chat_completion


class FakeResponse:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.call = None

    def post(self, *args, **kwargs):
        self.call = (args, kwargs)
        if self.error:
            raise self.error
        return self.response


def invoke(client, api_key="test-key"):
    return chat_completion(
        api_key=api_key,
        model="test/model",
        system_prompt="system",
        user_prompt="question",
        max_tokens=100,
        temperature=.1,
        http_client=client,
    )


def test_successful_chat_request_and_headers():
    client = FakeClient(FakeResponse({"choices": [{"message": {"content": "  Useful answer  "}}]}))
    assert invoke(client) == "Useful answer"
    _, kwargs = client.call
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "test/model"
    assert kwargs["timeout"] == (10, 55)


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": " "}}]}])
def test_malformed_or_empty_responses(payload):
    with pytest.raises(RuntimeError):
        invoke(FakeClient(FakeResponse(payload)))


def test_missing_key_http_error_and_timeout():
    with pytest.raises(RuntimeError):
        invoke(FakeClient(), api_key="")
    with pytest.raises(requests.HTTPError):
        invoke(FakeClient(FakeResponse({}, requests.HTTPError("bad status"))))
    with pytest.raises(requests.Timeout):
        invoke(FakeClient(error=requests.Timeout("slow")))


def test_hard_wall_clock_timeout_returns_control():
    class SlowClient:
        @staticmethod
        def post(*args, **kwargs):
            time.sleep(.25)
            return FakeResponse({"choices": [{"message": {"content": "late"}}]})

    started = time.monotonic()
    with pytest.raises(requests.Timeout, match="did not finish"):
        chat_completion(
            api_key="test-key",
            model="test/model",
            system_prompt="system",
            user_prompt="question",
            max_tokens=10,
            temperature=0,
            hard_timeout=.05,
            http_client=SlowClient,
        )
    assert time.monotonic() - started < .2


@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE_AI_TEST") != "1", reason="set RUN_LIVE_AI_TEST=1 to contact OpenRouter")
def test_optional_live_openrouter_smoke():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not configured")
    answer = chat_completion(
        api_key=api_key,
        model=model,
        system_prompt="Answer only from the supplied synthetic data.",
        user_prompt="Synthetic rows: region,revenue; West,10; East,20. Which region is larger?",
        max_tokens=80,
        temperature=0,
    )
    assert answer
