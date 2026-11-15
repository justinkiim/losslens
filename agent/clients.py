"""Model clients: live Anthropic API, a retry wrapper (exponential backoff on transient errors,
token accounting), and a scripted client for deterministic offline tests."""
import json
import os
import time
import urllib.error
import urllib.request

RETRYABLE_HTTP = (429, 500, 502, 503, 529)


class AnthropicClient(object):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model="claude-sonnet-5-5", api_key=None, max_tokens=2048):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        if not self.api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to use the live client")

    def create(self, system, messages, tools):
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "system": system, "messages": messages, "tools": tools}
        req = urllib.request.Request(
            self.URL, data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))


class RetryingClient(object):
    """Wraps any client: retries transient failures with exponential backoff and tracks token usage."""

    def __init__(self, inner, max_retries=4, base_delay=1.0, sleep=time.sleep):
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.sleep = sleep
        self.retries = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def create(self, system, messages, tools):
        for attempt in range(self.max_retries + 1):
            try:
                reply = self.inner.create(system, messages, tools)
                u = reply.get("usage") or {}
                self.usage["input_tokens"] += u.get("input_tokens", 0)
                self.usage["output_tokens"] += u.get("output_tokens", 0)
                return reply
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_HTTP or attempt == self.max_retries:
                    raise
            except urllib.error.URLError:
                if attempt == self.max_retries:
                    raise
            self.retries += 1
            self.sleep(self.base_delay * (2 ** attempt))


class ScriptedClient(object):
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def create(self, system, messages, tools):
        self.calls += 1
        return self.replies.pop(0)
