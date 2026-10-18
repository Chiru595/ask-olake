"""OpenRouter chat client: model fallbacks, retries, concurrency cap, streaming and a credit-based cost guard."""
import asyncio
import time
from collections.abc import AsyncIterator

import httpx
import structlog
from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.config import Settings

log = structlog.get_logger()


class LLMUnavailable(Exception):
    pass


class LLMClient:
    def __init__(self, s: Settings, http: httpx.AsyncClient):
        self.s, self.http = s, http
        self.client = AsyncOpenAI(base_url=s.openrouter_base_url, api_key=s.openrouter_api_key or "missing",
                                  timeout=s.llm_timeout_seconds, max_retries=0)
        self.sem = asyncio.Semaphore(s.llm_max_concurrency)
        self.credit_remaining: float | None = None
        self.credit_checked = 0.0
        self.calls = 0
        self.cost_usd = 0.0

    async def refresh_credit(self, force: bool = False) -> float | None:
        if not force and time.monotonic() - self.credit_checked < 60:
            return self.credit_remaining
        try:
            r = await self.http.get(f"{self.s.openrouter_base_url}/key", timeout=5,
                                    headers={"Authorization": f"Bearer {self.s.openrouter_api_key}"})
            r.raise_for_status()
            rem = r.json()["data"].get("limit_remaining")
            self.credit_remaining = float(rem) if rem is not None else None
        except Exception as e:  # noqa: BLE001
            log.warning("credit_check_failed", error=str(e)[:200])
        self.credit_checked = time.monotonic()
        return self.credit_remaining

    def _models(self) -> list[str]:
        rem = self.credit_remaining
        if rem is not None and rem < self.s.llm_budget_switch_usd:
            return [self.s.llm_cheap_model]
        return [self.s.llm_model] + [m for m in self.s.fallback_models if m != self.s.llm_model]

    def budget_exhausted(self) -> bool:
        return self.credit_remaining is not None and self.credit_remaining < self.s.llm_budget_stop_usd

    def _body(self, messages: list[dict], stream: bool) -> dict:
        models = self._models()
        extra = {"reasoning": {"effort": self.s.llm_reasoning_effort}, "usage": {"include": True}}
        if len(models) > 1:
            extra["models"] = models[1:]
        return dict(model=models[0], messages=messages, temperature=self.s.llm_temperature,
                    max_tokens=self.s.llm_max_tokens, stream=stream, extra_body=extra)

    def _track(self, usage) -> None:
        self.calls += 1
        cost = getattr(usage, "cost", None) if usage else None
        if cost is None and usage is not None and isinstance(getattr(usage, "model_extra", None), dict):
            cost = usage.model_extra.get("cost")
        if cost:
            self.cost_usd += float(cost)

    async def complete(self, messages: list[dict]) -> tuple[str, str]:
        """Returns (text, model_used)."""
        await self.refresh_credit()
        if self.budget_exhausted():
            raise LLMUnavailable("budget exhausted")
        last: Exception | None = None
        async with self.sem:
            for attempt in range(3):
                try:
                    resp = await self.client.chat.completions.create(**self._body(messages, False))
                    self._track(resp.usage)
                    return (resp.choices[0].message.content or "").strip(), resp.model
                except (APITimeoutError, RateLimitError, APIError, httpx.HTTPError) as e:
                    last = e
                    status = getattr(e, "status_code", None)
                    if status in (400, 401, 402, 403):
                        break
                    await asyncio.sleep(0.5 * 2**attempt)
        log.error("llm_failed", error=str(last)[:300])
        raise LLMUnavailable(str(last))

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        await self.refresh_credit()
        if self.budget_exhausted():
            raise LLMUnavailable("budget exhausted")
        async with self.sem:
            try:
                stream = await self.client.chat.completions.create(**self._body(messages, True))
                usage = None
                async for ev in stream:
                    if getattr(ev, "usage", None):
                        usage = ev.usage
                    if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                        yield ev.choices[0].delta.content
                self._track(usage)
            except (APITimeoutError, RateLimitError, APIError, httpx.HTTPError) as e:
                log.error("llm_stream_failed", error=str(e)[:300])
                raise LLMUnavailable(str(e)) from e
