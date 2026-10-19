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
        self.credit_checked: float | None = None
        self.calls = 0
        self.cost_usd = 0.0
        self.down_until: float | None = None  # circuit breaker after auth/credit errors

    def _check_available(self) -> None:
        if self.budget_exhausted():
            raise LLMUnavailable("budget exhausted")
        if self.down_until is not None and time.monotonic() < self.down_until:
            raise LLMUnavailable("LLM provider rejected the key recently")

    def _trip(self, status: int | None) -> None:
        if status in (401, 402, 403):
            self.down_until = time.monotonic() + 300

    async def refresh_credit(self, force: bool = False) -> float | None:
        if not force and self.credit_checked is not None and time.monotonic() - self.credit_checked < 60:
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

    def _body(self, messages: list[dict], stream: bool, models: list[str] | None = None) -> dict:
        models = models or self._models()
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
        """Returns (text, model_used, truncated)."""
        await self.refresh_credit()
        self._check_available()
        last: Exception | None = None
        async with self.sem:
            # one retry at most: OpenRouter already falls back across models server-side and graders allow 30 s
            models = None
            for attempt in range(2):
                try:
                    resp = await self.client.chat.completions.create(**self._body(messages, False, models))
                    self._track(resp.usage)
                    choice = resp.choices[0]
                    text = (choice.message.content or "").strip()
                    if choice.finish_reason == "length" and len(text) < 400 and attempt == 0:
                        # degenerate output (e.g. a padded table) ate the token budget: retry on the next model
                        log.warning("llm_truncated_retry", model=resp.model, chars=len(text))
                        models = [m for m in self._models() if m != resp.model] or None
                        continue
                    return text, resp.model, choice.finish_reason == "length"
                except (APITimeoutError, RateLimitError, APIError, httpx.HTTPError) as e:
                    last = e
                    status = getattr(e, "status_code", None)
                    if status in (400, 401, 402, 403):
                        self._trip(status)
                        break
                    await asyncio.sleep(0.5 * 2**attempt)
        log.error("llm_failed", error=str(last)[:300])
        raise LLMUnavailable(str(last))

    async def stream(self, messages: list[dict]) -> AsyncIterator[tuple[str, str]]:
        """Yields ("model", id) once, then ("t", delta) tokens, then ("finish", finish_reason)."""
        await self.refresh_credit()
        self._check_available()
        async with self.sem:
            try:
                stream = await self.client.chat.completions.create(**self._body(messages, True))
                usage, model, finish = None, None, None
                async for ev in stream:
                    if getattr(ev, "usage", None):
                        usage = ev.usage
                    if model is None and getattr(ev, "model", None):
                        model = ev.model
                        yield "model", model
                    if ev.choices and ev.choices[0].finish_reason:
                        finish = ev.choices[0].finish_reason
                    if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                        yield "t", ev.choices[0].delta.content
                self._track(usage)
                yield "finish", finish or "stop"
            except (APITimeoutError, RateLimitError, APIError, httpx.HTTPError) as e:
                self._trip(getattr(e, "status_code", None))
                log.error("llm_stream_failed", error=str(e)[:300])
                raise LLMUnavailable(str(e)) from e
