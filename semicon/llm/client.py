"""Provider-agnostic LLM client with response caching and a hard cost ceiling.

Three things this client guarantees, all of which matter more than speed:

1. **Repeat runs are free.** Responses are cached on disk keyed by a hash of the
   model, prompt and parameters. Re-running the pipeline never re-bills.
2. **The run cannot silently overspend.** Every call estimates its cost and adds
   it to a running total. Crossing ``llm.run_token_budget`` aborts with an
   explicit error rather than quietly continuing.
3. **The provider is swappable.** Only :meth:`_call_provider` knows about OpenAI.
   The pipeline depends on ``complete_json`` / ``complete_text`` and nothing else.

Only the OpenAI implementation is provided, because that is what this project
uses. A second provider is a small addition, not a rewrite.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings, get_api_key


class LlmError(RuntimeError):
    """Raised when a model call fails in a way the caller must handle."""


class BudgetExceeded(LlmError):
    """Raised before a call that would exceed the configured token budget."""


@dataclass
class Usage:
    """Token accounting for one run."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, model: str, inp: int, out: int, cost: float) -> None:
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += cost
        bucket = self.per_model.setdefault(
            model, {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
        )
        bucket["calls"] += 1
        bucket["input"] += inp
        bucket["output"] += out
        bucket["cost"] += cost

    def summary(self) -> str:
        if self.calls == 0 and self.cache_hits == 0:
            return "no LLM calls"
        return (
            f"{self.calls} calls, {self.cache_hits} cache hits, "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens, "
            f"${self.cost_usd:.4f}"
        )


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate (~4 chars/token, rounded up)."""
    return max(1, len(text) // 4 + 1)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Models sometimes wrap JSON in prose or a markdown fence despite instructions.
    We attempt recovery, but we never regex-scrape a partial object into
    existence - if the payload is not valid JSON after fence-stripping and
    brace-trimming, the caller gets an error.
    """
    candidate = text.strip()
    fence = _JSON_FENCE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    if not candidate:
        raise LlmError("Model returned an empty response")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Trim to the outermost object or array and retry once.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    snippet = candidate[:200].replace("\n", " ")
    raise LlmError(f"Model response was not valid JSON. Starts with: {snippet!r}")


class LlmClient:
    """Cached, budget-aware, provider-agnostic JSON/text completion client."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str | None = None,
        cache_root: Path | None = None,
        live: bool = True,
    ):
        self.settings = settings
        # get_api_key() falls back to reading .env when the environment is empty,
        # so a key pasted into .env works without exporting anything.
        self.api_key = api_key or get_api_key()
        self.usage = Usage()
        self.live = live
        self._cache_dir = (cache_root or settings.cache_root / "llm")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(timeout=120.0)

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LlmClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # -- caching ------------------------------------------------------------ #

    def _cache_key(
        self, model: str, system: str, user: str, json_mode: bool, temperature: float
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "system": system,
                "user": user,
                "json_mode": json_mode,
                "temperature": temperature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _cache_get(self, key: str) -> str | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.usage.cache_hits += 1
            return str(data["response"])
        except (json.JSONDecodeError, KeyError):
            return None

    def _cache_put(self, key: str, model: str, response: str, usage: dict) -> None:
        path = self._cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "response": response,
                    "usage": usage,
                    "cached_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- cost --------------------------------------------------------------- #

    def _cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        llm = self.settings.llm
        price_in = llm.price_per_mtok_input.get(model, 1.0)
        price_out = llm.price_per_mtok_output.get(model, 3.0)
        return (
            input_tokens / 1_000_000 * price_in
            + output_tokens / 1_000_000 * price_out
        )

    def _guard_budget(
        self, estimated_input: int, max_output: int, expected_output: int | None = None
    ) -> None:
        """Refuse a call that would plausibly push the run over its token budget.

        The subtlety here is which output figure to budget with. ``max_tokens`` is
        a hard cap on one call, not an expectation - reserving the full cap per
        call made the guard fire after 11 of 18 chunks on a run whose real usage
        is a fraction of that, aborting a perfectly affordable run halfway and
        wasting every call already made.

        So the projection uses ``expected_output`` (a realistic completion size)
        while ``max_tokens`` continues to cap any individual call. The guard
        therefore estimates true spend, and the cap still prevents a runaway
        single response.
        """
        projected = (
            self.usage.total_tokens
            + estimated_input
            + (expected_output if expected_output is not None else max_output)
        )
        limit = self.settings.llm.run_token_budget
        if projected > limit:
            raise BudgetExceeded(
                f"Refusing to call the model: this call projects the run at "
                f"~{projected:,} tokens, above the configured budget of {limit:,}. "
                f"Raise llm.run_token_budget in config/settings.yaml, or reduce "
                f"the extraction context with --preview to see what is being sent."
            )

    # -- provider ----------------------------------------------------------- #

    def _call_provider(
        self,
        model: str,
        system: str,
        user: str,
        *,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict]:
        """The only provider-specific method in the project.

        A second provider means implementing this method and nothing else.
        """
        if not self.api_key:
            raise LlmError(
                "No OpenAI API key found. Set the OPENAI_API_KEY environment "
                "variable (or create a .env file), then re-run. Steps 1-2 work "
                "without a key; steps 3-7 require one."
            )

        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = "unknown error"
        for attempt in range(3):
            try:
                resp = self._client.post(self.API_URL, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
                continue

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LlmError(f"Model returned no choices: {data}")
                return choices[0]["message"]["content"], data.get("usage") or {}

            if resp.status_code == 401:
                raise LlmError(
                    "OpenAI rejected the API key (HTTP 401). Check that "
                    "OPENAI_API_KEY is valid and has credit."
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(2**attempt)
                continue

            raise LlmError(f"HTTP {resp.status_code}: {resp.text[:400]}")

        raise LlmError(f"Model call failed after retries: {last_error}")

    # -- public API --------------------------------------------------------- #

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> str:
        """Run one completion, using the cache when possible."""
        llm = self.settings.llm
        temperature = llm.temperature if temperature is None else temperature
        max_tokens = max_tokens or llm.max_output_tokens

        key = self._cache_key(model, system, user, json_mode, temperature)
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        self._guard_budget(
            _estimate_tokens(system) + _estimate_tokens(user),
            max_tokens,
            expected_output=llm.budget_output_estimate,
        )

        response, usage = self._call_provider(
            model,
            system,
            user,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        if not input_tokens:
            input_tokens = _estimate_tokens(system) + _estimate_tokens(user)
        if not output_tokens:
            output_tokens = _estimate_tokens(response)

        self.usage.record(
            model, input_tokens, output_tokens, self._cost(model, input_tokens, output_tokens)
        )

        self._cache_put(key, model, response, usage)
        return response

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> Any:
        """Run a completion and parse it as JSON, with one repair attempt.

        The repair attempt is deliberately narrow: we re-ask the model to emit
        valid JSON, rather than trying to heuristically salvage a malformed
        payload. Salvage would risk silently accepting truncated data.
        """
        raw = self.complete(
            model=model,
            system=system,
            user=user,
            json_mode=True,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )
        try:
            return extract_json(raw)
        except LlmError as first_error:
            repair_user = (
                f"{user}\n\n---\nYour previous response was not valid JSON and "
                f"could not be parsed. Reply with ONLY a valid JSON object, no "
                f"prose and no markdown fences."
            )
            raw2 = self.complete(
                model=model,
                system=system,
                user=repair_user,
                json_mode=True,
                max_tokens=max_tokens,
                use_cache=False,
            )
            try:
                return extract_json(raw2)
            except LlmError:
                raise LlmError(
                    f"Model failed to produce valid JSON twice. First error: "
                    f"{first_error}"
                ) from first_error
