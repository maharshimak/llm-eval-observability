from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from urllib import request

from llm_eval.models import EvalCase, ModelOutput

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:/-]{1,200})\]")


@dataclass(slots=True)
class OpenAICompatibleCandidate:
    """Synchronous candidate adapter suitable for the existing ExperimentRunner."""

    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 60.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("base_url and model are required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def __call__(self, case: EvalCase) -> ModelOutput:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": case.prompt}],
                "temperature": self.temperature,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        req = request.Request(endpoint, data=payload, headers=headers, method="POST")
        started = perf_counter()
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode())
        latency_ms = (perf_counter() - started) * 1000

        text = str(data["choices"][0]["message"]["content"]).strip()
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        usage_reported = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
        if not usage_reported:
            prompt_tokens = len(case.prompt.split())
            completion_tokens = len(text.split())

        citations = list(dict.fromkeys(_CITATION_RE.findall(text)))
        return ModelOutput(
            text=text,
            citations=citations,
            latency_ms=latency_ms,
            input_tokens=int(prompt_tokens),
            output_tokens=int(completion_tokens),
            provider="openai-compatible",
            model=self.model,
            usage_source="reported" if usage_reported else "estimated_whitespace_tokens",
        )
