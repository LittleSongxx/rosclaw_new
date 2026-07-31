"""OpenAI-compatible HTTP runtime adapter.

Supports any server exposing the OpenAI API surface (vLLM, SGLang, TGI,
llama.cpp, LM Studio, ...). Selected with ``runtime.backend:
openai_compatible`` in provider.yaml; ``runtime.endpoint`` is the API base
URL (e.g. ``http://127.0.0.1:8000/v1``).

Manifest env hints (all optional):

- ``api_kind``: ``chat_completions`` (default) or ``embeddings``
- ``model``: served model name sent in requests; ``model_fallback`` is used
  for health matching when the server reports a path-like model id
- ``health_endpoint``: override for the health probe (default
  ``{endpoint}/models``)
- ``timeout_sec``, ``retries``, ``headers`` (same semantics as HTTPRuntime)
"""

from __future__ import annotations

import asyncio
from typing import Any

from rosclaw.provider.core.errors import RuntimeAdapterError
from rosclaw.provider.runtimes.base import RuntimeAdapter

_API_CHAT = "chat_completions"
_API_EMBEDDINGS = "embeddings"


class OpenAICompatRuntime(RuntimeAdapter):
    """Runtime adapter for OpenAI-compatible inference servers."""

    def __init__(
        self,
        name: str,
        endpoint: str,
        api_kind: str = _API_CHAT,
        model: str = "",
        model_fallback: str = "",
        health_endpoint: str = "",
        timeout_sec: float = 30.0,
        retries: int = 1,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(name, config={"endpoint": endpoint, "timeout": timeout_sec})
        if api_kind not in (_API_CHAT, _API_EMBEDDINGS):
            raise RuntimeAdapterError(
                f"Unsupported api_kind for openai_compatible backend: {api_kind!r}",
                provider=name,
            )
        self.endpoint = endpoint.rstrip("/")
        self.api_kind = api_kind
        self.model = model
        self.model_fallback = model_fallback
        self.health_endpoint = health_endpoint or f"{self.endpoint}/models"
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.headers = headers or {}
        self._session = None

    async def start(self) -> None:
        try:
            import aiohttp
        except ImportError as err:
            raise RuntimeError(
                "aiohttp is required for OpenAICompatRuntime. pip install aiohttp"
            ) from err
        self._session = aiohttp.ClientSession(headers=self.headers)
        self._started = True

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._started = False

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------
    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_started()
        if self.api_kind == _API_EMBEDDINGS:
            path, body = self._build_embeddings_request(payload)
        else:
            path, body = self._build_chat_request(payload)
        raw = await self._post(path, body)
        if self.api_kind == _API_EMBEDDINGS:
            return self._parse_embeddings_response(raw)
        return self._parse_chat_response(raw)

    def _build_chat_request(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        inputs = payload.get("inputs") or {}
        constraints = payload.get("constraints") or {}
        messages = inputs.get("messages")
        if not messages:
            messages = []
            system = inputs.get("system")
            if system:
                messages.append({"role": "system", "content": system})
            prompt = inputs.get("prompt") or inputs.get("text") or ""
            messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"messages": messages}
        if self.model:
            body["model"] = self.model
        for key in ("max_tokens", "temperature", "top_p", "stop"):
            if key in inputs:
                body[key] = inputs[key]
            elif key in constraints:
                body[key] = constraints[key]
        return "/chat/completions", body

    def _build_embeddings_request(
        self, payload: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        inputs = payload.get("inputs") or {}
        text_input = (
            inputs.get("input") or inputs.get("texts") or inputs.get("text") or ""
        )
        body: dict[str, Any] = {"input": text_input}
        if self.model:
            body["model"] = self.model
        if "dimensions" in inputs:
            body["dimensions"] = inputs["dimensions"]
        return "/embeddings", body

    @staticmethod
    def _parse_chat_response(raw: dict[str, Any]) -> dict[str, Any]:
        choices = raw.get("choices") or []
        content = ""
        finish_reason = None
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            finish_reason = choices[0].get("finish_reason")
        return {
            "result": content,
            "model": raw.get("model", ""),
            "finish_reason": finish_reason,
            "usage": raw.get("usage") or {},
        }

    @staticmethod
    def _parse_embeddings_response(raw: dict[str, Any]) -> dict[str, Any]:
        data = raw.get("data") or []
        vectors = [item.get("embedding") for item in data]
        vectors = [v for v in vectors if v is not None]
        dimension = len(vectors[0]) if vectors else 0
        result: Any = vectors[0] if len(vectors) == 1 else vectors
        return {
            "result": result,
            "dimension": dimension,
            "count": len(vectors),
            "model": raw.get("model", ""),
            "usage": raw.get("usage") or {},
        }

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeAdapterError("Session not initialized", provider=self.name)

        # Some servers (vLLM without --served-model-name) only accept a
        # path-like model id; retry with model_fallback on model-not-found.
        model_chain = [m for m in (self.model, self.model_fallback) if m]
        if not self.model_fallback:
            model_chain = model_chain[:1]

        last_error: Exception | None = None
        for mi, model in enumerate(model_chain or [""]):
            if model:
                body = {**body, "model": model}
            try:
                return await self._post_with_retries(path, body)
            except RuntimeAdapterError as e:
                last_error = e
                if mi + 1 < len(model_chain) and "404" in str(e):
                    continue
                raise

        raise RuntimeAdapterError(
            f"OpenAI-compatible invoke failed: {last_error}",
            provider=self.name,
        )

    async def _post_with_retries(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        url = f"{self.endpoint}{path}"
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                async with self._session.post(
                    url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
                ) as resp:
                    resp_body = await resp.json()
                    if resp.status >= 400:
                        raise RuntimeAdapterError(
                            f"HTTP {resp.status}: {resp_body}",
                            provider=self.name,
                        )
                    return resp_body
            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise RuntimeAdapterError(
            f"OpenAI-compatible invoke failed after {self.retries + 1} attempts: "
            f"{last_error}",
            provider=self.name,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def health_detail(self) -> dict[str, Any]:
        """Probe ``GET {endpoint}/models`` and check the expected model id.

        Returns a dict with ``reachable`` and, when the manifest declared a
        model, ``expected_model_present``. Never raises; errors are reported
        in the returned dict.
        """
        if self._session is None:
            return {"reachable": False, "error": "runtime not started"}

        import aiohttp

        try:
            async with self._session.get(
                self.health_endpoint,
                timeout=aiohttp.ClientTimeout(total=min(self.timeout_sec, 10.0)),
            ) as resp:
                if resp.status >= 400:
                    return {"reachable": False, "error": f"HTTP {resp.status}"}
                body = await resp.json()
        except Exception as e:  # noqa: BLE001 - health must not raise
            return {"reachable": False, "error": str(e)}

        served = [m.get("id", "") for m in (body.get("data") or [])]
        detail: dict[str, Any] = {"reachable": True, "served_models": served}
        expected = self.model or self.model_fallback
        if expected:
            candidates = {self.model, self.model_fallback} - {""}
            detail["expected_model"] = expected
            detail["expected_model_present"] = any(
                any(c and (c == s or c in s) for c in candidates) for s in served
            )
        return detail
