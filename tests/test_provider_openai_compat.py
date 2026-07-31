"""Tests for the OpenAI-compatible runtime and Hub-registry provider loading.

Covers:
- OpenAICompatRuntime request/response mapping (chat + embeddings)
- model_fallback retry on model-not-found (vLLM path-style model ids)
- GenericProvider backend selection for ``openai_compatible``
- ProviderLoader.scan_hub_registry reading hub-installed providers
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from rosclaw.provider.adapters.generic import GenericProvider
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.loader import ProviderLoader
from rosclaw.provider.runtimes.openai_compat_runtime import OpenAICompatRuntime


def _mock_aiohttp(resp_payload, status=200):
    """Return a mocked aiohttp module whose session returns resp_payload."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=resp_payload)
    mock_session = MagicMock()
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_session.close = AsyncMock()
    mock_aiohttp = MagicMock()
    mock_aiohttp.ClientSession.return_value = mock_session
    mock_aiohttp.ClientTimeout = MagicMock()
    return mock_aiohttp, mock_session


class TestOpenAICompatRuntime:
    @pytest.fixture(autouse=True)
    def cleanup_modules(self):
        yield
        sys.modules.pop("aiohttp", None)

    @pytest.mark.asyncio
    async def test_chat_completions_invoke(self):
        mock_aiohttp, mock_session = _mock_aiohttp(
            {
                "model": "m",
                "choices": [
                    {"message": {"content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {"total_tokens": 5},
            }
        )
        sys.modules["aiohttp"] = mock_aiohttp
        rt = OpenAICompatRuntime("t", "http://127.0.0.1:8001/v1", model="m")
        await rt.start()
        out = await rt.invoke({"inputs": {"prompt": "hi"}, "constraints": {"max_tokens": 8}})
        assert out["result"] == "hello"
        assert out["finish_reason"] == "stop"
        sent = mock_session.post.call_args
        assert sent[0][0] == "http://127.0.0.1:8001/v1/chat/completions"
        body = sent[1]["json"]
        assert body["model"] == "m"
        assert body["messages"][-1] == {"role": "user", "content": "hi"}
        assert body["max_tokens"] == 8
        await rt.stop()

    @pytest.mark.asyncio
    async def test_embeddings_invoke(self):
        mock_aiohttp, mock_session = _mock_aiohttp(
            {"model": "m", "data": [{"embedding": [0.1, 0.2, 0.3]}], "usage": {}}
        )
        sys.modules["aiohttp"] = mock_aiohttp
        rt = OpenAICompatRuntime(
            "t", "http://127.0.0.1:8000/v1", api_kind="embeddings", model="m"
        )
        await rt.start()
        out = await rt.invoke({"inputs": {"text": "hello world"}})
        assert out["result"] == [0.1, 0.2, 0.3]
        assert out["dimension"] == 3
        sent = mock_session.post.call_args
        assert sent[0][0] == "http://127.0.0.1:8000/v1/embeddings"
        assert sent[1]["json"]["input"] == "hello world"
        await rt.stop()

    @pytest.mark.asyncio
    async def test_model_fallback_on_404(self):
        """vLLM without --served-model-name rejects the alias with 404."""
        ok_payload = {"model": "/models/X", "choices": [{"message": {"content": "ok"}}]}
        mock_aiohttp, mock_session = _mock_aiohttp(ok_payload)
        # first call 404 (alias), second call 200 (fallback path id)
        resp_404 = MagicMock()
        resp_404.status = 404
        resp_404.json = AsyncMock(return_value={"error": {"message": "no model"}})
        resp_200 = MagicMock()
        resp_200.status = 200
        resp_200.json = AsyncMock(return_value=ok_payload)
        ctx_404 = MagicMock()
        ctx_404.__aenter__ = AsyncMock(return_value=resp_404)
        ctx_404.__aexit__ = AsyncMock(return_value=False)
        ctx_200 = MagicMock()
        ctx_200.__aenter__ = AsyncMock(return_value=resp_200)
        ctx_200.__aexit__ = AsyncMock(return_value=False)
        mock_session.post.side_effect = [ctx_404, ctx_200]
        sys.modules["aiohttp"] = mock_aiohttp

        rt = OpenAICompatRuntime(
            "t",
            "http://127.0.0.1:8001/v1",
            model="alias",
            model_fallback="/models/X",
            retries=0,
        )
        await rt.start()
        out = await rt.invoke({"inputs": {"prompt": "hi"}})
        assert out["result"] == "ok"
        second_body = mock_session.post.call_args_list[1][1]["json"]
        assert second_body["model"] == "/models/X"
        await rt.stop()

    @pytest.mark.asyncio
    async def test_health_detail_expected_model(self):
        mock_aiohttp, _ = _mock_aiohttp({"data": [{"id": "/models/X"}]})
        sys.modules["aiohttp"] = mock_aiohttp
        rt = OpenAICompatRuntime(
            "t", "http://127.0.0.1:8000/v1", model="x", model_fallback="/models/X"
        )
        await rt.start()
        detail = await rt.health_detail()
        assert detail["reachable"] is True
        assert detail["expected_model_present"] is True
        await rt.stop()

    @pytest.mark.asyncio
    async def test_health_detail_unreachable(self):
        mock_aiohttp, mock_session = _mock_aiohttp({})
        mock_session.get.side_effect = ConnectionError("refused")
        sys.modules["aiohttp"] = mock_aiohttp
        rt = OpenAICompatRuntime("t", "http://127.0.0.1:9/v1")
        await rt.start()
        detail = await rt.health_detail()
        assert detail["reachable"] is False
        await rt.stop()

    def test_invalid_api_kind_rejected(self):
        from rosclaw.provider.core.errors import RuntimeAdapterError

        with pytest.raises(RuntimeAdapterError):
            OpenAICompatRuntime("t", "http://x/v1", api_kind="bogus")


class TestGenericProviderOpenAICompat:
    def test_backend_selection(self):
        manifest = ProviderManifest.from_dict(
            {
                "name": "p",
                "version": "1.0.0",
                "type": "embedding",
                "capabilities": ["embedding.text"],
                "runtime": {
                    "backend": "openai_compatible",
                    "endpoint": "http://127.0.0.1:8000/v1",
                    "env": {"api_kind": "embeddings", "model": "m"},
                },
            }
        )
        provider = GenericProvider(manifest)
        assert isinstance(provider._runtime, OpenAICompatRuntime)
        assert provider._runtime.api_kind == "embeddings"

    @pytest.mark.asyncio
    async def test_infer_roundtrip(self):
        mock_aiohttp, _ = _mock_aiohttp(
            {"model": "m", "data": [{"embedding": [1.0, 2.0]}], "usage": {}}
        )
        sys.modules["aiohttp"] = mock_aiohttp
        try:
            manifest = ProviderManifest.from_dict(
                {
                    "name": "p",
                    "version": "1.0.0",
                    "type": "embedding",
                    "capabilities": ["embedding.text"],
                    "runtime": {
                        "backend": "openai_compatible",
                        "endpoint": "http://127.0.0.1:8000/v1",
                        "env": {"api_kind": "embeddings", "model": "m"},
                    },
                }
            )
            provider = GenericProvider(manifest)
            await provider.load()
            resp = await provider.infer(
                ProviderRequest(
                    request_id="r1",
                    capability="embedding.text",
                    inputs={"text": "abc"},
                )
            )
            assert resp.status == "ok"
            assert resp.result == [1.0, 2.0]
            health = await provider.health()
            assert health["endpoint"]["reachable"] is True
            await provider.unload()
        finally:
            sys.modules.pop("aiohttp", None)


class TestScanHubRegistry:
    def _write_hub_registry(self, tmp_path, asset_dir):
        registries = tmp_path / "runtime" / "registries"
        registries.mkdir(parents=True)
        (registries / "providers.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "updated_at": "",
                    "assets": {
                        "rosclaw://provider/ns/p@1.0.0": {
                            "ref": "rosclaw://provider/ns/p@1.0.0",
                            "asset_dir": str(asset_dir),
                        }
                    },
                }
            )
        )
        return registries

    def _write_provider_yaml(self, asset_dir):
        asset_dir.mkdir(parents=True, exist_ok=True)
        (asset_dir / "provider.yaml").write_text(
            "name: hub_p\n"
            "version: '1.0.0'\n"
            "type: embedding\n"
            "capabilities: [embedding.text]\n"
            "runtime:\n"
            "  backend: openai_compatible\n"
            "  endpoint: http://127.0.0.1:8000/v1\n"
            "  env:\n"
            "    api_kind: embeddings\n"
            "    model: m\n"
        )

    def test_scan_hub_registry_loads_provider(self, tmp_path):
        asset_dir = tmp_path / "hub" / "assets" / "p"
        self._write_provider_yaml(asset_dir)
        registries = self._write_hub_registry(tmp_path, asset_dir)
        registry = ProviderRegistry()
        loader = ProviderLoader(registry)
        loaded = loader.scan_hub_registry(registries)
        assert loaded == ["hub_p"]
        assert "hub_p" in registry.list_providers()

    def test_scan_hub_registry_missing_file(self, tmp_path):
        loader = ProviderLoader(ProviderRegistry())
        assert loader.scan_hub_registry(tmp_path / "nonexistent") == []

    def test_scan_hub_registry_missing_provider_yaml(self, tmp_path):
        asset_dir = tmp_path / "hub" / "assets" / "p"
        asset_dir.mkdir(parents=True)
        registries = self._write_hub_registry(tmp_path, asset_dir)
        loader = ProviderLoader(ProviderRegistry())
        assert loader.scan_hub_registry(registries) == []
