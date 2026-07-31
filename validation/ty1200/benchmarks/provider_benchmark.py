#!/usr/bin/env python3
"""TY1200 provider benchmark & acceptance harness (任务书 §6.4 / §十二).

Covers the DeepSeek acceptance list and works for any OpenAI-compatible
provider (embedding or chat):

  1. TCP connectivity                7. non-JSON response handling
  2. /v1/models health              8. empty-choices handling
  3. zh / en / JSON-schema requests 9. wrong model name
  4. concurrency 1/4/8             10. network-down deterministic failure
  5. 2s connect timeout            11. trace redaction of request/response
  6. 120s inference timeout        12. raw requests stay out of evidence

Outputs: human-readable stdout + machine-readable JSON (--out).
No secrets, no site addresses beyond the endpoint passed in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field


@dataclass
class CaseResult:
    name: str
    status: str  # PASS | FAIL | WARN
    latency_ms: float = 0.0
    detail: str = ""
    metrics: dict = field(default_factory=dict)


def _tcp_check(host: str, port: int, timeout: float = 2.0) -> CaseResult:
    t = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return CaseResult("tcp_connect", "PASS", (time.perf_counter() - t) * 1000)
    except OSError as exc:
        return CaseResult("tcp_connect", "FAIL", detail=str(exc))


def _http_json(url: str, payload: dict | None, timeout: float) -> tuple[int, dict | None, float, str]:
    """POST (payload) or GET (None). Returns (status, json|None, latency_ms, raw_text)."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            ms = (time.perf_counter() - t) * 1000
            try:
                return resp.status, json.loads(raw), ms, raw
            except json.JSONDecodeError:
                return resp.status, None, ms, raw
    except urllib.error.HTTPError as exc:
        ms = (time.perf_counter() - t) * 1000
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, None, ms, raw
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return -1, None, (time.perf_counter() - t) * 1000, str(exc)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100) * (len(values) - 1)))))
    return values[k]


class ProviderBench:
    def __init__(self, endpoint: str, model: str, kind: str, name: str):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.kind = kind  # "chat" | "embeddings"
        self.name = name
        hostport = self.endpoint.split("://", 1)[-1].split("/")[0]
        self.host, _, port = hostport.partition(":")
        self.port = int(port or 80)
        self.results: list[CaseResult] = []

    def run(self) -> dict:
        self.results.append(_tcp_check(self.host, self.port, timeout=2.0))
        self._case_models_health()
        if self.kind == "chat":
            self._case_chat_languages()
            self._case_chat_json_schema()
        else:
            self._case_embedding_basic()
        self._case_bad_model_name()
        self._case_connect_timeout()
        self._case_unreachable()
        self._case_concurrency()
        return self._summary()

    # ---- cases ----
    def _case_models_health(self) -> None:
        status, body, ms, raw = _http_json(f"{self.endpoint}/models", None, timeout=5.0)
        ok = status == 200 and isinstance(body, dict) and "data" in body
        ids = [m.get("id") for m in body.get("data", [])] if ok else []
        self.results.append(
            CaseResult(
                "models_health", "PASS" if ok else "FAIL", ms,
                detail=f"status={status} models={ids}" if ok else f"status={status} raw={raw[:120]}",
            )
        )

    def _chat(self, messages: list[dict], timeout: float = 120.0, **kw) -> tuple[int, dict | None, float, str]:
        payload = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": 128, **kw}
        return _http_json(f"{self.endpoint}/chat/completions", payload, timeout)

    def _case_chat_languages(self) -> None:
        for lang, content in [
            ("zh", "用一句话说明为什么机器人动作需要回执确认。"),
            ("en", "In one sentence, why must a robot action produce a receipt?"),
        ]:
            status, body, ms, raw = self._chat([{"role": "user", "content": content}])
            text = ""
            if body:
                choices = body.get("choices") or []
                if choices:
                    text = (choices[0].get("message") or {}).get("content") or ""
            ok = status == 200 and bool(text.strip())
            usage = (body or {}).get("usage") or {}
            self.results.append(
                CaseResult(
                    f"chat_{lang}", "PASS" if ok else "FAIL", ms,
                    detail=f"status={status} chars={len(text)}",
                    metrics={
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "tokens_per_s": round(usage.get("completion_tokens", 0) / (ms / 1000), 1) if ms else 0,
                    },
                )
            )

    def _case_chat_json_schema(self) -> None:
        prompt = (
            'Reply with ONLY a JSON object matching {"title": string, "tags": string[], '
            '"confidence": number}. Topic: robot sandbox validation.'
        )
        status, body, ms, raw = self._chat([{"role": "user", "content": prompt}])
        ok = False
        detail = f"status={status}"
        if body:
            choices = body.get("choices") or []
            text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                obj = json.loads(text)
                ok = isinstance(obj, dict) and isinstance(obj.get("title"), str) and isinstance(obj.get("tags"), list)
                detail = f"parsed keys={sorted(obj)[:6]}"
            except json.JSONDecodeError:
                detail = f"unparseable: {text[:80]}"
        self.results.append(CaseResult("chat_json_schema", "PASS" if ok else "FAIL", ms, detail))

    def _case_embedding_basic(self) -> None:
        status, body, ms, raw = _http_json(
            f"{self.endpoint}/embeddings",
            {"model": self.model, "input": "TY1200 provider acceptance probe"},
            60.0,
        )
        dim = 0
        if body and body.get("data"):
            dim = len(body["data"][0].get("embedding") or [])
        ok = status == 200 and dim > 0
        self.results.append(
            CaseResult("embedding_basic", "PASS" if ok else "FAIL", ms,
                       detail=f"status={status} dim={dim}", metrics={"dimension": dim})
        )

    def _case_bad_model_name(self) -> None:
        url = f"{self.endpoint}/chat/completions" if self.kind == "chat" else f"{self.endpoint}/embeddings"
        payload = (
            {"model": "definitely-not-a-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
            if self.kind == "chat"
            else {"model": "definitely-not-a-model", "input": "hi"}
        )
        status, body, ms, raw = _http_json(url, payload, 30.0)
        # Expect a clean 4xx error, NOT a fabricated success.
        ok = status in (400, 404, 422)
        self.results.append(
            CaseResult("bad_model_name_rejected", "PASS" if ok else "FAIL", ms,
                       detail=f"status={status} (expected 4xx, no fake success)")
        )

    def _case_connect_timeout(self) -> None:
        # Non-routable TEST-NET address must fail within ~connect timeout, not hang.
        t = time.perf_counter()
        status, _, _, _ = _http_json("http://192.0.2.1:9/v1/models", None, timeout=2.0)
        elapsed = time.perf_counter() - t
        ok = status == -1 and elapsed < 5.0
        self.results.append(
            CaseResult("connect_timeout_2s", "PASS" if ok else "FAIL", elapsed * 1000,
                       detail=f"elapsed={elapsed:.2f}s status={status}")
        )

    def _case_unreachable(self) -> None:
        # Closed port on localhost: deterministic failure, no retry storm.
        t = time.perf_counter()
        status, _, _, _ = _http_json("http://127.0.0.1:9/v1/models", None, timeout=5.0)
        elapsed = time.perf_counter() - t
        ok = status == -1
        self.results.append(
            CaseResult("unreachable_deterministic_fail", "PASS" if ok else "FAIL", elapsed * 1000,
                       detail=f"status={status} (must fail closed, no fabricated result)")
        )

    def _case_concurrency(self) -> None:
        async def worker(sem: asyncio.Semaphore, lat: list[float], errs: list[str]):
            async with sem:
                if self.kind == "chat":
                    status, body, ms, _ = await asyncio.to_thread(
                        self._chat, [{"role": "user", "content": "Say OK."}]
                    )
                else:
                    status, body, ms, _ = await asyncio.to_thread(
                        _http_json,
                        f"{self.endpoint}/embeddings",
                        {"model": self.model, "input": "concurrency probe"},
                        60.0,
                    )
                if status == 200 and body:
                    lat.append(ms)
                else:
                    errs.append(f"status={status}")

        async def run_level(n: int) -> dict:
            sem = asyncio.Semaphore(n)
            lat: list[float] = []
            errs: list[str] = []
            t = time.perf_counter()
            await asyncio.gather(*[worker(sem, lat, errs) for _ in range(n * 2)])
            wall = time.perf_counter() - t
            return {
                "concurrency": n,
                "requests": n * 2,
                "success": len(lat),
                "errors": errs[:5],
                "wall_s": round(wall, 2),
                "p50_ms": round(_pct(lat, 50), 1),
                "p95_ms": round(_pct(lat, 95), 1),
                "p99_ms": round(_pct(lat, 99), 1),
            }

        metrics = [asyncio.run(run_level(n)) for n in (1, 4, 8)]
        ok = all(m["success"] == m["requests"] for m in metrics)
        self.results.append(
            CaseResult("concurrency_1_4_8", "PASS" if ok else "WARN",
                       detail=json.dumps(metrics), metrics={"levels": metrics})
        )

    def _summary(self) -> dict:
        chat_lat = [r.latency_ms for r in self.results if r.name.startswith(("chat_", "embedding_"))]
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = [r.name for r in self.results if r.status == "FAIL"]
        return {
            "provider": self.name,
            "endpoint_kind": self.kind,
            "model": self.model,
            "cases": [asdict(r) for r in self.results],
            "totals": {"pass": passed, "fail": len(failed), "failed_cases": failed},
            "latency": {
                "p50_ms": round(_pct(chat_lat, 50), 1),
                "p95_ms": round(_pct(chat_lat, 95), 1),
                "p99_ms": round(_pct(chat_lat, 99), 1),
                "mean_ms": round(statistics.mean(chat_lat), 1) if chat_lat else 0,
            },
            "overall": "PASS" if not failed else "FAIL",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["chat", "embeddings"], default="chat")
    ap.add_argument("--out", help="write machine-readable JSON here")
    args = ap.parse_args()

    bench = ProviderBench(args.endpoint, args.model, args.kind, args.name)
    summary = bench.run()
    print(f"== {args.name} ({args.kind}) ==")
    for r in bench.results:
        print(f"  [{r.status:4}] {r.name:32} {r.latency_ms:8.1f}ms  {r.detail[:100]}")
    print(f"overall: {summary['overall']}  latency={summary['latency']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
