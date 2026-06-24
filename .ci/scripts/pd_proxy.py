#!/usr/bin/env python3
"""Small OpenAI-compatible P/D proxy for DeployCase smoke validation.

The proxy is intentionally scoped for CI: it sends a short prefill request to a
prefiller worker, forwards returned kv_transfer_params to a decoder worker, and
returns the decoder response. It lives under .ci because Jenkins runtime
containers only mount .ci, reports, logs, and MODEL_ROOT.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

LOG = logging.getLogger("vllm_ascend_ci_pd_proxy")

app = FastAPI()
client: httpx.AsyncClient | None = None

LAYERWISE_MODE = "layerwise"
V1_MODE = "v1"


def _split_urls(value: str) -> list[str]:
    return [_normalize_backend_address(item) for item in value.split(",") if item.strip()]


def _normalize_backend_address(value: str) -> str:
    address = value.strip().rstrip("/")
    if "://" in address:
        return address
    return f"http://{address}"


def _pick(urls: list[str], role: str) -> str:
    if not urls:
        raise HTTPException(status_code=503, detail=f"no {role} backend configured")
    return random.choice(urls)


def _api_request_ids(endpoint: str, request_id: str) -> list[str]:
    if endpoint == "/v1/completions":
        return [request_id, f"cmpl-{request_id}-0"]
    if endpoint == "/v1/chat/completions":
        return [request_id, f"chatcmpl-{request_id}"]
    return [request_id]


def _origin_request_id(endpoint: str, external_request_id: str) -> str:
    if endpoint == "/v1/completions" and external_request_id.startswith("cmpl-") and external_request_id.endswith("-0"):
        return external_request_id.removeprefix("cmpl-")[:-2]
    if endpoint == "/v1/chat/completions" and external_request_id.startswith("chatcmpl-"):
        return external_request_id.removeprefix("chatcmpl-")
    return external_request_id


async def _client() -> httpx.AsyncClient:
    if client is None:
        raise HTTPException(status_code=500, detail="proxy HTTP client is not initialized")
    return client


def _prefill_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prefill = dict(payload)
    prefill["stream"] = False
    prefill["max_tokens"] = 1
    prefill.pop("stream_options", None)
    prefill.pop("max_completion_tokens", None)
    prefill["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return prefill


def _layerwise_prefill_payload(payload: dict[str, Any], kv_transfer_params: dict[str, Any]) -> dict[str, Any]:
    prefill = dict(payload)
    prefill["stream"] = False
    prefill["max_tokens"] = 1
    prefill["min_tokens"] = 1
    if "max_completion_tokens" in prefill:
        prefill["max_completion_tokens"] = 1
    prefill.pop("stream_options", None)
    prefill["kv_transfer_params"] = kv_transfer_params
    return prefill


def _layerwise_decode_payload(payload: dict[str, Any], endpoint: str, request_id: str) -> dict[str, Any]:
    decode_payload = dict(payload)
    decode_payload["kv_transfer_params"] = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "metaserver": f"http://{app.state.proxy_endpoint}/v1/metaserver",
    }
    for key in _api_request_ids(endpoint, request_id):
        app.state.layerwise_requests[key] = (dict(payload), endpoint)
    return decode_payload


async def _post_json(url: str, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
    response = await (await _client()).post(url, json=payload, headers={"x-request-id": request_id})
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail=f"backend returned non-object JSON: {url}")
    return parsed


async def _prepare_decode_payload(payload: dict[str, Any], endpoint: str, request_id: str) -> dict[str, Any]:
    prefiller_url = _pick(app.state.prefiller_urls, "prefiller")
    prefill_response = await _post_json(f"{prefiller_url}{endpoint}", _prefill_payload(payload), request_id)
    decode_payload = dict(payload)
    kv_transfer_params = prefill_response.get("kv_transfer_params")
    if kv_transfer_params:
        decode_payload["kv_transfer_params"] = kv_transfer_params
    return decode_payload


async def _forward_non_stream(payload: dict[str, Any], endpoint: str, request_id: str) -> dict[str, Any]:
    if app.state.proxy_mode == LAYERWISE_MODE:
        decoder_url = _pick(app.state.decoder_urls, "decoder")
        decode_payload = _layerwise_decode_payload(payload, endpoint, request_id)
        return await _post_json(f"{decoder_url}{endpoint}", decode_payload, request_id)
    decode_payload = await _prepare_decode_payload(payload, endpoint, request_id)
    decoder_url = _pick(app.state.decoder_urls, "decoder")
    return await _post_json(f"{decoder_url}{endpoint}", decode_payload, request_id)


async def _forward_stream(payload: dict[str, Any], endpoint: str, request_id: str) -> AsyncIterator[bytes]:
    if app.state.proxy_mode == LAYERWISE_MODE:
        decode_payload = _layerwise_decode_payload(payload, endpoint, request_id)
    else:
        decode_payload = await _prepare_decode_payload(payload, endpoint, request_id)
    decoder_url = _pick(app.state.decoder_urls, "decoder")
    async with (await _client()).stream(
        "POST",
        f"{decoder_url}{endpoint}",
        json=decode_payload,
        headers={"x-request-id": request_id},
    ) as response:
        if response.status_code >= 400:
            body = await response.aread()
            raise HTTPException(status_code=response.status_code, detail=body.decode(errors="replace"))
        async for chunk in response.aiter_bytes():
            if chunk:
                yield chunk


@app.on_event("startup")
async def on_startup() -> None:
    global client
    client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=1024, max_keepalive_connections=1024))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global client
    if client is not None:
        await client.aclose()
        client = None


@app.get("/health")
async def health() -> JSONResponse:
    async def probe(url: str) -> bool:
        try:
            response = await (await _client()).get(f"{url}/health", timeout=5)
            return 200 <= response.status_code < 300
        except Exception:
            return False

    prefiller_results = [await probe(url) for url in app.state.prefiller_urls]
    decoder_results = [await probe(url) for url in app.state.decoder_urls]
    healthy = all(prefiller_results) and all(decoder_results)
    return JSONResponse(
        {
            "proxy": "healthy" if healthy else "unhealthy",
            "mode": app.state.proxy_mode,
            "prefiller_backends": dict(zip(app.state.prefiller_urls, prefiller_results, strict=False)),
            "decoder_backends": dict(zip(app.state.decoder_urls, decoder_results, strict=False)),
        },
        status_code=200 if healthy else 503,
    )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    decoder_url = _pick(app.state.decoder_urls, "decoder")
    response = await (await _client()).get(f"{decoder_url}/v1/models")
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return JSONResponse(response.json())


async def _handle_openai_request(request: Request, endpoint: str):
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    if payload.get("stream"):
        return StreamingResponse(_forward_stream(payload, endpoint, request_id), media_type="text/event-stream")
    return JSONResponse(await _forward_non_stream(payload, endpoint, request_id))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_request(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_openai_request(request, "/v1/completions")


@app.post("/v1/metaserver")
async def metaserver(request: Request) -> JSONResponse:
    if app.state.proxy_mode != LAYERWISE_MODE:
        raise HTTPException(status_code=404, detail="metaserver is only available in layerwise mode")
    kv_transfer_params = await request.json()
    if not isinstance(kv_transfer_params, dict):
        raise HTTPException(status_code=400, detail="metaserver body must be a JSON object")
    request_id = str(kv_transfer_params.get("request_id") or "")
    if not request_id:
        raise HTTPException(status_code=400, detail="kv_transfer_params.request_id is required")
    stored = app.state.layerwise_requests.get(request_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"unknown layerwise request_id: {request_id}")
    payload, endpoint = stored
    origin_request_id = _origin_request_id(endpoint, request_id)
    prefiller_url = _pick(app.state.prefiller_urls, "prefiller")
    await _post_json(
        f"{prefiller_url}{endpoint}",
        _layerwise_prefill_payload(payload, kv_transfer_params),
        origin_request_id,
    )
    return JSONResponse({"status": "ok"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vLLM Ascend CI P/D proxy.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    app.state.prefiller_urls = _split_urls(os.getenv("PREFILL_SERVERS", ""))
    app.state.decoder_urls = _split_urls(os.getenv("DECODE_SERVERS", ""))
    app.state.proxy_mode = os.getenv("VLLM_CI_PD_PROXY_MODE", V1_MODE).strip().lower() or V1_MODE
    if app.state.proxy_mode not in {V1_MODE, LAYERWISE_MODE}:
        raise SystemExit(f"unsupported VLLM_CI_PD_PROXY_MODE={app.state.proxy_mode!r}")
    app.state.proxy_endpoint = os.getenv("VLLM_CI_PD_PROXY_ENDPOINT", f"{args.host}:{args.port}").strip()
    if app.state.proxy_mode == LAYERWISE_MODE and app.state.proxy_endpoint.startswith("0.0.0.0:"):
        raise SystemExit("VLLM_CI_PD_PROXY_ENDPOINT must be reachable by decoder in layerwise mode")
    app.state.layerwise_requests = {}
    if not app.state.prefiller_urls:
        raise SystemExit("PREFILL_SERVERS is required")
    if not app.state.decoder_urls:
        raise SystemExit("DECODE_SERVERS is required")
    LOG.info(
        "proxy=%s:%s mode=%s endpoint=%s prefiller=%s decoder=%s",
        args.host,
        args.port,
        app.state.proxy_mode,
        app.state.proxy_endpoint,
        app.state.prefiller_urls,
        app.state.decoder_urls,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
