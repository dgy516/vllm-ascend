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


def _split_urls(value: str) -> list[str]:
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


def _pick(urls: list[str], role: str) -> str:
    if not urls:
        raise HTTPException(status_code=503, detail=f"no {role} backend configured")
    return random.choice(urls)


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
    decode_payload = await _prepare_decode_payload(payload, endpoint, request_id)
    decoder_url = _pick(app.state.decoder_urls, "decoder")
    return await _post_json(f"{decoder_url}{endpoint}", decode_payload, request_id)


async def _forward_stream(payload: dict[str, Any], endpoint: str, request_id: str) -> AsyncIterator[bytes]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vLLM Ascend CI P/D proxy.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefiller-urls", required=True)
    parser.add_argument("--decoder-urls", required=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    app.state.prefiller_urls = _split_urls(args.prefiller_urls)
    app.state.decoder_urls = _split_urls(args.decoder_urls)
    LOG.info(
        "proxy=%s:%s prefiller=%s decoder=%s",
        args.host,
        args.port,
        app.state.prefiller_urls,
        app.state.decoder_urls,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
