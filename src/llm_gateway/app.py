"""HTTP layer: OpenAI-compatible routes on top of the gateway core."""

# No ``from __future__ import annotations`` here: FastAPI must evaluate the route
# signatures, and the ``Authenticated`` alias only exists inside ``create_app``.
import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from llm_gateway import __version__
from llm_gateway.config import GatewayConfig, KeyConfig
from llm_gateway.errors import GatewayError
from llm_gateway.gateway import Gateway
from llm_gateway.logs import request_id
from llm_gateway.metrics import CONTENT_TYPE
from llm_gateway.router import Sleep

logger = logging.getLogger(__name__)

CHAT_PATH = "/v1/chat/completions"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """Validates the fields the gateway relies on; everything else is forwarded untouched."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)


def create_app(
    config: GatewayConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Sleep = asyncio.sleep,
) -> FastAPI:
    gateway = Gateway(config, transport=transport, sleep=sleep)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await gateway.aclose()

    app = FastAPI(title="llm-gateway", version=__version__, lifespan=lifespan)
    app.state.gateway = gateway

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request_id.set(rid)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["x-request-id"] = rid
        if request.url.path == CHAT_PATH:
            # Labels come from request.state, set only after validation, so arbitrary
            # client input can never blow up metric cardinality.
            model = getattr(request.state, "model", "-")
            key = getattr(request.state, "key", "-")
            gateway.metrics.requests.inc(key=key, model=model, status=str(response.status_code))
            gateway.metrics.latency.observe(elapsed, model=model)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed * 1000, 1),
            },
        )
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"] if part != "body")
        error = GatewayError(400, f"{location}: {first['msg']}", param=location or None)
        return JSONResponse(error.body(), status_code=400)

    def authenticate(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> KeyConfig:
        key = gateway.authenticate(authorization)
        request.state.key = key.name
        return key

    Authenticated = Annotated[KeyConfig, Depends(authenticate)]

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(gateway.render_metrics(), media_type=CONTENT_TYPE)

    @app.get("/v1/models")
    async def list_models(_: Authenticated) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": alias, "object": "model", "created": 0, "owned_by": "llm-gateway"}
                for alias in config.models
            ],
        }

    @app.post(CHAT_PATH, response_model=None)
    async def chat_completions(
        request: Request, body: ChatCompletionRequest, key: Authenticated
    ) -> Response:
        gateway.admit(key)
        gateway.check_model(body.model)
        request.state.model = body.model
        payload = body.model_dump(exclude_unset=True)

        if body.stream:
            target, events = await gateway.stream(key, payload)
            return StreamingResponse(
                events,
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-gateway-target": str(target)},
            )

        completion = await gateway.complete(key, payload)
        return JSONResponse(
            completion.body,
            headers={
                "x-cache": "hit" if completion.cached else "miss",
                "x-gateway-target": str(completion.target),
            },
        )

    return app
