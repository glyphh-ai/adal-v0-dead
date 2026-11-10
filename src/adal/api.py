"""adaL's HTTP API. It is called by the Glyphh server, never by a customer directly: the server signs people in,
checks their plan and meters them, then calls here with the service token and the tenant the call is for.

    Authorization: Bearer <ADAL_TOKEN>
    X-Glyphh-Tenant: <tenant>

    GET    /v1/health                    no token: the process is up, and with which model
    POST   /v1/facts        {text|texts[], source?}        tell one or more facts
    GET    /v1/facts        ?q=&after=&limit=              list them, oldest first; `next` pages on
    DELETE /v1/facts/{id}                                  forget one
    DELETE /v1/facts                                       forget everything (the tenant's memory is removed)
    POST   /v1/ingest       {text, source?}                free text, rewritten into facts and told
    POST   /v1/chat         {messages[], max_tokens?}      a reply, with the tenant's memory in the loop

Every handler is a plain function: FastAPI runs them on its thread pool, and the engine's own locks do the rest."""

import hmac
import logging
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from adal import __version__
from adal.backends import Backend, make
from adal.config import Settings, load
from adal.engine import Engine
from adal.errors import AdalError, Invalid, Unauthorized
from adal.store import check_tenant

log = logging.getLogger("adal")


class Tell(BaseModel):
    text: str | None = None
    texts: list[str] | None = Field(default=None, max_length=500)
    source: str | None = Field(default=None, max_length=200)


class Ingest(BaseModel):
    text: str = Field(max_length=20_000)
    source: str | None = Field(default=None, max_length=200)


class Chat(BaseModel):
    messages: list[dict[str, Any]] = Field(max_length=100)
    max_tokens: int | None = Field(default=None, ge=1)


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    settings = settings or load()
    engine = Engine(backend or make(settings.backend, settings.model), settings)
    app = FastAPI(title="adaL", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.engine = engine

    def tenant(authorization: str = Header(default=""), x_glyphh_tenant: str = Header(default="")) -> str:
        given = authorization.removeprefix("Bearer ").strip()
        if settings.token and not hmac.compare_digest(given.encode(), settings.token.encode()):
            raise Unauthorized("A valid service token is required.")
        return check_tenant(x_glyphh_tenant)

    @app.exception_handler(AdalError)
    def _refused(request: Request, err: AdalError) -> JSONResponse:
        return JSONResponse({"error": {"code": err.code, "message": err.message, **err.detail}}, err.status)

    @app.exception_handler(RequestValidationError)
    def _malformed(request: Request, err: RequestValidationError) -> JSONResponse:
        where = ".".join(str(x) for x in err.errors()[0]["loc"][1:]) if err.errors() else ""
        return JSONResponse(
            {"error": {"code": "E_INVALID", "message": "The request is malformed.", "field": where}}, 422
        )

    @app.exception_handler(Exception)
    def _crashed(request: Request, err: Exception) -> JSONResponse:
        log.exception("crash on %s %s", request.method, request.url.path)
        return JSONResponse({"error": {"code": "E_INTERNAL", "message": "Something went wrong."}}, 500)

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "backend": settings.backend, "model": engine.backend.name}

    @app.post("/v1/facts", status_code=201)
    def tell(body: Tell, who: str = Depends(tenant)) -> dict[str, Any]:
        texts = [*([body.text] if body.text is not None else []), *(body.texts or [])]
        if not texts:
            raise Invalid("Give `text` or `texts`.", reason="text")
        return {"facts": [f.public() for f in engine.tell(who, texts, body.source)]}

    @app.get("/v1/facts")
    def facts(q: str = "", after: int = 0, limit: int = 50, who: str = Depends(tenant)) -> dict[str, Any]:
        limit = max(1, min(limit, 200))
        found = engine.facts(who, q, after, limit + 1)
        page = found[:limit]
        return {"facts": [f.public() for f in page], "next": page[-1].seq if len(found) > limit else None}

    @app.delete("/v1/facts/{fact_id}", status_code=204)
    def forget(fact_id: str, who: str = Depends(tenant)) -> Response:
        engine.forget(who, fact_id)
        return Response(status_code=204)

    @app.delete("/v1/facts", status_code=204)
    def forget_all(who: str = Depends(tenant)) -> Response:
        engine.forget_all(who)
        return Response(status_code=204)

    @app.post("/v1/ingest", status_code=201)
    def ingest(body: Ingest, who: str = Depends(tenant)) -> dict[str, Any]:
        kept, dropped = engine.ingest(who, body.text, body.source)
        return {"facts": [f.public() for f in kept], "dropped": dropped}

    @app.post("/v1/chat")
    def chat(body: Chat, who: str = Depends(tenant)) -> dict[str, Any]:
        reply = engine.chat(who, body.messages, body.max_tokens)
        memory = {"recalled": reply.recalled, "fired": reply.fired}
        return {"reply": reply.text, "memory": memory, "tokens": reply.tokens}

    return app
