# dsbf/service/app.py
"""The FastAPI application over a run repository."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dsbf.service.problems import install_problem_handlers
from dsbf.service.routes import router
from dsbf.service.security import LOCAL_HOSTS, LocalHostOnly
from dsbf.store import RunRepository

API_VERSION = "1.0.0"
"""The version of the OpenAPI document; the breaking-change gate reads it."""


def create_app(
    repository: RunRepository,
    *,
    cors_origins: Sequence[str] = (),
    allowed_hosts: tuple[str, ...] = LOCAL_HOSTS,
    static_dir: Path | None = None,
) -> FastAPI:
    """The service over ``repository``.

    It answers only requests whose ``Host`` is in ``allowed_hosts`` and sends no CORS
    headers unless ``cors_origins`` names origins. When ``static_dir`` exists it
    serves the built dashboard from it, after the API routes. The caller closes the
    repository.
    """
    app = FastAPI(
        title="DSBF API",
        version=API_VERSION,
        summary="Read access to stored DSBF runs.",
        docs_url="/v2/docs",
        redoc_url=None,
        openapi_url="/v2/openapi.json",
    )
    app.state.repository = repository
    install_problem_handlers(app)
    app.include_router(router)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=list(cors_origins), allow_methods=["GET"]
        )
    app.add_middleware(LocalHostOnly, hosts=allowed_hosts)
    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="dashboard")
    return app


def openapi_document() -> dict[str, Any]:
    """The OpenAPI document, built without a repository or a server."""
    return create_app(cast("RunRepository", None)).openapi()
