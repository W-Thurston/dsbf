# dsbf/service/app.py
"""The FastAPI application over a run repository."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from dsbf.service.job_routes import job_router
from dsbf.service.jobs import JobRunner
from dsbf.service.problems import install_problem_handlers
from dsbf.service.routes import router
from dsbf.service.security import LOCAL_HOSTS, JsonWritesOnly, LocalHostOnly
from dsbf.store import RunRepository

API_VERSION = "1.6.0"
"""The version of the OpenAPI document; the breaking-change gate reads it."""


class DashboardFiles(StaticFiles):
    """The built dashboard, which answers a page route (``/runs/<id>``) with its
    ``index.html`` so the browser router can take over after a reload or a pasted link.
    A missing file with an extension, or under ``/v2``, is still a 404."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if error.status_code != 404 or "." in Path(path).name:
                raise
            return await super().get_response("index.html", scope)


def create_app(
    repository: RunRepository,
    *,
    cors_origins: Sequence[str] = (),
    allowed_hosts: tuple[str, ...] = LOCAL_HOSTS,
    static_dir: Path | None = None,
) -> FastAPI:
    """The service over ``repository``.

    It answers only requests whose ``Host`` is in ``allowed_hosts`` and sends no CORS
    headers unless ``cors_origins`` names origins. A write needs a JSON body and a
    local (or named) ``Origin``. The job runner starts and stops with the app. When
    ``static_dir`` exists it serves the built dashboard from it, after the API
    routes. The caller closes the repository.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runner.start()
        try:
            yield
        finally:
            app.state.runner.stop()

    app = FastAPI(
        title="DSBF API",
        version=API_VERSION,
        summary="Stored DSBF runs, and jobs that profile a local file into the store.",
        lifespan=lifespan,
        docs_url="/v2/docs",
        redoc_url=None,
        openapi_url="/v2/openapi.json",
    )
    app.state.repository = repository
    app.state.runner = JobRunner(repository)
    install_problem_handlers(app)
    app.include_router(router)
    app.include_router(job_router)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
        )
    app.add_middleware(JsonWritesOnly, hosts=allowed_hosts, origins=tuple(cors_origins))
    app.add_middleware(LocalHostOnly, hosts=allowed_hosts)
    if static_dir is not None and static_dir.is_dir():
        app.mount(
            "/", DashboardFiles(directory=static_dir, html=True), name="dashboard"
        )
    return app


def openapi_document() -> dict[str, Any]:
    """The OpenAPI document, built without a repository or a server."""
    return create_app(cast("RunRepository", None)).openapi()
