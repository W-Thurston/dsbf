# dsbf/service/problems.py
"""Errors as RFC 9457 problem details."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from dsbf.store import StoreError
from dsbf.store.repository import JobNotFoundError, RunNotFoundError

PROBLEM_MEDIA_TYPE = "application/problem+json"


class Problem(BaseModel):
    """What went wrong, in the shape of RFC 9457."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str


class InvalidField(BaseModel):
    """One request field that failed validation."""

    location: str
    message: str


class ValidationProblem(Problem):
    """A request that did not pass validation, with each field that failed."""

    errors: list[InvalidField]


def problem_response(
    request: Request, status: int, title: str, detail: str, **extra: object
) -> JSONResponse:
    body = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        **extra,
    }
    return JSONResponse(body, status_code=status, media_type=PROBLEM_MEDIA_TYPE)


def install_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(RunNotFoundError)
    async def _not_found(request: Request, error: RunNotFoundError) -> JSONResponse:
        return problem_response(request, 404, "Not Found", str(error))

    @app.exception_handler(JobNotFoundError)
    async def _no_job(request: Request, error: JobNotFoundError) -> JSONResponse:
        return problem_response(request, 404, "Not Found", str(error))

    @app.exception_handler(StoreError)
    async def _store(request: Request, error: StoreError) -> JSONResponse:
        return problem_response(request, 500, "Store Error", str(error))

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        fields = [
            {
                "location": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        return problem_response(
            request,
            422,
            "Unprocessable Content",
            "the request did not pass validation",
            errors=fields,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, error: StarletteHTTPException) -> JSONResponse:
        detail = error.detail if isinstance(error.detail, str) else "request failed"
        return problem_response(request, error.status_code, detail, detail)
