# dsbf/service/__init__.py
"""The local HTTP service: API v2 over the run store.

Needs the ``server`` extra. The core never imports this package.
"""

try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401

    import dsbf.store  # noqa: F401
except ImportError as error:
    raise ImportError(
        "dsbf.service needs the `server` extra. Install it with "
        "`pip install 'dsbf[server]'`."
    ) from error

from dsbf.service.app import API_VERSION, create_app, openapi_document

__all__ = ["API_VERSION", "create_app", "openapi_document"]
