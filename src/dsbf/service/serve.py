# dsbf/service/serve.py
"""Run the service on the local run store (behind ``dsbf dashboard``)."""

import threading
import webbrowser
from pathlib import Path

import uvicorn

from dsbf.service.app import create_app
from dsbf.service.security import LOCAL_HOSTS
from dsbf.store import open_repository

DEFAULT_PORT = 8765
_BROWSER_DELAY_SECONDS = 0.7


class NotLocalError(ValueError):
    """The service was asked to listen on an address other than this machine."""


def dashboard_directory() -> Path | None:
    """The built dashboard: ``frontend/dist`` in a source checkout (the build a
    developer just made), else the one packaged in the wheel, else ``None`` (a wheel
    built without the frontend)."""
    dist = Path(__file__).parents[3] / "frontend" / "dist"
    if (dist / "index.html").is_file():
        return dist
    packaged = Path(__file__).parent / "dashboard"
    return packaged if (packaged / "index.html").is_file() else None


def serve(
    host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, open_browser: bool = True
) -> None:
    """Serve the API (and the dashboard, when built) over the store in ``DSBF_HOME``
    until interrupted.

    Raises:
        NotLocalError: ``host`` is not a local address; the service has no
            authentication, so it never listens beyond this machine.
    """
    if host not in {name.strip("[]") for name in LOCAL_HOSTS}:
        raise NotLocalError(
            f"{host!r} is not a local address; the service has no authentication "
            "and listens on 127.0.0.1, localhost or ::1 only"
        )
    repository = open_repository()
    try:
        app = create_app(repository, static_dir=dashboard_directory())
        if open_browser:
            shown = f"[{host}]" if ":" in host else host
            url = f"http://{shown}:{port}/"
            threading.Timer(_BROWSER_DELAY_SECONDS, webbrowser.open, [url]).start()
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        repository.close()
