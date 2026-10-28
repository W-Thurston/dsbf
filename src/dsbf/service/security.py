# dsbf/service/security.py
"""Rules for a service that has no authentication: it answers only local requests."""

import json
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def host_name(header: str) -> str:
    """The host in a ``Host`` header, without its port."""
    if header.startswith("["):
        return header[: header.find("]") + 1]
    return header.split(":", 1)[0]


def _refuse(status: int, title: str, detail: str, path: str) -> tuple[dict, bytes]:
    body = json.dumps(
        {
            "type": "about:blank",
            "title": title,
            "status": status,
            "detail": detail,
            "instance": path,
        }
    ).encode()
    start = {
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/problem+json"),
            (b"content-length", str(len(body)).encode()),
        ],
    }
    return start, body


class LocalHostOnly:
    """Refuse (421) a request whose ``Host`` is not one of ``hosts``.

    A page on another site can make the browser send requests to ``localhost`` under
    the attacker's own name (DNS rebinding); the ``Host`` it carries gives it away.
    """

    def __init__(self, app: ASGIApp, hosts: tuple[str, ...] = LOCAL_HOSTS) -> None:
        self.app = app
        self.hosts = frozenset(host.lower() for host in hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        host = host_name(headers.get(b"host", b"").decode("latin-1")).lower()
        if host in self.hosts:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        start, body = _refuse(
            421,
            "Misdirected Request",
            f"this service answers only local hosts, not {host!r}",
            scope["path"],
        )
        await send(start)
        await send({"type": "http.response.body", "body": body})


class JsonWritesOnly:
    """Refuse a write a web page could have sent for the visitor.

    A page on any site can make the browser POST a form to ``localhost``. A write
    must carry a ``Content-Type`` of ``application/json`` when it has a body (a form
    cannot send one without a CORS preflight, which this service does not grant
    unless ``origins`` names the page), and an ``Origin`` header, when the browser
    sends one, must be a local page or one of ``origins``.
    """

    def __init__(
        self,
        app: ASGIApp,
        hosts: tuple[str, ...] = LOCAL_HOSTS,
        origins: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.hosts = frozenset(host.lower() for host in hosts)
        self.origins = frozenset(origin.rstrip("/").lower() for origin in origins)

    def _foreign(self, origin: str) -> bool:
        if origin.rstrip("/").lower() in self.origins:
            return False
        host = (urlsplit(origin).hostname or "").lower()
        return not (host in self.hosts or f"[{host}]" in self.hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        origin = headers.get(b"origin", b"").decode("latin-1")
        content_type = headers.get(b"content-type", b"").decode("latin-1")
        has_body = headers.get(b"content-length", b"0") != b"0" or (
            b"transfer-encoding" in headers
        )
        if origin and self._foreign(origin):
            refusal = (403, "Forbidden", f"writes from {origin!r} are not accepted")
        elif has_body and content_type.split(";")[0].strip().lower() != (
            "application/json"
        ):
            refusal = (
                415,
                "Unsupported Media Type",
                "send the body as application/json",
            )
        else:
            await self.app(scope, receive, send)
            return
        start, body = _refuse(*refusal, scope["path"])
        await send(start)
        await send({"type": "http.response.body", "body": body})
