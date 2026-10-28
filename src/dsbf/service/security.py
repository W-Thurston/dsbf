# dsbf/service/security.py
"""Rules for a service that has no authentication: it answers only local requests."""

import json

from starlette.types import ASGIApp, Receive, Scope, Send

LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def host_name(header: str) -> str:
    """The host in a ``Host`` header, without its port."""
    if header.startswith("["):
        return header[: header.find("]") + 1]
    return header.split(":", 1)[0]


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
        body = json.dumps(
            {
                "type": "about:blank",
                "title": "Misdirected Request",
                "status": 421,
                "detail": f"this service answers only local hosts, not {host!r}",
                "instance": scope["path"],
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 421,
                "headers": [
                    (b"content-type", b"application/problem+json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
