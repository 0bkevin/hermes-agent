"""
Brio relay tunnel connector.

Turns Hermes into a direct participant in the Brio control plane: instead of
running the Go ``brio companion`` binary beside Hermes, this module dials the
Brio cloud relay over a WebSocket and forwards tunnel ``request`` frames to
the local Hermes API server (``gateway/platforms/api_server.py``), streaming
SSE responses back chunk-by-chunk.

Behavior is a port of brio's Go implementation
(``apps/companion/internal/tunnel/tunnel.go``):

- WebSocket at ``GET {relay}/tunnel/companion/{agentID}?token={relay_token}``
  with a 12 MiB receive limit and no library heartbeat (we send our own
  protocol-level pings every 25s with a 10s pong timeout).
- Frame protocol (JSON text frames; ``packages/protocol/tunnel-frame.schema.json``):
  inbound ``request`` → local HTTP request; outbound ``response`` /
  ``stream_chunk`` + ``stream_end`` (SSE; ``stream_end`` carries the final
  parsed JSON body when available, else ``null``) / ``error`` frames; inbound
  ``ping`` → outbound ``pong``.
- Companion-era path compatibility mapping (the mobile app still speaks
  companion paths like ``/chat/responses``).
- Concurrency cap of 16 in-flight local requests (``COMPANION_BUSY`` beyond
  that) and a 10 MiB response/stream cap (``RESPONSE_TOO_LARGE``).
- Reconnect with exponential backoff 1s doubling to a 32s cap, reset after a
  connection that lasted >30s, with small jitter.

Runs standalone via ``hermes brio connect`` (see ``hermes_cli/brio.py``) —
no gateway required, though the local API server must be reachable.

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import contextlib
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import quote, urlsplit, urlunsplit

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - aiohttp is a core dependency
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (ported from tunnel.go)
# ---------------------------------------------------------------------------

MAX_INFLIGHT_REQUESTS = 16          # COMPANION_BUSY cap (buffered channel of 16)
MAX_RESPONSE_BYTES = 10 * 1024 * 1024   # 10 MiB response / stream cap
WS_MAX_MSG_SIZE = 12 * 1024 * 1024      # 12 MiB websocket receive limit
STREAM_CHUNK_SIZE = 32 * 1024
PING_INTERVAL_SECONDS = 25.0
PING_TIMEOUT_SECONDS = 10.0
FRAME_WRITE_TIMEOUT_SECONDS = 15.0
LOCAL_REQUEST_TIMEOUT_SECONDS = 300.0   # Go http.Client{Timeout: 5 * time.Minute}
BACKOFF_INITIAL_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0               # doubling stops at/after 30 → sleeps cap at 32s
BACKOFF_RESET_AFTER_SECONDS = 30.0       # a connection that lasted >30s resets backoff
BACKOFF_JITTER_RATIO = 0.1

# ---------------------------------------------------------------------------
# Path compatibility mapping (companion-era mobile paths → API server paths)
# ---------------------------------------------------------------------------

_EXACT_PATH_MAP = {
    "/chat/responses": "/v1/responses",
    "/sessions": "/v1/sessions",
    "/capabilities": "/v1/capabilities",
    "/memory": "/v1/memory",
    "/health": "/health",
}

_SESSION_MESSAGES_RE = re.compile(r"^/sessions/[^/]+/messages$")

# Prefixes that pass through to the local API server unchanged.
_PASSTHROUGH_PREFIXES = ("/v1/runs", "/api/jobs")


def map_request_path(path: str) -> Optional[str]:
    """Map a companion-era request path to a local API server path.

    Query strings are preserved.  Returns ``None`` for unmapped paths (the
    caller replies with a 404 JSON error, like the companion's router did).
    """
    if not path:
        return None
    base, sep, query = path.partition("?")
    if base in _EXACT_PATH_MAP:
        mapped = _EXACT_PATH_MAP[base]
    elif _SESSION_MESSAGES_RE.match(base):
        mapped = "/v1" + base
    elif base.startswith(_PASSTHROUGH_PREFIXES):
        mapped = base
    else:
        return None
    return mapped + sep + query if query else mapped


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def final_sse_json_body(text: str) -> Any:
    """Best-effort final JSON body from an accumulated SSE stream.

    Scans ``data:`` payloads and returns the last one that parses as JSON;
    ``response.completed`` envelopes are unwrapped to the inner response
    object so the value matches what the Brio mobile app's SSE parser
    (``ResponsesSSEParser.finish()``) would return.  Returns ``None`` when
    no payload parses.  Multi-line ``data:`` fields (rare on OpenAI-wire
    endpoints) are treated line-by-line; the app falls back to joining
    ``stream_chunk`` data when the body is unusable.
    """
    final: Any = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            final = json.loads(payload)
        except ValueError:
            continue
    if isinstance(final, dict) and final.get("type") == "response.completed":
        inner = final.get("response")
        if isinstance(inner, dict):
            return inner
    return final


def error_frame(frame_id: str, code: str, message: str) -> Dict[str, Any]:
    """Build a tunnel ``error`` frame (mirrors tunnel.go errorFrame)."""
    return {"type": "error", "id": frame_id, "code": code, "message": message}


def response_frame(frame_id: str, status: int, content_type: str, body: Any) -> Dict[str, Any]:
    """Build a tunnel ``response`` frame (mirrors tunnel.go Frame marshal)."""
    return {
        "type": "response",
        "id": frame_id,
        "status": status,
        "headers": {"Content-Type": content_type},
        "body": body,
    }


def tunnel_url(relay_url: str, role: str, agent_id: str, token: str = "") -> str:
    """Build the WebSocket tunnel URL from a base relay URL (tunnel.go tunnelURL).

    http→ws, https→wss; ws/wss pass through; anything else is rejected.
    """
    relay_url = (relay_url or "").strip().rstrip("/")
    parts = urlsplit(relay_url)
    scheme = parts.scheme
    if scheme == "http":
        scheme = "ws"
    elif scheme == "https":
        scheme = "wss"
    elif scheme not in ("ws", "wss"):
        raise ValueError(f"unsupported relay URL scheme: {parts.scheme or '(none)'}")
    path = parts.path.rstrip("/") + f"/tunnel/{role}/{quote(agent_id, safe='')}"
    query = f"token={quote(token, safe='')}" if token else ""
    return urlunsplit((scheme, parts.netloc, path, query, ""))


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class BrioConnector:
    """Foregroud tunnel client connecting Hermes to the Brio relay.

    ``run()`` blocks (reconnecting with backoff) until ``stop()`` is called
    or the task is cancelled.  Each inbound ``request`` frame is proxied to
    ``api_base_url`` with the API server bearer token injected.
    """

    def __init__(
        self,
        relay_url: str,
        agent_id: str,
        relay_token: str,
        api_base_url: str = "http://127.0.0.1:8642",
        api_key: str = "",
        max_inflight_requests: int = MAX_INFLIGHT_REQUESTS,
    ):
        self.relay_url = relay_url.rstrip("/") if relay_url else ""
        self.agent_id = agent_id or ""
        self.relay_token = relay_token or ""
        self.api_base_url = (api_base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self._max_inflight = max_inflight_requests
        self._stop_event: Optional[asyncio.Event] = None
        self._ws: Optional["aiohttp.ClientWebSocketResponse"] = None
        self._session: Optional["aiohttp.ClientSession"] = None
        self._pong_received: Optional[asyncio.Event] = None
        self._inflight = 0
        # Exposed for status reporting: monotonic timestamp of the last
        # successful tunnel connection, or None.
        self.last_connected_at: Optional[float] = None

    # -- lifecycle --------------------------------------------------------

    def stop(self) -> None:
        """Request a clean exit from ``run()`` (call from the same loop)."""
        if self._stop_event is not None:
            self._stop_event.set()
        ws = self._ws
        if ws is not None and not ws.closed:
            # Wake the blocked read loop; run() exits on the stop event.
            try:
                asyncio.get_running_loop().create_task(
                    ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"bye")
                )
            except RuntimeError:
                pass  # no running loop (e.g. called during teardown)

    async def run(self, http_session: Optional["aiohttp.ClientSession"] = None) -> None:
        """Reconnect loop with exponential backoff (port of tunnel.Run)."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for the Brio connector")
        if not self.relay_url or not self.agent_id:
            logger.warning("brio connector not configured (relay URL / agent id missing)")
            return

        self._stop_event = asyncio.Event()
        own_session = http_session is None
        if own_session:
            # No session-level total timeout — the WebSocket is long-lived.
            # Per-request timeouts are applied in _local_request (Go:
            # http.Client{Timeout: 5 * time.Minute}).
            http_session = aiohttp.ClientSession()
        assert http_session is not None
        try:
            backoff = BACKOFF_INITIAL_SECONDS
            while not self._stop_event.is_set():
                started = time.monotonic()
                try:
                    await self._run_connection(http_session)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("brio relay tunnel disconnected: %s", exc)
                if time.monotonic() - started > BACKOFF_RESET_AFTER_SECONDS:
                    backoff = BACKOFF_INITIAL_SECONDS
                jitter = backoff * BACKOFF_JITTER_RATIO * random.random()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff + jitter)
                except asyncio.TimeoutError:
                    pass
                if backoff < BACKOFF_CAP_SECONDS:
                    backoff *= 2
        finally:
            if own_session:
                await http_session.close()

    async def _run_connection(self, http_session: "aiohttp.ClientSession") -> None:
        """One WebSocket connection: connect, serve frames until it drops."""
        url = tunnel_url(self.relay_url, "companion", self.agent_id, self.relay_token)
        ws = await http_session.ws_connect(
            url,
            # We send our own pings; disable aiohttp's heartbeat.
            heartbeat=None,
            # autoping=False so PONG frames surface in receive() (with the
            # default they are swallowed and our pong timeout could never
            # fire); PING frames are answered manually in the read loop.
            autoping=False,
            # Mirror the relay/companion 12 MiB read limit.
            max_msg_size=WS_MAX_MSG_SIZE,
        )
        self._ws = ws
        self._session = http_session
        self._inflight = 0
        self._pong_received = asyncio.Event()
        self.last_connected_at = time.monotonic()
        ping_task = asyncio.create_task(self._ping_loop(ws))
        logger.info(
            "brio relay tunnel connected (agent_id=%s relay=%s local=%s)",
            self.agent_id, self.relay_url, self.api_base_url,
        )
        try:
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data)
                elif msg.type == aiohttp.WSMsgType.PING:
                    await ws.pong(msg.data)
                elif msg.type == aiohttp.WSMsgType.PONG:
                    if self._pong_received is not None:
                        self._pong_received.set()
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise ws.exception() or RuntimeError("websocket error")
        finally:
            ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ping_task
            if not ws.closed:
                await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"bye")
            self._ws = None
            self._session = None
            logger.info("brio relay tunnel connection closed (agent_id=%s)", self.agent_id)

    # -- ping/pong --------------------------------------------------------

    async def _ping_loop(self, ws: "aiohttp.ClientWebSocketResponse") -> None:
        """Send ws pings every 25s; close the socket when a pong times out."""
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            if self._pong_received is not None:
                self._pong_received.clear()
            try:
                await ws.ping()
                if self._pong_received is not None:
                    # wait_for cancels at the timeout; the event is set by the
                    # receive loop when the pong arrives.
                    await asyncio.wait_for(
                        self._pong_received.wait(), timeout=PING_TIMEOUT_SECONDS,
                    )
            except Exception as exc:
                logger.warning("brio relay tunnel ping failed: %s", exc)
                if not ws.closed:
                    await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"ping failed")
                return

    # -- frame dispatch ----------------------------------------------------

    async def _handle_message(self, ws: "aiohttp.ClientWebSocketResponse", raw: str) -> None:
        try:
            frame = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return  # mirror tunnel.go: unparseable frames are skipped
        if not isinstance(frame, dict):
            return
        frame_type = frame.get("type")
        if frame_type == "request":
            await self._dispatch_request(ws, frame)
        elif frame_type == "ping":
            # The relay hub answers pings itself; handle direct pings anyway.
            await self._send_frame(ws, {"type": "pong", "id": str(frame.get("id") or "")})
        # Other frame types (pong, responses) are ignored, like tunnel.go.

    async def _dispatch_request(self, ws: "aiohttp.ClientWebSocketResponse", frame: Dict[str, Any]) -> None:
        frame_id = str(frame.get("id") or "")
        if self._inflight >= self._max_inflight:
            await self._send_frame(
                ws, error_frame(frame_id, "COMPANION_BUSY", "too many requests are in progress"),
            )
            return
        self._inflight += 1
        # The slot is released in _serve_request's finally — mirroring the Go
        # goroutine's ``defer func() { <-requests }()``.
        asyncio.create_task(self._serve_request(ws, frame, frame_id))

    def _release_inflight(self) -> None:
        self._inflight = max(0, self._inflight - 1)

    async def _serve_request(
        self, ws: "aiohttp.ClientWebSocketResponse", frame: Dict[str, Any], frame_id: str,
    ) -> None:
        try:
            await self._proxy_local_frames(ws, frame, frame_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Mirror tunnel.go: on emit failure the connection is closed with
            # an internal-error close code and the outer loop reconnects.
            logger.exception("brio tunnel request failed (frame_id=%s)", frame_id)
            if not ws.closed:
                await ws.close(code=aiohttp.WSCloseCode.INTERNAL_ERROR, message=b"could not write response")
        finally:
            self._release_inflight()

    # -- local proxying ----------------------------------------------------

    async def _proxy_local_frames(
        self, ws: "aiohttp.ClientWebSocketResponse", frame: Dict[str, Any], frame_id: str,
    ) -> None:
        """Port of tunnel.go proxyLocalFrames: request → response/stream frames."""
        method = frame.get("method") or "GET"
        path = frame.get("path") or ""
        if not path or not path.startswith("/"):
            await self._send_frame(
                ws, error_frame(frame_id, "BAD_REQUEST", "request path must start with /"),
            )
            return

        mapped = map_request_path(path)
        if mapped is None:
            await self._send_frame(
                ws,
                response_frame(frame_id, 404, "application/json", {"error": f"unsupported path: {path}"}),
            )
            return

        body = frame.get("body")
        headers: Dict[str, str] = {}
        json_body: Optional[str] = None
        if body is not None:
            try:
                json_body = json.dumps(body)
            except (TypeError, ValueError) as exc:
                await self._send_frame(ws, error_frame(frame_id, "BAD_REQUEST", str(exc)))
                return
            headers["Content-Type"] = "application/json"
        # Always replace the inbound Authorization header with the local API
        # server key — relay credentials never reach the local API server.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.api_base_url + mapped
        try:
            resp = await self._local_request(ws, method, url, headers, json_body, frame_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._send_frame(ws, error_frame(frame_id, "LOCAL_UNREACHABLE", str(exc)))
            return

        try:
            content_type = resp.headers.get("Content-Type", "")
            is_event_stream = (
                resp.status < 400 and "text/event-stream" in content_type.lower()
            )
            if is_event_stream:
                await self._proxy_event_stream(ws, frame_id, resp)
                return

            data = bytearray()
            async for chunk in resp.content.iter_chunked(STREAM_CHUNK_SIZE):
                data.extend(chunk)
                if len(data) > MAX_RESPONSE_BYTES:
                    await self._send_frame(
                        ws,
                        error_frame(frame_id, "RESPONSE_TOO_LARGE", "local response is larger than 10 MiB"),
                    )
                    return
            body_value: Any = None
            if len(data) > 0 and "json" in content_type.lower():
                try:
                    body_value = json.loads(bytes(data))
                except ValueError:
                    body_value = None
            if body_value is None:
                body_value = bytes(data).decode("utf-8", errors="replace")
            await self._send_frame(
                ws, response_frame(frame_id, resp.status, content_type, body_value),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._send_frame(ws, error_frame(frame_id, "LOCAL_READ_FAILED", str(exc)))
        finally:
            resp.close()

    async def _local_request(
        self,
        ws: "aiohttp.ClientWebSocketResponse",
        method: str,
        url: str,
        headers: Dict[str, str],
        json_body: Optional[str],
        frame_id: str,
    ) -> "aiohttp.ClientResponse":
        """Issue the local API server request through the connection's session."""
        session = self._session
        if session is None:
            raise RuntimeError("no active HTTP session for brio connector")
        return await session.request(
            method,
            url,
            headers=headers,
            data=json_body.encode("utf-8") if json_body is not None else None,
            timeout=aiohttp.ClientTimeout(total=LOCAL_REQUEST_TIMEOUT_SECONDS),
        )

    async def _proxy_event_stream(
        self, ws: "aiohttp.ClientWebSocketResponse", frame_id: str, resp: "aiohttp.ClientResponse",
    ) -> None:
        """Port of tunnel.go proxyEventStream: chunk raw SSE bytes, then end.

        Beyond the Go original, the ``stream_end`` frame carries the final
        parsed JSON body (the last valid ``data:`` payload) when available,
        else ``null`` — the mobile app resolves it directly as the terminal
        response instead of re-parsing the chunk stream.
        """
        total = 0
        content_type = resp.headers.get("Content-Type", "")
        parts: list[str] = []
        async for chunk in resp.content.iter_chunked(STREAM_CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                await self._send_frame(
                    ws,
                    error_frame(frame_id, "RESPONSE_TOO_LARGE", "local response stream is larger than 10 MiB"),
                )
                return
            text = chunk.decode("utf-8", errors="replace")
            parts.append(text)
            await self._send_frame(ws, {"type": "stream_chunk", "id": frame_id, "data": text})
        await self._send_frame(ws, {
            "type": "stream_end",
            "id": frame_id,
            "status": resp.status,
            "headers": {"Content-Type": content_type},
            "body": final_sse_json_body("".join(parts)),
        })

    async def _send_frame(self, ws: "aiohttp.ClientWebSocketResponse", frame: Dict[str, Any]) -> None:
        """Serialize and send one frame with a 15s write timeout (writeFrame)."""
        await asyncio.wait_for(
            ws.send_str(json.dumps(frame)),
            timeout=FRAME_WRITE_TIMEOUT_SECONDS,
        )


# ---------------------------------------------------------------------------
# Gateway platform adapter
# ---------------------------------------------------------------------------


def check_brio_requirements() -> bool:
    """Whether the connector's dependencies are importable (adapter pattern)."""
    return AIOHTTP_AVAILABLE


class BrioAdapter(BasePlatformAdapter):
    """Gateway adapter that runs the Brio connector alongside the gateway.

    Enabled automatically by ``gateway/config.py`` when BRIO_RELAY_URL,
    BRIO_RELAY_TOKEN and BRIO_AGENT_ID are set (written by
    ``hermes brio enroll``), so ``hermes gateway`` keeps the relay tunnel up
    without a separate supervised process.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.BRIO)
        extra = config.extra or {}
        self._relay_url: str = str(extra.get("relay_url") or os.getenv("BRIO_RELAY_URL", ""))
        self._agent_id: str = str(extra.get("agent_id") or os.getenv("BRIO_AGENT_ID", ""))
        self._relay_token: str = str(extra.get("relay_token") or os.getenv("BRIO_RELAY_TOKEN", ""))
        raw_port = extra.get("api_server_port") or os.getenv("API_SERVER_PORT", "8642")
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            port = 8642
        self._api_port: int = port
        self._api_key: str = str(extra.get("api_server_key") or os.getenv("API_SERVER_KEY", ""))
        self._connector: Optional[BrioConnector] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "Brio"

    async def connect(self) -> bool:
        if not check_brio_requirements():
            logger.warning("[%s] aiohttp not installed", self.name)
            return False
        if not (self._relay_url and self._agent_id and self._relay_token):
            logger.warning(
                "[%s] BRIO_RELAY_URL / BRIO_RELAY_TOKEN / BRIO_AGENT_ID not set; "
                "run `hermes brio enroll` and restart the gateway",
                self.name,
            )
            return False
        self._connector = BrioConnector(
            relay_url=self._relay_url,
            agent_id=self._agent_id,
            relay_token=self._relay_token,
            api_base_url=f"http://127.0.0.1:{self._api_port}",
            api_key=self._api_key,
        )
        self._task = asyncio.create_task(
            self._connector.run(), name=f"{self.name.lower()}-connector"
        )
        self._mark_connected()
        logger.info(
            "[%s] relay tunnel started (agent=%s relay=%s)",
            self.name,
            self._agent_id,
            self._relay_url,
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        connector, task = self._connector, self._task
        self._connector, self._task = None, None
        if connector is not None:
            connector.stop()
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("[%s] connector task ended with %s", self.name, exc)
        logger.info("[%s] relay tunnel stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Not used — the tunnel is request/response, not message delivery."""
        return SendResult(success=False, error="Brio tunnel uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "name": "Brio",
            "type": "tunnel",
            "agent_id": self._agent_id,
            "relay_url": self._relay_url,
        }
