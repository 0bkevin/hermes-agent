"""
Tests for the Brio relay tunnel connector (gateway/platforms/brio_connector.py)
and the ``hermes brio`` enrollment CLI (hermes_cli/brio.py).

An in-process aiohttp "mock relay" implements the relay's hub semantics from
brio's apps/relay/internal/server/server.go: it routes ``request`` frames from
a "mobile" peer to the connected companion peer and delivers
response/stream/error frames back to the requester.

Covered:
- tunnel URL building and companion-era path mapping (pure unit tests)
- enroll persists BRIO_* env keys + enables the API server
- request → response round trip through the connector into a local API mock
- SSE chunk streaming + stream_end
- path mapping (incl. query strings) and unmapped-path 404 responses
- Authorization header injection (relay credentials never forwarded)
- LOCAL_UNREACHABLE / COMPANION_BUSY error frames
- JSON ping → pong
- reconnect after the relay drops the companion connection
"""

import asyncio
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.brio_connector import (
    BrioAdapter,
    BrioConnector,
    map_request_path,
    tunnel_url,
)

# ---------------------------------------------------------------------------
# Mock relay (hub semantics ported from apps/relay/internal/server/server.go)
# ---------------------------------------------------------------------------


class MockRelayHub:
    """Minimal Brio relay hub for tests."""

    def __init__(self):
        self.agent_tokens = {}        # agent_id -> companion token
        self.companions = {}          # agent_id -> companion websocket
        self.pending = {}             # frame_id -> mobile websocket
        self.companion_connections = 0
        self.drop_first_companion = False
        self.claimed_codes = {}       # code -> agent payload

    # -- HTTP handlers ----------------------------------------------------

    async def handle_health(self, request):
        return web.json_response({
            "service": "brio-relay",
            "ok": True,
            "agents": len(self.companions),
            "peers": len(self.companions),
            "pending_requests": len(self.pending),
        })

    async def handle_claim(self, request):
        code = request.match_info["code"].upper()
        body = await request.json()
        agent_id = (body.get("agent_id") or "").strip()
        if not agent_id:
            return web.json_response({"error": "agent_id is required"}, status=400)
        if code not in self.claimed_codes:
            return web.json_response({"error": "enrollment not found"}, status=404)
        relay_token = "brio_agent_test_" + agent_id
        self.agent_tokens[agent_id] = relay_token
        return web.json_response({
            "agent": {"id": agent_id, "name": body.get("name") or "Hermes"},
            "relay_token": relay_token,
        }, status=201)

    async def handle_tunnel(self, request):
        role = request.match_info["role"]
        agent_id = request.match_info["agent_id"]
        token = request.query.get("token", "")
        if role not in ("mobile", "companion"):
            return web.json_response({"error": "role must be mobile or companion"}, status=400)
        if role == "companion":
            if not token:
                return web.json_response({"error": "missing companion token"}, status=401)
            if self.agent_tokens.get(agent_id) != token:
                return web.json_response({"error": "invalid companion token"}, status=401)
        else:
            if not token:
                return web.json_response({"error": "missing device token"}, status=401)

        ws = web.WebSocketResponse(max_msg_size=12 * 1024 * 1024)
        await ws.prepare(request)

        if role == "companion":
            self.companions[agent_id] = ws
            self.companion_connections += 1
            if self.drop_first_companion and self.companion_connections == 1:
                await ws.close(code=1001, message=b"simulated relay restart")

        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                try:
                    frame = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                await self._route(role, agent_id, ws, frame, msg.data)
        finally:
            if role == "companion" and self.companions.get(agent_id) is ws:
                self.companions.pop(agent_id, None)
                for fid, mobile in list(self.pending.items()):
                    await mobile.send_str(json.dumps({
                        "type": "error", "id": fid,
                        "code": "COMPANION_DISCONNECTED",
                        "message": "companion disconnected before responding",
                    }))
                self.pending.clear()
        return ws

    async def _route(self, role, agent_id, sender, frame, raw):
        frame_type = frame.get("type")
        if frame_type == "ping":
            await sender.send_str(json.dumps({"type": "pong", "id": frame.get("id", "")}))
            return
        if frame_type == "pong":
            return
        if role == "mobile":
            if frame_type != "request":
                await sender.send_str(json.dumps({
                    "type": "error", "id": frame.get("id", ""),
                    "code": "BAD_FRAME",
                    "message": "mobile clients may only send request frames",
                }))
                return
            companion = self.companions.get(agent_id)
            if companion is None:
                await sender.send_str(json.dumps({
                    "type": "error", "id": frame.get("id", ""),
                    "code": "AGENT_OFFLINE",
                    "message": "no companion is connected for this agent",
                }))
                return
            self.pending[frame["id"]] = sender
            await companion.send_str(raw)
        elif role == "companion":
            if frame_type not in ("response", "stream_chunk", "stream_end", "error"):
                return
            mobile = self.pending.get(frame.get("id"))
            if mobile is not None:
                await mobile.send_str(raw)
                if frame_type in ("response", "stream_end", "error"):
                    self.pending.pop(frame.get("id"), None)

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self.handle_health)
        app.router.add_post("/enrollments/{code}/claim", self.handle_claim)
        app.router.add_get("/tunnel/{role}/{agent_id}", self.handle_tunnel)
        return app


# ---------------------------------------------------------------------------
# Local API server mock (what gateway/platforms/api_server.py would serve)
# ---------------------------------------------------------------------------


class LocalAPIMock:
    """Records proxied requests; serves JSON or SSE on /v1/responses."""

    def __init__(self, delay=0.0):
        self.requests = []
        self.delay = delay

    async def _handle(self, request):
        body = None
        raw = await request.read()
        if raw:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw.decode("utf-8", errors="replace")
        self.requests.append({
            "method": request.method,
            "path": request.path,
            "query": dict(request.query),
            "headers": dict(request.headers),
            "body": body,
        })
        if self.delay:
            await asyncio.sleep(self.delay)

        if request.path == "/v1/responses" and isinstance(body, dict) and body.get("stream"):
            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for event in ("data: {\"delta\":\"Hello\"}\n\n", "data: {\"delta\":\" world\"}\n\n", "data: [DONE]\n\n"):
                await resp.write(event.encode("utf-8"))
                await asyncio.sleep(0.01)
            await resp.write_eof()
            return resp

        if request.path == "/v1/sessions":
            return web.json_response({"sessions": [{"id": "s1", "title": "t"}]})
        if request.path == "/v1/responses":
            return web.json_response({"id": "resp_1", "output": "hi"}, status=200)
        if request.path == "/v1/capabilities":
            return web.json_response({"capabilities": True})
        if request.path == "/health":
            return web.json_response({"status": "ok"})
        return web.json_response({"error": f"not found: {request.path}"}, status=404)

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        return app


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------


class TestTunnelURL:
    def test_http_becomes_ws(self):
        assert tunnel_url("http://relay.example/", "companion", "a1", "tok") == \
            "ws://relay.example/tunnel/companion/a1?token=tok"

    def test_https_becomes_wss_with_trailing_slash_trim(self):
        assert tunnel_url("https://relay.example//", "mobile", "a1", "t") == \
            "wss://relay.example/tunnel/mobile/a1?token=t"

    def test_ws_passthrough_and_no_token(self):
        assert tunnel_url("ws://r:8082", "companion", "a b", "") == \
            "ws://r:8082/tunnel/companion/a%20b"

    def test_unsupported_scheme_raises(self):
        with pytest.raises(ValueError):
            tunnel_url("ftp://relay.example", "companion", "a1")


class TestPathMapping:
    def test_exact_mappings(self):
        assert map_request_path("/chat/responses") == "/v1/responses"
        assert map_request_path("/sessions") == "/v1/sessions"
        assert map_request_path("/capabilities") == "/v1/capabilities"
        assert map_request_path("/memory") == "/v1/memory"
        assert map_request_path("/health") == "/health"

    def test_session_messages_with_id(self):
        assert map_request_path("/sessions/abc-123/messages") == "/v1/sessions/abc-123/messages"

    def test_query_strings_preserved(self):
        assert map_request_path("/sessions?limit=50") == "/v1/sessions?limit=50"
        assert map_request_path("/sessions/abc/messages?x=1&y=2") == "/v1/sessions/abc/messages?x=1&y=2"

    def test_runs_and_jobs_passthrough(self):
        assert map_request_path("/v1/runs") == "/v1/runs"
        assert map_request_path("/v1/runs/r-1/events") == "/v1/runs/r-1/events"
        assert map_request_path("/api/jobs") == "/api/jobs"
        assert map_request_path("/api/jobs/abc123/pause") == "/api/jobs/abc123/pause"

    def test_unmapped_paths_return_none(self):
        assert map_request_path("/files") is None
        assert map_request_path("/sessions/search") is None
        assert map_request_path("/config/raw") is None
        assert map_request_path("") is None
        assert map_request_path("/chat/other") is None


class TestFinalSSEJsonBody:
    def test_last_valid_data_payload_wins(self):
        from gateway.platforms.brio_connector import final_sse_json_body
        stream = (
            'event: response.delta\ndata: {"type":"response.delta"}\n\n'
            'event: response.completed\ndata: {"type":"response.completed","output":[]}\n\n'
            "data: [DONE]\n\n"
        )
        assert final_sse_json_body(stream) == {"type": "response.completed", "output": []}

    def test_response_completed_envelope_unwrapped(self):
        """The mobile app's parser returns event.response — match that shape."""
        from gateway.platforms.brio_connector import final_sse_json_body
        stream = (
            'event: response.completed\ndata: {"type":"response.completed",'
            '"response":{"id":"resp_1","status":"completed","output":[]}}\n\n'
            "data: [DONE]\n\n"
        )
        assert final_sse_json_body(stream) == {"id": "resp_1", "status": "completed", "output": []}

    def test_no_json_payloads_returns_none(self):
        from gateway.platforms.brio_connector import final_sse_json_body
        assert final_sse_json_body("data: [DONE]\n\n: keepalive\n\n") is None
        assert final_sse_json_body("") is None


# ---------------------------------------------------------------------------
# End-to-end connector tests (in-loop mock relay + local API mock)
# ---------------------------------------------------------------------------

AGENT_ID = "hermes_test_agent"
RELAY_TOKEN = "brio_agent_test_" + AGENT_ID
DEVICE_TOKEN = "device-token-1"


@pytest.fixture
def hub():
    hub = MockRelayHub()
    hub.agent_tokens[AGENT_ID] = RELAY_TOKEN
    return hub


async def _wait_for(predicate, timeout=5.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {what}")


class TestConnectorTunnel:
    @pytest.mark.asyncio
    async def test_request_response_round_trip(self, hub):
        local = LocalAPIMock()
        relay_server = TestServer(hub.make_app())
        local_server = TestServer(local.make_app())
        async with relay_server as relay, local_server as api, TestClient(relay) as relay_cli:
            hub_url = str(relay.make_url("/"))
            connector = BrioConnector(
                relay_url=hub_url,
                agent_id=AGENT_ID,
                relay_token=RELAY_TOKEN,
                api_base_url=str(api.make_url("/")),
                api_key="local-key",
            )
            task = asyncio.create_task(connector.run())
            try:
                await _wait_for(lambda: AGENT_ID in hub.companions, what="companion connect")

                mobile = await relay_cli.ws_connect(
                    f"/tunnel/mobile/{AGENT_ID}?token={DEVICE_TOKEN}",
                )
                await mobile.send_json({
                    "type": "request", "id": "req-1", "method": "GET",
                    "path": "/sessions?limit=5",
                    "headers": {"Authorization": "Bearer device-side-token"},
                })
                frame = json.loads((await mobile.receive()).data)
                assert frame["type"] == "response"
                assert frame["id"] == "req-1"
                assert frame["status"] == 200
                assert frame["body"] == {"sessions": [{"id": "s1", "title": "t"}]}

                # The request hit the mapped local path with the injected key.
                assert len(local.requests) == 1
                assert local.requests[0]["path"] == "/v1/sessions"
                assert local.requests[0]["query"] == {"limit": "5"}
                assert local.requests[0]["headers"]["Authorization"] == "Bearer local-key"
                await mobile.close()
            finally:
                connector.stop()
                await asyncio.wait_for(task, timeout=5)

    @pytest.mark.asyncio
    async def test_sse_stream_chunked_back(self, hub):
        local = LocalAPIMock()
        relay_server = TestServer(hub.make_app())
        local_server = TestServer(local.make_app())
        async with relay_server as relay, local_server as api, TestClient(relay) as relay_cli:
            connector = BrioConnector(
                relay_url=str(relay.make_url("/")),
                agent_id=AGENT_ID,
                relay_token=RELAY_TOKEN,
                api_base_url=str(api.make_url("/")),
                api_key="local-key",
            )
            task = asyncio.create_task(connector.run())
            try:
                await _wait_for(lambda: AGENT_ID in hub.companions)
                mobile = await relay_cli.ws_connect(f"/tunnel/mobile/{AGENT_ID}?token={DEVICE_TOKEN}")
                await mobile.send_json({
                    "type": "request", "id": "req-sse", "method": "POST",
                    "path": "/chat/responses",
                    "body": {"stream": True, "input": "hello"},
                })
                chunks = []
                end_frame = None
                for _ in range(10):
                    frame = json.loads((await mobile.receive()).data)
                    if frame["type"] == "stream_chunk":
                        chunks.append(frame["data"])
                    elif frame["type"] == "stream_end":
                        end_frame = frame
                        break
                    else:
                        pytest.fail(f"unexpected frame: {frame}")
                assert chunks == [
                    "data: {\"delta\":\"Hello\"}\n\n",
                    "data: {\"delta\":\" world\"}\n\n",
                    "data: [DONE]\n\n",
                ]
                assert end_frame["id"] == "req-sse"
                assert end_frame["status"] == 200
                assert "text/event-stream" in end_frame["headers"]["Content-Type"]
                # stream_end carries the final parsed JSON body (last valid
                # data: payload) — [DONE] markers are skipped.
                assert end_frame["body"] == {"delta": " world"}
                # Companion-era path was mapped before proxying.
                assert local.requests[-1]["path"] == "/v1/responses"
                await mobile.close()
            finally:
                connector.stop()
                await asyncio.wait_for(task, timeout=5)

    @pytest.mark.asyncio
    async def test_unmapped_path_returns_404_response(self, hub):
        local = LocalAPIMock()
        relay_server = TestServer(hub.make_app())
        local_server = TestServer(local.make_app())
        async with relay_server as relay, local_server as api, TestClient(relay) as relay_cli:
            connector = BrioConnector(
                relay_url=str(relay.make_url("/")),
                agent_id=AGENT_ID,
                relay_token=RELAY_TOKEN,
                api_base_url=str(api.make_url("/")),
            )
            task = asyncio.create_task(connector.run())
            try:
                await _wait_for(lambda: AGENT_ID in hub.companions)
                mobile = await relay_cli.ws_connect(f"/tunnel/mobile/{AGENT_ID}?token={DEVICE_TOKEN}")
                await mobile.send_json({
                    "type": "request", "id": "req-404", "method": "GET",
                    "path": "/files",
                })
                frame = json.loads((await mobile.receive()).data)
                assert frame["type"] == "response"
                assert frame["status"] == 404
                assert "error" in frame["body"]
                assert local.requests == []  # never proxied
                await mobile.close()
            finally:
                connector.stop()
                await asyncio.wait_for(task, timeout=5)

    @pytest.mark.asyncio
    async def test_local_unreachable_sends_error_frame(self, hub):
        relay_server = TestServer(hub.make_app())
        async with relay_server as relay, TestClient(relay) as relay_cli:
            # Point the connector at a closed port.
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            dead_port = sock.getsockname()[1]
            sock.close()
            connector = BrioConnector(
                relay_url=str(relay.make_url("/")),
                agent_id=AGENT_ID,
                relay_token=RELAY_TOKEN,
                api_base_url=f"http://127.0.0.1:{dead_port}",
            )
            task = asyncio.create_task(connector.run())
            try:
                await _wait_for(lambda: AGENT_ID in hub.companions)
                mobile = await relay_cli.ws_connect(f"/tunnel/mobile/{AGENT_ID}?token={DEVICE_TOKEN}")
                await mobile.send_json({
                    "type": "request", "id": "req-err", "method": "GET",
                    "path": "/health",
                })
                frame = json.loads((await mobile.receive()).data)
                assert frame["type"] == "error"
                assert frame["id"] == "req-err"
                assert frame["code"] == "LOCAL_UNREACHABLE"
                await mobile.close()
            finally:
                connector.stop()
                await asyncio.wait_for(task, timeout=5)

    @pytest.mark.asyncio
    async def test_busy_cap_returns_companion_busy(self, hub):
        local = LocalAPIMock(delay=0.3)
        relay_server = TestServer(hub.make_app())
        local_server = TestServer(local.make_app())
        async with relay_server as relay, local_server as api, TestClient(relay) as relay_cli:
            connector = BrioConnector(
                relay_url=str(relay.make_url("/")),
                agent_id=AGENT_ID,
                relay_token=RELAY_TOKEN,
                api_base_url=str(api.make_url("/")),
                max_inflight_requests=1,
            )
            task = asyncio.create_task(connector.run())
            try:
                await _wait_for(lambda: AGENT_ID in hub.companions)
                mobile = await relay_cli.ws_connect(f"/tunnel/mobile/{AGENT_ID}?token={DEVICE_TOKEN}")
                for i in range(2):
                    await mobile.send_json({
                        "type": "request", "id": f"req-busy-{i}", "method": "GET",
                        "path": "/capabilities",
                    })
                frames = {}
                for _ in range(2):
                    frame = json.loads((await mobile.receive()).data)
                    frames[frame["id"]] = frame
                by_code = {}
                for f in frames.values():
                    by_code.setdefault(f["type"], []).append(f)
                assert len(by_code["response"]) == 1
                busy = by_code["error"]
                assert len(busy) == 1
                assert busy[0]["code"] == "COMPANION_BUSY"
                assert busy[0]["message"] == "too many requests are in progress"
                await mobile.close()
            finally:
                connector.stop()
                await asyncio.wait_for(task, timeout=5)

    @pytest.mark.asyncio
    async def test_json_ping_gets_pong_from_connector(self):
        """Direct unit check: the connector answers inbound ping frames itself."""
        class FakeWS:
            def __init__(self):
                self.sent = []

            async def send_str(self, data):
                self.sent.append(json.loads(data))

        ws = FakeWS()
        connector = BrioConnector("http://relay", "a", "t")
        await connector._handle_message(ws, json.dumps({"type": "ping", "id": "p1"}))
        assert ws.sent == [{"type": "pong", "id": "p1"}]

    @pytest.mark.asyncio
    async def test_reconnect_after_relay_drop(self, hub):
        hub.drop_first_companion = True
        local = LocalAPIMock()
        relay_server = TestServer(hub.make_app())
        local_server = TestServer(local.make_app())
        async with relay_server as relay, local_server as api, TestClient(relay) as relay_cli:
            connector = BrioConnector(
                relay_url=str(relay.make_url("/")),
                agent_id=AGENT_ID,
                relay_token=RELAY_TOKEN,
                api_base_url=str(api.make_url("/")),
            )
            task = asyncio.create_task(connector.run())
            try:
                # First connection is dropped immediately; the connector must
                # reconnect with backoff (1s + jitter).
                await _wait_for(
                    lambda: hub.companion_connections >= 2, timeout=10,
                    what="reconnect",
                )
                await _wait_for(lambda: AGENT_ID in hub.companions)

                mobile = await relay_cli.ws_connect(f"/tunnel/mobile/{AGENT_ID}?token={DEVICE_TOKEN}")
                await mobile.send_json({
                    "type": "request", "id": "req-after", "method": "GET",
                    "path": "/sessions",
                })
                frame = json.loads((await mobile.receive()).data)
                assert frame["type"] == "response"
                assert frame["status"] == 200
                await mobile.close()
            finally:
                connector.stop()
                await asyncio.wait_for(task, timeout=5)

    @pytest.mark.asyncio
    async def test_bad_relay_token_rejected(self, hub):
        relay_server = TestServer(hub.make_app())
        async with relay_server as relay:
            connector = BrioConnector(
                relay_url=str(relay.make_url("/")),
                agent_id=AGENT_ID,
                relay_token="wrong-token",
            )
            task = asyncio.create_task(connector.run())
            try:
                # The relay rejects auth; the connector keeps retrying but
                # never registers a companion.
                await asyncio.sleep(0.5)
                assert AGENT_ID not in hub.companions
                assert not task.done()
            finally:
                connector.stop()
                await asyncio.wait_for(task, timeout=5)


# ---------------------------------------------------------------------------
# Enroll CLI (threaded mock relay — brio_enroll calls asyncio.run internally)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ThreadedRelay:
    """Runs the mock relay in a background thread with its own event loop."""

    def __init__(self, hub: MockRelayHub):
        self.hub = hub
        self.port = _free_port()
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._runner = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=10), "relay failed to start"
        return self

    def __exit__(self, *exc):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        import aiohttp
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _serve():
            self._runner = aiohttp.web.AppRunner(self.hub.make_app())
            await self._runner.setup()
            site = aiohttp.web.TCPSite(self._runner, "127.0.0.1", self.port)
            await site.start()
            self._ready.set()

        self._loop.run_until_complete(_serve())
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()


class TestBrioEnroll:
    def test_enroll_persists_env_and_enables_api_server(self):
        from hermes_cli.config import get_env_value
        from hermes_constants import get_hermes_home

        hub = MockRelayHub()
        hub.claimed_codes["ABCD1234"] = {}
        with ThreadedRelay(hub) as relay:
            from hermes_cli.brio import brio_enroll

            args = SimpleNamespace(
                relay_url=relay.url, code="abcd1234", name="Test Agent",
            )
            rc = brio_enroll(args)
            assert rc == 0

        assert get_env_value("BRIO_RELAY_URL") == relay.url
        assert get_env_value("BRIO_AGENT_ID").startswith("hermes_")
        assert (get_env_value("BRIO_RELAY_TOKEN") or "").startswith("brio_agent_test_")
        assert get_env_value("API_SERVER_ENABLED") == "true"
        # A random API server key was generated and persisted.
        api_key = get_env_value("API_SERVER_KEY") or ""
        assert len(api_key) >= 32

        # The relay saw the claim with the expected payload.
        assert get_env_value("BRIO_AGENT_ID") in hub.agent_tokens

        # Unrelated .env entries are preserved (updateDotEnvFile semantics).
        env_file = get_hermes_home() / ".env"
        content = env_file.read_text(encoding="utf-8")
        assert "BRIO_RELAY_TOKEN=" in content
        assert "API_SERVER_ENABLED=true" in content

    def test_enroll_preserves_existing_api_server_key(self):
        from hermes_cli.config import get_env_value, save_env_value

        save_env_value("API_SERVER_KEY", "sk-existing-key-1234567890")
        hub = MockRelayHub()
        hub.claimed_codes["XYZ9999"] = {}
        with ThreadedRelay(hub) as relay:
            from hermes_cli.brio import brio_enroll

            args = SimpleNamespace(relay_url=relay.url, code="XYZ9999", name=None)
            rc = brio_enroll(args)
            assert rc == 0
        assert get_env_value("API_SERVER_KEY") == "sk-existing-key-1234567890"

    def test_enroll_reuses_stored_agent_id(self):
        from hermes_cli.config import get_env_value, save_env_value

        save_env_value("BRIO_AGENT_ID", "hermes_keepme")
        hub = MockRelayHub()
        hub.claimed_codes["KEEP01"] = {}
        with ThreadedRelay(hub) as relay:
            from hermes_cli.brio import brio_enroll

            args = SimpleNamespace(relay_url=relay.url, code="KEEP01", name=None)
            assert brio_enroll(args) == 0
        assert get_env_value("BRIO_AGENT_ID") == "hermes_keepme"

    def test_enroll_bad_code_fails_without_persisting(self):
        from hermes_cli.config import get_env_value, remove_env_value

        remove_env_value("BRIO_RELAY_TOKEN")
        hub = MockRelayHub()  # no codes registered
        with ThreadedRelay(hub) as relay:
            from hermes_cli.brio import brio_enroll

            args = SimpleNamespace(relay_url=relay.url, code="NOPE1", name=None)
            assert brio_enroll(args) == 1
        assert not get_env_value("BRIO_RELAY_TOKEN")

    def test_enroll_requires_relay_and_code(self, capsys):
        from hermes_cli.brio import brio_enroll

        assert brio_enroll(SimpleNamespace(relay_url="", code="X", name=None)) == 1
        assert brio_enroll(SimpleNamespace(relay_url="http://x", code="", name=None)) == 1


class TestBrioAdapter:
    """Gateway adapter lifecycle: connect() starts the tunnel, disconnect() stops it."""

    @staticmethod
    def _adapter_config(relay_url: str) -> PlatformConfig:
        return PlatformConfig(extra={
            "relay_url": relay_url,
            "relay_token": RELAY_TOKEN,
            "agent_id": AGENT_ID,
        })

    @pytest.mark.asyncio
    async def test_connect_starts_tunnel_and_disconnect_stops_it(self, hub):
        local = LocalAPIMock()
        relay_server = TestServer(hub.make_app())
        local_server = TestServer(local.make_app())
        async with relay_server as relay, local_server as api:
            hub_url = str(relay.make_url("/"))
            adapter = BrioAdapter(self._adapter_config(hub_url))
            adapter._api_port = api.port  # point the proxy at the local API mock
            assert await adapter.connect() is True
            try:
                await _wait_for(lambda: AGENT_ID in hub.companions, what="adapter tunnel connect")
                assert adapter.is_connected is True
            finally:
                await adapter.disconnect()
            await _wait_for(lambda: AGENT_ID not in hub.companions, what="adapter tunnel close")
            assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_returns_false_without_credentials(self, monkeypatch):
        for key in ("BRIO_RELAY_URL", "BRIO_RELAY_TOKEN", "BRIO_AGENT_ID"):
            monkeypatch.delenv(key, raising=False)
        adapter = BrioAdapter(PlatformConfig())
        assert await adapter.connect() is False

    @pytest.mark.asyncio
    async def test_env_gating_enables_brio_platform(self, monkeypatch):
        monkeypatch.setenv("BRIO_RELAY_URL", "http://127.0.0.1:18082")
        monkeypatch.setenv("BRIO_RELAY_TOKEN", "brio_agent_t")
        monkeypatch.setenv("BRIO_AGENT_ID", "hermes_t")
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        brio_cfg = config.platforms.get(Platform.BRIO)
        assert brio_cfg is not None and brio_cfg.enabled
        assert brio_cfg.extra["agent_id"] == "hermes_t"
