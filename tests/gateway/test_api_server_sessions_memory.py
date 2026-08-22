"""
Tests for the Brio-facing API server endpoints:

- GET /v1/sessions            — list recent sessions (companion-compatible shape)
- GET /v1/sessions/{id}/messages — messages for one session
- GET /v1/memory / PUT /v1/memory — MEMORY.md / USER.md read/write

The JSON shapes mirror brio's companion handlers (sessions / sessionMessages /
memory / updateMemory in apps/companion/internal/server/server.go) so the Brio
mobile app can talk to the API server directly.
"""

import os
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app with the sessions/memory routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    # Match the real adapter's body limit (10 MB) so oversize-memory tests
    # exercise the handler's 1 MiB memory-file cap, not the server's limit.
    app = web.Application(middlewares=mws, client_max_size=10_000_000)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/sessions", adapter._handle_sessions)
    app.router.add_get("/v1/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_get("/v1/memory", adapter._handle_get_memory)
    app.router.add_put("/v1/memory", adapter._handle_update_memory)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


@pytest.fixture
def seeded_db():
    """Seed the isolated HERMES_HOME state.db with two sessions."""
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session(session_id="sess-a", source="cli", model="model-x", user_id="user-1")
        db.append_message("sess-a", role="user", content="Hello")
        db.append_message("sess-a", role="assistant", content="Hi there!")
        db.append_message("sess-a", role="tool", content="result", tool_name="terminal")
        db.end_session("sess-a", end_reason="user_exit")

        db.create_session(session_id="sess-b", source="telegram", model="model-y")
        db.append_message("sess-b", role="user", content="Second session")
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# /v1/sessions
# ---------------------------------------------------------------------------


class TestSessionsEndpoint:
    @pytest.mark.asyncio
    async def test_sessions_returns_companion_shape(self, adapter, seeded_db):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/sessions")
            assert resp.status == 200
            data = await resp.json()
            assert set(data.keys()) == {"sessions"}
            assert len(data["sessions"]) == 2
            first = data["sessions"][0]
            # Exact companion field set (server.go sessions handler)
            assert set(first.keys()) == {
                "id", "source", "user_id", "model", "started_at",
                "ended_at", "message_count", "title",
            }
            # Epoch-second floats
            assert isinstance(first["started_at"], float)
            by_id = {s["id"]: s for s in data["sessions"]}
            assert by_id["sess-a"]["source"] == "cli"
            assert by_id["sess-a"]["model"] == "model-x"
            assert by_id["sess-a"]["user_id"] == "user-1"
            assert by_id["sess-a"]["message_count"] == 3
            assert isinstance(by_id["sess-a"]["ended_at"], float)
            assert by_id["sess-b"]["ended_at"] is None

    @pytest.mark.asyncio
    async def test_sessions_limit_clamped(self, adapter, seeded_db):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/sessions", params={"limit": "1"})
            assert resp.status == 200
            data = await resp.json()
            assert len(data["sessions"]) == 1

            # Out-of-range limits fall back to clamped bounds, not errors.
            resp = await cli.get("/v1/sessions", params={"limit": "9999"})
            assert resp.status == 200
            resp = await cli.get("/v1/sessions", params={"limit": "not-an-int"})
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_sessions_requires_auth_when_key_configured(self, auth_adapter, seeded_db):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/sessions")
            assert resp.status == 401

            resp = await cli.get(
                "/v1/sessions", headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_sessions_empty_store_returns_empty_list(self, adapter):
        # SessionDB's DEFAULT_DB_PATH is cached at import time, so scrub the
        # shared state.db (hermes_state is imported once per test process).
        from hermes_state import DEFAULT_DB_PATH
        for suffix in ("", "-wal", "-shm"):
            Path(str(DEFAULT_DB_PATH) + suffix).unlink(missing_ok=True)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/sessions")
            assert resp.status == 200
            data = await resp.json()
            assert data["sessions"] == []


# ---------------------------------------------------------------------------
# /v1/sessions/{id}/messages
# ---------------------------------------------------------------------------


class TestSessionMessagesEndpoint:
    @pytest.mark.asyncio
    async def test_messages_returns_companion_shape(self, adapter, seeded_db):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/sessions/sess-a/messages")
            assert resp.status == 200
            data = await resp.json()
            assert set(data.keys()) == {"messages"}
            assert len(data["messages"]) == 3
            first = data["messages"][0]
            assert set(first.keys()) == {"role", "content", "tool_name", "timestamp"}
            assert first["role"] == "user"
            assert first["content"] == "Hello"
            assert first["tool_name"] == ""
            assert isinstance(first["timestamp"], float)
            tool_msg = data["messages"][2]
            assert tool_msg["role"] == "tool"
            assert tool_msg["tool_name"] == "terminal"

    @pytest.mark.asyncio
    async def test_messages_unknown_session_returns_error_key(self, adapter, seeded_db):
        """Companion behavior: unknown sessions yield 200 + empty list + error."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/sessions/does-not-exist/messages")
            assert resp.status == 200
            data = await resp.json()
            assert data["messages"] == []
            assert "error" in data

    @pytest.mark.asyncio
    async def test_messages_requires_auth_when_key_configured(self, auth_adapter, seeded_db):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/sessions/sess-a/messages")
            assert resp.status == 401


# ---------------------------------------------------------------------------
# /v1/memory
# ---------------------------------------------------------------------------


def _memories_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "memories"


class TestMemoryEndpoints:
    @pytest.mark.asyncio
    async def test_get_memory_returns_both_files(self, adapter):
        memories = _memories_dir()
        memories.mkdir(parents=True, exist_ok=True)
        (memories / "MEMORY.md").write_text("agent notes", encoding="utf-8")
        (memories / "USER.md").write_text("user profile", encoding="utf-8")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/memory")
            assert resp.status == 200
            data = await resp.json()
            assert data == {"memory": "agent notes", "user": "user profile"}

    @pytest.mark.asyncio
    async def test_get_memory_missing_files_returns_empty_strings(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/memory")
            assert resp.status == 200
            data = await resp.json()
            assert data == {"memory": "", "user": ""}

    @pytest.mark.asyncio
    async def test_put_memory_writes_only_provided_files(self, adapter):
        memories = _memories_dir()
        memories.mkdir(parents=True, exist_ok=True)
        (memories / "MEMORY.md").write_text("old", encoding="utf-8")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put("/v1/memory", json={"user": "new user"})
            assert resp.status == 200
            data = await resp.json()
            assert data == {"ok": True}

            assert (memories / "MEMORY.md").read_text(encoding="utf-8") == "old"
            assert (memories / "USER.md").read_text(encoding="utf-8") == "new user"

    @pytest.mark.asyncio
    async def test_put_memory_creates_directory_and_files(self, adapter):
        memories = _memories_dir()
        # The conftest hermetic home pre-creates the memories dir; the files
        # themselves must not exist yet.
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put("/v1/memory", json={"memory": "m", "user": "u"})
            assert resp.status == 200
            assert (memories / "MEMORY.md").read_text(encoding="utf-8") == "m"
            assert (memories / "USER.md").read_text(encoding="utf-8") == "u"

    @pytest.mark.asyncio
    async def test_put_memory_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/v1/memory", data="{not json", headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert data == {"error": "invalid JSON"}

    @pytest.mark.asyncio
    async def test_put_memory_non_string_value_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put("/v1/memory", json={"memory": 123})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_put_memory_empty_body_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put("/v1/memory", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_put_memory_oversized_value_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put("/v1/memory", json={"memory": "x" * (1024 * 1024 + 10)})
            assert resp.status == 400
            data = await resp.json()
            assert "larger than" in data["error"]

    @pytest.mark.asyncio
    async def test_put_memory_requires_auth_when_key_configured(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put("/v1/memory", json={"memory": "x"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_memory_round_trip(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put("/v1/memory", json={"memory": "round", "user": "trip"})
            assert resp.status == 200
            resp = await cli.get("/v1/memory")
            data = await resp.json()
            assert data == {"memory": "round", "user": "trip"}

    @pytest.mark.skipif(
        sys.platform.startswith("win"), reason="POSIX file modes not enforced on NTFS",
    )
    @pytest.mark.asyncio
    async def test_put_memory_files_are_owner_only(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put("/v1/memory", json={"memory": "secret"})
            assert resp.status == 200
        mode = os.stat(_memories_dir() / "MEMORY.md").st_mode & 0o777
        assert mode == 0o600
