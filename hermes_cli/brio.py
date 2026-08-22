"""
Brio connector subcommand for the hermes CLI.

``hermes brio`` turns Hermes into a direct participant in the Brio control
plane (https://github.com/0bkevin/brio) — no separate companion binary:

- ``hermes brio enroll --relay-url URL --code CODE [--name NAME]``
    Claims an enrollment code on the relay, stores the relay credentials in
    ``~/.hermes/.env`` (BRIO_RELAY_URL / BRIO_RELAY_TOKEN / BRIO_AGENT_ID),
    and enables the local API server for the connector.

- ``hermes brio connect``
    Runs the tunnel connector in the foreground (the long-running process).

- ``hermes brio status``
    Shows the stored configuration (redacted), relay reachability, and
    whether the tunnel credentials are accepted by the relay.

- ``hermes brio recover --relay-url URL --agent-id ID --device-token TOKEN``
    Recovers relay credentials for an owned agent using the owner's device
    token (mirrors ``brio companion recover``).

Enrollment / recovery flows port brio's Go implementation
(``apps/companion/internal/cli/setup.go`` and ``companion.go``); the tunnel
client itself lives in ``gateway/platforms/brio_connector.py``.
"""

import asyncio
import json
import logging
import secrets
import socket as _socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli.colors import Colors, color

logger = logging.getLogger(__name__)

DEFAULT_API_SERVER_PORT = 8642

ENV_RELAY_URL = "BRIO_RELAY_URL"
ENV_RELAY_TOKEN = "BRIO_RELAY_TOKEN"
ENV_AGENT_ID = "BRIO_AGENT_ID"


def _mask(value: Optional[str]) -> str:
    """Redact a secret for display (never print full tokens)."""
    from agent.redact import mask_secret
    return mask_secret(value or "", empty=color("(not set)", Colors.DIM))


def _load_http() -> Any:
    try:
        import aiohttp
        return aiohttp
    except ImportError:  # pragma: no cover - aiohttp is a core dependency
        print(color("Error: aiohttp is required for the Brio connector.", Colors.RED), file=sys.stderr)
        raise SystemExit(1)


async def _post_json(aiohttp: Any, url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float = 30.0):
    """POST JSON and return (status, body_dict_or_text)."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            try:
                body = json.loads(text) if text else {}
            except json.JSONDecodeError:
                body = {"raw": text}
            return resp.status, body


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------


def _generate_agent_id() -> str:
    """Stable per-machine agent id: ``hermes_`` + random hex (see setup.go)."""
    return "hermes_" + secrets.token_hex(6)


def brio_enroll(args) -> int:
    """Claim an enrollment code and persist relay credentials."""
    from hermes_cli.config import get_env_value, save_env_value

    aiohttp = _load_http()
    relay_url = (args.relay_url or "").strip().rstrip("/")
    code = (args.code or "").strip().upper()
    name = (args.name or "").strip() or _socket.gethostname()

    if not relay_url:
        print(color("Error: --relay-url is required", Colors.RED), file=sys.stderr)
        return 1
    if not code:
        print(color("Error: --code is required", Colors.RED), file=sys.stderr)
        return 1

    agent_id = (get_env_value(ENV_AGENT_ID) or "").strip() or _generate_agent_id()

    payload: Dict[str, Any] = {"agent_id": agent_id}
    if name:
        payload["name"] = name

    from urllib.parse import quote
    claim_url = f"{relay_url}/enrollments/{quote(code, safe='')}/claim"
    print(color(f"Claiming enrollment {code} from {relay_url} ...", Colors.DIM))
    try:
        status, body = asyncio.run(
            _post_json(aiohttp, claim_url, payload, {"Content-Type": "application/json"}),
        )
    except Exception as exc:
        print(color(f"Error: could not reach relay: {exc}", Colors.RED), file=sys.stderr)
        return 1

    if status >= 300:
        detail = body.get("error") if isinstance(body, dict) else body
        print(
            color(f"Error: enrollment claim failed (HTTP {status}): {detail}", Colors.RED),
            file=sys.stderr,
        )
        return 1

    agent = body.get("agent") or {}
    relay_token = body.get("relay_token") or ""
    final_agent_id = str(agent.get("id") or agent_id)
    final_name = str(agent.get("name") or name)
    if not relay_token or not final_agent_id:
        print(color("Error: enrollment response is incomplete", Colors.RED), file=sys.stderr)
        return 1

    # Persist credentials + enable the local API server for the connector
    # (mirrors setup.go ensureHermesAPIConfig: preserve an existing key).
    save_env_value(ENV_RELAY_URL, relay_url)
    save_env_value(ENV_RELAY_TOKEN, relay_token)
    save_env_value(ENV_AGENT_ID, final_agent_id)
    save_env_value("API_SERVER_ENABLED", "true")
    existing_key = (get_env_value("API_SERVER_KEY") or "").strip()
    if not existing_key:
        save_env_value("API_SERVER_KEY", secrets.token_hex(32))

    print()
    print(color("◆ Brio enrollment complete", Colors.GREEN, Colors.BOLD))
    print(f"  Agent:      {final_name}")
    print(f"  Agent ID:   {final_agent_id}")
    print(f"  Relay:      {relay_url}")
    print(f"  Relay token: {_mask(relay_token)}")
    print()
    print(color("Next steps:", Colors.CYAN))
    print("  1. The agent should now appear in the Brio mobile app.")
    print("  2. Restart the gateway — it now runs the Brio relay tunnel")
    print("     automatically whenever BRIO_* credentials are configured:")
    print(color("       hermes gateway restart", Colors.YELLOW))
    print("     (not installed as a service yet? run: hermes gateway install)")
    return 0


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


def brio_connect(args) -> int:
    """Run the Brio tunnel connector in the foreground."""
    from gateway.platforms.brio_connector import BrioConnector
    from hermes_cli.config import get_env_value

    aiohttp = _load_http()
    relay_url = (getattr(args, "relay_url", None) or get_env_value(ENV_RELAY_URL) or "").strip()
    relay_token = (getattr(args, "relay_token", None) or get_env_value(ENV_RELAY_TOKEN) or "").strip()
    agent_id = (getattr(args, "agent_id", None) or get_env_value(ENV_AGENT_ID) or "").strip()

    if not relay_url or not relay_token or not agent_id:
        print(
            color(
                "Brio is not enrolled yet. Run:\n"
                "  hermes brio enroll --relay-url <relay> --code <code>",
                Colors.YELLOW,
            ),
            file=sys.stderr,
        )
        return 1

    port = (get_env_value("API_SERVER_PORT") or str(DEFAULT_API_SERVER_PORT)).strip() or str(DEFAULT_API_SERVER_PORT)
    api_base = f"http://127.0.0.1:{port}"
    api_key = (get_env_value("API_SERVER_KEY") or "").strip()

    connector = BrioConnector(
        relay_url=relay_url,
        agent_id=agent_id,
        relay_token=relay_token,
        api_base_url=api_base,
        api_key=api_key,
    )

    # Foreground process: log to the console unless something already
    # configured handlers (e.g. when embedded in the gateway).
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    print(color(f"◆ Brio connector (agent {agent_id})", Colors.CYAN))
    print(f"  Relay:      {relay_url}")
    print(f"  Local API:  {api_base}")
    print(f"  API key:    {_mask(api_key)}")
    print()

    async def _run() -> None:
        session = aiohttp.ClientSession()
        try:
            await connector.run(http_session=session)
        finally:
            await session.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print()
        print(color("Brio connector stopped.", Colors.DIM))
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


async def _probe(aiohttp: Any, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.get(url, headers=headers or {}) as resp:
            return resp.status, await resp.text()


async def _probe_tunnel(aiohttp: Any, ws_url: str, timeout: float = 5.0) -> str:
    """Open the companion tunnel socket briefly to verify credentials."""
    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(
            ws_url, heartbeat=None, max_msg_size=12 * 1024 * 1024,
            timeout=aiohttp.ClientWSTimeout(ws_close=timeout),
        )
        await ws.close()
        return "authorized"


def brio_status(args) -> int:
    """Show stored Brio configuration (redacted) and relay/tunnel health."""
    from hermes_cli.config import get_env_value

    aiohttp = _load_http()
    relay_url = (get_env_value(ENV_RELAY_URL) or "").strip()
    relay_token = (get_env_value(ENV_RELAY_TOKEN) or "").strip()
    agent_id = (get_env_value(ENV_AGENT_ID) or "").strip()

    print()
    print(color("◆ Brio connector status", Colors.CYAN, Colors.BOLD))
    print(f"  Relay URL:   {relay_url or color('(not enrolled)', Colors.DIM)}")
    print(f"  Agent ID:    {agent_id or color('(not set)', Colors.DIM)}")
    print(f"  Relay token: {_mask(relay_token)}")

    if not relay_url or not agent_id:
        print()
        print(color("Not enrolled. Run `hermes brio enroll --relay-url <relay> --code <code>`.", Colors.DIM))
        return 1

    async def _checks() -> Dict[str, str]:
        results: Dict[str, str] = {}
        try:
            status, text = await _probe(aiohttp, f"{relay_url}/health")
            results["relay"] = f"reachable (HTTP {status}, {text.strip()[:120]})"
        except Exception as exc:
            results["relay"] = f"unreachable ({exc})"
        try:
            from gateway.platforms.brio_connector import tunnel_url
            ws_url = tunnel_url(relay_url, "companion", agent_id, relay_token)
            state = await _probe_tunnel(aiohttp, ws_url)
            results["tunnel"] = f"credentials valid ({state}) — connect with `hermes brio connect`"
        except Exception as exc:
            detail = str(exc)
            results["tunnel"] = f"rejected ({detail[:160]})"
        return results

    results = asyncio.run(_checks())
    print(f"  Relay:       {results.get('relay', 'unknown')}")
    print(f"  Tunnel:      {results.get('tunnel', 'unknown')}")
    print()
    return 0


# ---------------------------------------------------------------------------
# recover
# ---------------------------------------------------------------------------


def brio_recover(args) -> int:
    """Recover relay credentials for an owned agent via the owner device token."""
    from hermes_cli.config import get_env_value, save_env_value

    aiohttp = _load_http()
    relay_url = (args.relay_url or "").strip().rstrip("/")
    agent_id = (args.agent_id or "").strip()
    device_token = (args.device_token or "").strip()
    name = (args.name or "").strip()

    if not relay_url:
        print(color("Error: --relay-url is required", Colors.RED), file=sys.stderr)
        return 1
    if not agent_id:
        print(color("Error: --agent-id is required", Colors.RED), file=sys.stderr)
        return 1
    if not device_token:
        print(color("Error: --device-token is required", Colors.RED), file=sys.stderr)
        return 1

    from urllib.parse import quote
    recover_url = f"{relay_url}/agents/{quote(agent_id, safe='')}/recover"
    payload: Dict[str, Any] = {}
    if name:
        payload["name"] = name
    headers = {
        "Authorization": f"Bearer {device_token}",
        "Content-Type": "application/json",
    }
    print(color(f"Recovering agent {agent_id} from {relay_url} ...", Colors.DIM))
    try:
        status, body = asyncio.run(_post_json(aiohttp, recover_url, payload, headers))
    except Exception as exc:
        print(color(f"Error: could not reach relay: {exc}", Colors.RED), file=sys.stderr)
        return 1

    if status >= 300:
        detail = body.get("error") if isinstance(body, dict) else body
        print(
            color(f"Error: relay recovery failed (HTTP {status}): {detail}", Colors.RED),
            file=sys.stderr,
        )
        return 1

    new_token = body.get("agent_token") or ""
    final_agent_id = str(body.get("agent_id") or agent_id)
    if not new_token:
        print(color("Error: relay recovery response is incomplete", Colors.RED), file=sys.stderr)
        return 1

    save_env_value(ENV_RELAY_URL, relay_url)
    save_env_value(ENV_RELAY_TOKEN, new_token)
    save_env_value(ENV_AGENT_ID, final_agent_id)

    print()
    print(color("◆ Brio credentials recovered", Colors.GREEN, Colors.BOLD))
    print(f"  Agent ID:   {final_agent_id}")
    print(f"  Name:       {body.get('name') or '(unchanged)'}")
    print(f"  Relay token: {_mask(new_token)}")
    if body.get("code"):
        print(f"  Pairing code (valid 10 min, for the mobile app): {body['code']}")
    print()
    print("Reconnect with:")
    print(color("  hermes brio connect", Colors.YELLOW))
    return 0


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def brio_command(args) -> int:
    """Handle `hermes brio` subcommands."""
    subcmd = getattr(args, "brio_command", None)

    if subcmd == "enroll":
        return brio_enroll(args)
    if subcmd == "connect":
        return brio_connect(args)
    if subcmd == "status":
        return brio_status(args)
    if subcmd == "recover":
        return brio_recover(args)

    print(
        "usage: hermes brio <subcommand>\n"
        "\n"
        "subcommands:\n"
        "  enroll    Claim an enrollment code and store relay credentials\n"
        "  connect   Run the relay tunnel connector in the foreground\n"
        "  status    Show stored configuration and relay/tunnel health\n"
        "  recover   Recover relay credentials with an owner device token\n"
        "\n"
        "Run `hermes brio <subcommand> -h` for details.",
        file=sys.stderr,
    )
    return 1
