"""
Centralized Backend Server Configuration for SignAssistant.
Manages network targets (HTTP and WebSocket) with support for LAN IPs and localhost.
"""
import os
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

_DEFAULT_HOST = os.environ.get("SIGN_SERVER_HOST", "192.168.1.43")
_DEFAULT_PORT = int(os.environ.get("SIGN_SERVER_PORT", "8000"))

_current_host: str = _DEFAULT_HOST
_current_port: int = _DEFAULT_PORT


def get_server_host() -> str:
    """Returns current server host or IP."""
    return _current_host


def set_server_host(host: str) -> None:
    """Updates server host or IP dynamically."""
    global _current_host
    clean = host.strip()
    for prefix in ("http://", "https://", "ws://", "wss://"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    if ":" in clean:
        parts = clean.split(":")
        clean = parts[0]
        try:
            set_server_port(int(parts[1].split("/")[0]))
        except ValueError:
            pass
    clean = clean.rstrip("/")
    if clean:
        _current_host = clean
        logger.info(f"[Config] Updated server host to: {_current_host}:{_current_port}")


def get_server_port() -> int:
    """Returns server port."""
    return _current_port


def set_server_port(port: int) -> None:
    """Sets server port."""
    global _current_port
    if 1 <= port <= 65535:
        _current_port = port


def get_server_base_url() -> str:
    """Returns HTTP base URL."""
    return f"http://{_current_host}:{_current_port}"


def get_server_ws_url() -> str:
    """Returns WebSocket URL."""
    return f"ws://{_current_host}:{_current_port}/ws/sign_stream"


async def check_server_health(timeout: float = 2.5) -> Optional[dict]:
    """Checks whether the FastAPI backend is online and responding."""
    url = f"{get_server_base_url()}/health"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.get(url)
            if res.status_code == 200:
                return res.json()
    except Exception as e:
        logger.debug(f"Health check failed: {e}")
    return None
