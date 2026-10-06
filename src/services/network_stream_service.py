import asyncio
import base64
import json
import logging
import threading
from typing import Callable, Optional
import websockets

logger = logging.getLogger(__name__)


class NetworkSignStreamer:
    """
    Persistent background WebSocket client for communicating with the FastAPI backend.
    Maintains a continuous connection from app start across all views and modes.
    Drops lagging/stale frames to maintain real-time low latency.
    """

    def __init__(
        self,
        server_ws_url: Optional[str] = None,
        on_result_callback: Optional[Callable[[dict], None]] = None,
        on_status_callback: Optional[Callable[[str], None]] = None,
    ):
        from services.config import get_server_ws_url
        self.server_ws_url = server_ws_url or get_server_ws_url()
        self._result_callbacks: list[Callable[[dict], None]] = []
        self._status_callbacks: list[Callable[[str], None]] = []
        if on_result_callback:
            self._result_callbacks.append(on_result_callback)
        if on_status_callback:
            self._status_callbacks.append(on_status_callback)

        self.is_running = False
        self._latest_payload: Optional[str] = None
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._active_ws = None
        self.last_status: str = "Connecting..."

    def register_result_callback(self, cb: Callable[[dict], None]):
        """Registers a listener for incoming server messages."""
        with self._lock:
            if cb not in self._result_callbacks:
                self._result_callbacks.append(cb)

    def unregister_result_callback(self, cb: Callable[[dict], None]):
        """Unregisters a listener."""
        with self._lock:
            if cb in self._result_callbacks:
                self._result_callbacks.remove(cb)

    def register_status_callback(self, cb: Callable[[str], None]):
        """Registers a listener for connection status changes."""
        with self._lock:
            if cb not in self._status_callbacks:
                self._status_callbacks.append(cb)
        try:
            cb(self.last_status)
        except Exception:
            pass

    def unregister_status_callback(self, cb: Callable[[str], None]):
        """Unregisters a status listener."""
        with self._lock:
            if cb in self._status_callbacks:
                self._status_callbacks.remove(cb)

    def update_url(self, new_ws_url: str):
        """Updates the WebSocket endpoint and forces reconnection if connected."""
        with self._lock:
            self.server_ws_url = new_ws_url
        logger.info(f"Streamer URL updated to: {new_ws_url}")
        # Close current ws to trigger automatic reconnect to new address
        if self._active_ws and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._active_ws.close(), self._loop)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    def _run_event_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_handler())
        except (asyncio.CancelledError, RuntimeError):
            pass
        finally:
            try:
                # Cancel all remaining tasks cleanly
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                self._loop.close()

    def queue_frame(self, frame_bytes: bytes, is_front: bool = True, rotate: int = 0, target_lang: str = "en"):
        """Enqueues latest frame as Base64 string or JSON payload with target translation language."""
        if not self.is_running or not frame_bytes:
            return
        b64 = base64.b64encode(frame_bytes).decode("utf-8")
        payload = json.dumps({
            "frame": b64,
            "flip": is_front,
            "rotate": rotate,
            "target_lang": target_lang,
        })
        with self._lock:
            self._latest_payload = payload

    def send_command(self, action: str):
        """Sends an immediate action command (e.g. 'clear_sentence') to the server."""
        if not self.is_running:
            return
        payload = json.dumps({"action": action})
        with self._lock:
            self._latest_payload = payload

    def _notify_status(self, msg: str):
        self.last_status = msg
        with self._lock:
            cbs = list(self._status_callbacks)
        for cb in cbs:
            try:
                cb(msg)
            except Exception as ex:
                logger.debug(f"Error in status callback: {ex}")

    async def _ws_handler(self):
        while self.is_running:
            with self._lock:
                target_url = self.server_ws_url

            try:
                self._notify_status(f"Connecting to: {target_url}...")
                logger.info(f"Connecting to WS: {target_url}")

                async with websockets.connect(
                    target_url,
                    ping_interval=10,
                    ping_timeout=5,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    self._active_ws = ws
                    logger.info("Connected to Sign Translation Server.")
                    self._notify_status("Connected to Server")

                    async def sender():
                        sent_count = 0
                        while self.is_running:
                            payload = None
                            with self._lock:
                                if self._latest_payload is not None:
                                    payload = self._latest_payload
                                    self._latest_payload = None

                            if payload:
                                await ws.send(payload)
                                sent_count += 1
                                if sent_count == 1:
                                    print(f"[WS CLIENT] First frame sent over WebSocket successfully!", flush=True)
                            await asyncio.sleep(0.04)

                    async def receiver():
                        while self.is_running:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            with self._lock:
                                cbs = list(self._result_callbacks)
                            for cb in cbs:
                                try:
                                    cb(data)
                                except Exception as exc:
                                    logger.debug(f"Error in result callback: {exc}")

                    # Run sender and receiver concurrently until one exits/fails
                    tasks = [
                        asyncio.create_task(sender()),
                        asyncio.create_task(receiver()),
                    ]
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        # Raise any exception from failed task
                        task.result()

            except Exception as e:
                self._active_ws = None
                err_msg = f"Disconnected ({type(e).__name__}). Retrying in 2s..."
                logger.warning(f"WebSocket disconnected ({e}). Retrying in 2s...")
                self._notify_status(err_msg)
                await asyncio.sleep(2.0)

    def stop(self):
        self.is_running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


# --- Global Persistent WebSocket Streamer ---
_global_streamer: Optional[NetworkSignStreamer] = None
_streamer_lock = threading.Lock()


def get_global_streamer() -> NetworkSignStreamer:
    """
    Returns the persistent global WebSocket client, initializing and starting it
    if it hasn't already been started. Maintains a persistent connection across the app.
    """
    global _global_streamer
    with _streamer_lock:
        if _global_streamer is None:
            from services.config import get_server_ws_url
            _global_streamer = NetworkSignStreamer(server_ws_url=get_server_ws_url())
            _global_streamer.start()
        return _global_streamer


def stop_global_streamer() -> None:
    """Stops the global WebSocket client on application shutdown."""
    global _global_streamer
    with _streamer_lock:
        if _global_streamer is not None:
            _global_streamer.stop()
            _global_streamer = None