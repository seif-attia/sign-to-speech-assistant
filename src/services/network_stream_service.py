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
    Background WebSocket client for streaming camera frames to FastAPI backend.
    Drops lagging/stale frames to maintain real-time low latency.
    """

    def __init__(
        self,
        server_ws_url: str,
        on_result_callback: Callable[[dict], None],
        on_status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.server_ws_url = server_ws_url
        self.on_result = on_result_callback
        self.on_status = on_status_callback

        self.is_running = False
        self._latest_payload: Optional[str] = None
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._active_ws = None

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
        except asyncio.CancelledError:
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

    def _notify_status(self, msg: str):
        if self.on_status:
            try:
                self.on_status(msg)
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
                            if self.on_result:
                                self.on_result(data)

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