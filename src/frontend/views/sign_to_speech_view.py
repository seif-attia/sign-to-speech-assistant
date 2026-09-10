import asyncio
import io
import logging
import os
from pathlib import Path
import threading
import time
import flet as ft
import flet_camera as fc
from PIL import Image, ImageOps

from components.camera_view import CameraView
from components.vector_view import VectorView
from services.config import (
    get_server_ws_url,
    get_server_host,
    set_server_host,
    get_server_port,
    set_server_port,
    check_server_health,
)
from services.network_stream_service import NetworkSignStreamer
from frontend.components.bottom_nav_bar import create_nav_bar

logger = logging.getLogger(__name__)

VIEWPORT_WIDTH = 360
VIEWPORT_HEIGHT = 380


class SignToSpeechView(ft.View):
    def __init__(self, page: ft.Page):
        self.app_page = page
        self._is_active: bool = True
        self.is_front_camera: bool = True

        # State Variables
        self.sentence_ribbon: list[str] = []
        self.last_commit_word: str = ""
        self.last_pred_time: float = 0.0
        self.HOLD_DURATION = 1.2

        # Speech Synthesis State
        self.is_speech_enabled: bool = True
        self._last_spoken_word: str = ""
        self._last_spoken_sentence: str = ""
        self._last_spoken_time: float = 0.0

        async def close(e: ft.ControlEvent):
            await self.cleanup_async()
            await self.app_page.push_route("/")

        super().__init__(
            route="/sign-speech",
            padding=ft.Padding.only(top=15, left=15, right=15, bottom=15),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            appbar=ft.AppBar(
                leading=ft.IconButton(ft.Icons.ARROW_BACK, on_click=close),
                title=ft.Text("Sign To Speech", color=ft.Colors.BLACK),
                bgcolor=ft.Colors.SURFACE,
                actions=[
                    ft.IconButton(
                        ft.Icons.SETTINGS_ETHERNET,
                        tooltip="Configure Server IP",
                        on_click=self._open_server_settings_dialog,
                    )
                ],
            ),
            navigation_bar=create_nav_bar(1, self.app_page),
        )

        # 1. Camera View
        self.camera_view_component = CameraView(
            width=VIEWPORT_WIDTH,
            height=VIEWPORT_HEIGHT,
            resolution=fc.ResolutionPreset.MEDIUM,
            lens_direction=fc.CameraLensDirection.FRONT,
            on_lens_change=self._on_camera_flip,
        )
        self.hand_display = VectorView(width=VIEWPORT_WIDTH - 30)

        # 2. UI Status and Ribbon Cards
        self.gesture_icon = ft.Icon(ft.Icons.FRONT_HAND, color=ft.Colors.BLUE, size=28)
        self.prediction_text = ft.Text(
            "STANDBY",
            size=20,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.BLUE_900,
        )
        self.confidence_text = ft.Text(
            "Raise hands in camera view",
            size=12,
            color=ft.Colors.GREY_700,
        )
        self.sentence_ribbon_text = ft.Text(
            "GLOSS: (waiting for signs...)",
            size=12,
            weight=ft.FontWeight.W_500,
            color=ft.Colors.BLUE_GREY_800,
        )
        self.qwen_sentence_text = ft.Text(
            "English: (translating...)",
            size=13,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.GREEN_900,
        )
        self.status_bar_text = ft.Text(
            f"Server: {get_server_host()} (Connecting...)",
            size=11,
            color=ft.Colors.GREY_600,
            italic=True,
        )
        self.speech_button = ft.IconButton(
            icon=ft.Icons.VOLUME_UP,
            icon_color=ft.Colors.BLUE_700,
            tooltip="Mute / Unmute Speech",
            on_click=self._toggle_speech,
        )

        self.camera_diag_text = ft.Text(
            "Frames: 0 sent",
            size=10,
            color=ft.Colors.GREY_500,
        )

        self.prediction_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.gesture_icon,
                            ft.Column(
                                controls=[
                                    self.prediction_text,
                                    self.confidence_text,
                                ],
                                spacing=2,
                                expand=True,
                            ),
                            self.speech_button,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Divider(height=1, color=ft.Colors.BLUE_100),
                    self.sentence_ribbon_text,
                    self.qwen_sentence_text,
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.WIFI, size=14, color=ft.Colors.GREEN_600),
                            self.status_bar_text,
                            ft.Container(expand=True),
                            self.camera_diag_text,
                        ],
                        spacing=4,
                    ),
                ],
                spacing=6,
            ),
            padding=10,
            border_radius=8,
            bgcolor=ft.Colors.BLUE_50,
            width=VIEWPORT_WIDTH,
        )

        # 3. WebSocket Streamer
        self.streamer = NetworkSignStreamer(
            server_ws_url=get_server_ws_url(),
            on_result_callback=self._handle_server_prediction,
            on_status_callback=self._handle_server_status,
        )
        self.streamer.start()

        self.controls = [
            ft.Column(
                controls=[
                    self.camera_view_component,
                    self.prediction_card,
                    self.hand_display,
                ],
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                scroll=ft.ScrollMode.ALWAYS,
            )
        ]

        # 4. Inlined Camera Frame Pumping Task
        self._is_capturing: bool = False
        self._capture_task = self.app_page.run_task(self._frame_pump_loop)

    def _open_server_settings_dialog(self, e: ft.ControlEvent):
        """Allows dynamically updating server host IP in mobile or desktop interface."""
        host_input = ft.TextField(
            label="FastAPI Server IP / Host",
            value=get_server_host(),
            hint_text="e.g. 192.168.1.43 or 127.0.0.1",
            width=280,
        )
        port_input = ft.TextField(
            label="Port",
            value=str(get_server_port()),
            hint_text="8000",
            width=280,
        )
        dialog_status = ft.Text("", size=12, italic=True)

        async def save_and_reconnect(ev):
            new_host = host_input.value.strip()
            try:
                new_port = int(port_input.value.strip())
                set_server_port(new_port)
            except ValueError:
                pass

            if new_host:
                set_server_host(new_host)

            dialog_status.value = "Testing connection..."
            dialog_status.color = ft.Colors.BLUE_700
            self.app_page.update()

            health = await check_server_health(timeout=2.0)
            if health:
                dialog_status.value = f"Success! Backend online ({health.get('classes_count', '?')} classes)"
                dialog_status.color = ft.Colors.GREEN_700
            else:
                dialog_status.value = "Warning: Could not reach /health. Reconnecting WS anyway..."
                dialog_status.color = ft.Colors.ORANGE_800

            new_ws_url = get_server_ws_url()
            self.streamer.update_url(new_ws_url)
            self.status_bar_text.value = f"Server: {get_server_host()} (Connecting...)"
            self._safe_update(self.prediction_card)
            self.app_page.update()
            await asyncio.sleep(1.0)
            settings_dialog.open = False
            self.app_page.update()

        def cancel_dialog(ev):
            settings_dialog.open = False
            self.app_page.update()

        settings_dialog = ft.AlertDialog(
            title=ft.Text("Backend Connection Settings"),
            content=ft.Column(
                controls=[
                    ft.Text("Enter workstation IP where FastAPI server is running:"),
                    host_input,
                    port_input,
                    dialog_status,
                ],
                spacing=8,
                tight=True,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=cancel_dialog),
                ft.ElevatedButton("Save & Connect", on_click=save_and_reconnect),
            ],
        )
        self.app_page.overlay.append(settings_dialog)
        settings_dialog.open = True
        self.app_page.update()

    def _handle_server_status(self, status: str) -> None:
        """Called whenever WebSocket connection state changes."""
        self.status_bar_text.value = f"Status: {status}"
        self._safe_update(self.prediction_card)

    def _process_and_compress_frame(self, raw_input) -> bytes:
        """
        Processes captured image (bytes or filepath string), corrects orientation with EXIF,
        mirrors if front camera, downscales to 240x320, and compresses as JPEG.
        """
        try:
            if isinstance(raw_input, (bytes, bytearray)):
                img_io = io.BytesIO(raw_input)
                img = Image.open(img_io)
            elif isinstance(raw_input, str) and os.path.exists(raw_input):
                img = Image.open(raw_input)
            else:
                return b""

            with img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")

                if self.is_front_camera:
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

                # Resize to efficient 240x320 dimensions for low latency transmission
                img = img.resize((240, 320), Image.Resampling.BILINEAR)

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=50, optimize=False)
                return buffer.getvalue()
        except Exception as e:
            print(f"[FRAME COMPRESS ERROR] {e}", flush=True)
            return b""
        finally:
            if isinstance(raw_input, str):
                try:
                    if os.path.exists(raw_input):
                        os.remove(raw_input)
                except Exception:
                    pass

    async def _frame_pump_loop(self):
        """Waits for hardware camera readiness and loops image capture."""
        logger.info("[CLIENT CAM] Waiting for camera hardware to initialize...")

        cam = self.camera_view_component.camera

        # Wait until camera is properly initialized by CameraView
        ready_wait_count = 0
        while self._is_active:
            is_ready = getattr(self.camera_view_component, "is_camera_ready", False)
            status_val = (self.camera_view_component.status_text.value or "").lower()
            if is_ready or "ready" in status_val:
                break
            ready_wait_count += 1
            if ready_wait_count % 5 == 0:
                raw_st = self.camera_view_component.status_text.value or "Init..."
                self.camera_diag_text.value = f"Cam: {raw_st[:15]}"
                self._safe_update(self.camera_diag_text)
                print(f"[CLIENT CAM] Waiting for camera... Current status: '{raw_st}'", flush=True)
            await asyncio.sleep(0.2)

        if not self._is_active:
            return

        self.camera_diag_text.value = "Frames: 0 sent"
        self._safe_update(self.camera_diag_text)
        print(f"[CLIENT CAM] Camera initialized: {self.camera_view_component.status_text.value}. Starting frame transmission.", flush=True)
        capture_err_count = 0
        frames_sent_count = 0
        last_diag_time = time.perf_counter()

        while self._is_active:
            start_t = time.perf_counter()

            # Check if camera is currently marked ready
            if getattr(self.camera_view_component, "is_camera_ready", True) and not self._is_capturing:
                self._is_capturing = True
                try:
                    raw_img = await cam.take_picture()
                    if raw_img:
                        capture_err_count = 0
                        # Process and compress in background worker thread
                        frame_bytes = await asyncio.to_thread(
                            self._process_and_compress_frame, raw_img
                        )
                        if frame_bytes and len(frame_bytes) > 0:
                            frames_sent_count += 1
                            self.streamer.queue_frame(
                                frame_bytes,
                                is_front=False,  # Already mirrored in _process_and_compress_frame
                                rotate=0,
                            )
                            if frames_sent_count == 1:
                                print(f"[CLIENT CAM] First frame captured & queued ({len(frame_bytes)} bytes)!", flush=True)

                            now_diag = time.perf_counter()
                            if now_diag - last_diag_time >= 1.0:
                                last_diag_time = now_diag
                                self.camera_diag_text.value = f"Frames: {frames_sent_count} sent"
                                self._safe_update(self.camera_diag_text)
                except Exception as exc:
                    capture_err_count += 1
                    if capture_err_count % 30 == 1:
                        print(f"[CLIENT CAM ERROR] Capture failed ({capture_err_count}x): {exc}", flush=True)
                        logger.warning(f"[CLIENT CAM] Capture pause: {exc}")
                        self.camera_diag_text.value = f"Cam Err: {type(exc).__name__}"
                        self._safe_update(self.camera_diag_text)
                finally:
                    self._is_capturing = False

            # Target ~12-15 FPS cadence for smooth real-time tracking
            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.02, 0.07 - elapsed)
            await asyncio.sleep(sleep_time)

    def _safe_update(self, control: ft.Control) -> None:
        if not self._is_active:
            return

        def _update():
            try:
                if self.app_page:
                    control.update()
            except Exception:
                pass

        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.app_page.run_thread(_update)

    def _speak_text(self, text: str) -> None:
        """Non-blocking TTS speech output."""
        if not self.is_speech_enabled or not self._is_active or not text:
            return

        now = time.time()
        if (now - self._last_spoken_time) < 1.0:
            return
        self._last_spoken_time = now

        def _tts():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.debug(f"TTS error: {e}")

        threading.Thread(target=_tts, daemon=True).start()

    def _toggle_speech(self, e: ft.ControlEvent) -> None:
        self.is_speech_enabled = not self.is_speech_enabled
        self.speech_button.icon = (
            ft.Icons.VOLUME_UP if self.is_speech_enabled else ft.Icons.VOLUME_OFF
        )
        self.speech_button.icon_color = (
            ft.Colors.BLUE_700 if self.is_speech_enabled else ft.Colors.GREY_500
        )
        self._safe_update(self.speech_button)

    def _handle_server_prediction(self, payload: dict) -> None:
        if not self._is_active:
            return

        status = payload.get("status")
        ribbon = payload.get("sentence_ribbon", [])
        hands_present = payload.get("hands_present", False)
        is_signing = payload.get("is_signing", False)
        qwen_trans = payload.get("qwen_sentence")

        self.status_bar_text.value = f"Server: {get_server_host()} (Streaming Live)"

        # 1. Update Gloss & Natural Sentence
        if ribbon:
            self.sentence_ribbon_text.value = f"GLOSS: {' '.join(ribbon)}"
        if qwen_trans:
            self.qwen_sentence_text.value = f"English: {qwen_trans}"
            if qwen_trans != self._last_spoken_sentence:
                self._last_spoken_sentence = qwen_trans
                self._speak_text(qwen_trans)

        # 2. Live Gesture HUD
        now_ts = time.perf_counter()
        if status == "PREDICTION" and payload.get("word"):
            word = payload["word"]
            conf = payload.get("confidence", 0.0)

            self.last_pred_time = now_ts
            self.prediction_text.value = f"🎯 {word.upper()}"
            self.prediction_text.color = (
                ft.Colors.GREEN_800 if conf >= 0.50 else ft.Colors.CYAN_900
            )
            self.confidence_text.value = f"Confidence: {conf * 100:.1f}%"

            if len(ribbon) > 0 and ribbon[-1] == word and word != self.last_commit_word:
                self.last_commit_word = word
                # Speak detected sign word if no sentence has been translated yet
                if not qwen_trans:
                    self._speak_text(word)

        elif not hands_present:
            hold_active = (now_ts - self.last_pred_time) < self.HOLD_DURATION
            if not hold_active:
                self.prediction_text.value = "STANDBY"
                self.prediction_text.color = ft.Colors.BLUE_900
                self.confidence_text.value = "Raise hands into camera view"
        elif not is_signing:
            hold_active = (now_ts - self.last_pred_time) < self.HOLD_DURATION
            if not hold_active:
                self.prediction_text.value = "RESTING"
                self.prediction_text.color = ft.Colors.BLUE_700
                self.confidence_text.value = "Lift hands into signing space"

        # 3. Update Vector View text preview
        tracking_info = (
            f"Hands Detected: {'YES' if hands_present else 'NO'}\n"
            f"Signing Active: {'YES' if is_signing else 'NO'}\n"
            f"Model Status:   {status}\n"
            f"Last Sign:      {payload.get('word', 'None')}\n"
            f"Gloss Count:    {len(ribbon)}"
        )
        self.hand_display.text_control.value = tracking_info

        self._safe_update(self.prediction_card)
        self._safe_update(self.hand_display)

    async def cleanup_async(self):
        self._is_active = False
        if hasattr(self, "_capture_task") and self._capture_task:
            self._capture_task.cancel()
        if hasattr(self, "streamer") and self.streamer:
            self.streamer.stop()

    def _on_camera_flip(self, new_direction: fc.CameraLensDirection):
        self.is_front_camera = (new_direction == fc.CameraLensDirection.FRONT)
        self.status_bar_text.value = f"Camera: {'Front' if self.is_front_camera else 'Back'}"
        self._safe_update(self.prediction_card)