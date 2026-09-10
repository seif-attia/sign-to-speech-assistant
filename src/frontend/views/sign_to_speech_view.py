import asyncio
import io
import logging
import os
import threading
import time
from PIL import Image, ImageOps
import flet as ft
import flet_camera as fc

from components.camera_view import CameraView
from components.vector_view import VectorView
from services.config import (
    check_server_health,
    get_server_host,
    get_server_port,
    get_server_ws_url,
    set_server_host,
    set_server_port,
)
from services.network_stream_service import NetworkSignStreamer
from frontend.theme import (
    BG_DARK,
    CARD_BG,
    CARD_BORDER,
    ACCENT_EMERALD,
    ACCENT_MINT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    create_glass_card,
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar

logger = logging.getLogger(__name__)

VIEWPORT_HEIGHT = 380


class SignToSpeechView(ft.View):
    """
    Screen 3: Sign-to-Speech Mode View (/sign-speech).
    Designed matching the WESAL UI:
    - Top bar: '← Modes' back button on left, Server IP settings on top right, Front Cam flip button.
    - Clean rounded camera viewfinder.
    - Bottom 'DETECTING GESTURES...' glass card with live recognized English sentence ribbon.
    """

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

        # Top Bar: Left '← Modes', Right: Server IP dialog + Front Cam toggle
        left_modes_btn = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ARROW_BACK_IOS_NEW, size=13, color=ACCENT_MINT),
                    ft.Text("Modes", size=13, weight=ft.FontWeight.W_500, color=ACCENT_MINT),
                ],
                spacing=4,
            ),
            on_click=close,
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
            border_radius=8,
            bgcolor="#132323",
        )

        server_ip_btn = ft.IconButton(
            icon=ft.Icons.SETTINGS_ETHERNET,
            icon_color=ACCENT_MINT,
            icon_size=18,
            tooltip="Configure Server IP",
            on_click=self._open_server_settings_dialog,
        )

        self.front_cam_label = ft.Text("Front Cam", size=11, color=TEXT_SECONDARY, weight=ft.FontWeight.W_500)
        self.front_cam_btn = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.SYNC_ROUNDED, size=14, color=ACCENT_MINT),
                    self.front_cam_label,
                ],
                spacing=4,
            ),
            padding=ft.Padding.symmetric(horizontal=8, vertical=5),
            border_radius=10,
            bgcolor="#132323",
            on_click=self._flip_camera_trigger,
        )

        top_nav_row = ft.Row(
            controls=[
                left_modes_btn,
                ft.Row(
                    controls=[
                        server_ip_btn,
                        self.front_cam_btn,
                    ],
                    spacing=6,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            width=VIEWPORT_WIDTH,
        )

        super().__init__(
            route="/sign-speech",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=16, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(0, self.app_page),
        )

        # 1. Camera View Component
        self.camera_view_component = CameraView(
            width=VIEWPORT_WIDTH,
            height=VIEWPORT_HEIGHT,
            resolution=fc.ResolutionPreset.MEDIUM,
            lens_direction=fc.CameraLensDirection.FRONT,
            on_lens_change=self._on_camera_flip,
        )

        # 2. Bottom Live Gesture Output Card
        self.speech_button = ft.IconButton(
            icon=ft.Icons.VOLUME_UP_ROUNDED,
            icon_color=ACCENT_MINT,
            icon_size=20,
            tooltip="Mute / Unmute Speech",
            on_click=self._toggle_speech,
        )

        self.detecting_label = ft.Text(
            "DETECTING GESTURES...",
            size=10,
            weight=ft.FontWeight.BOLD,
            color=ACCENT_MINT,
            style=ft.TextStyle(letter_spacing=1.2)
        )

        self.gesture_output_text = ft.Text(
            '"Waiting for signs..."',
            size=15,
            weight=ft.FontWeight.BOLD,
            color=TEXT_PRIMARY,
        )

        self.confidence_text = ft.Text(
            "Confidence: 0.0%",
            size=11,
            color=TEXT_MUTED,
        )

        self.camera_diag_text = ft.Text(
            "Frames: 0 sent",
            size=10,
            color=TEXT_MUTED,
        )

        self.status_bar_text = ft.Text(
            f"Server: {get_server_host()}",
            size=10,
            color=TEXT_MUTED,
        )

        self.prediction_card = create_glass_card(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Container(width=6, height=6, border_radius=3, bgcolor=ACCENT_MINT),
                                    self.detecting_label,
                                ],
                                spacing=6,
                            ),
                            self.speech_button,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.gesture_output_text,
                    ft.Row(
                        controls=[
                            self.confidence_text,
                            ft.Container(expand=True),
                            self.status_bar_text,
                            ft.Container(width=6),
                            self.camera_diag_text,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ],
                spacing=8,
            ),
            padding=16,
            border_radius=18,
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
                    top_nav_row,
                    ft.Container(height=4),
                    self.camera_view_component,
                    ft.Container(height=4),
                    self.prediction_card,
                ],
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                scroll=ft.ScrollMode.ALWAYS,
            )
        ]

        # 4. Inlined Camera Frame Pumping Task
        self._is_capturing: bool = False
        self._capture_task = self.app_page.run_task(self._frame_pump_loop)

    async def _flip_camera_trigger(self, e):
        await self.camera_view_component.flip_camera(e)

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
            self.status_bar_text.value = f"Server: {get_server_host()}"
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
        self.status_bar_text.value = f"Status: {status[:12]}"
        self._safe_update(self.prediction_card)

    def _process_and_compress_frame(self, raw_input) -> bytes:
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
        logger.info("[CLIENT CAM] Waiting for camera hardware to initialize...")
        cam = self.camera_view_component.camera

        ready_wait_count = 0
        while self._is_active:
            is_ready = getattr(self.camera_view_component, "is_camera_ready", False)
            status_val = (self.camera_view_component.status_text.value or "").lower()
            if is_ready or "ready" in status_val:
                break
            ready_wait_count += 1
            if ready_wait_count % 5 == 0:
                raw_st = self.camera_view_component.status_text.value or "Init..."
                self.camera_diag_text.value = f"Cam: {raw_st[:12]}"
                self._safe_update(self.camera_diag_text)
            await asyncio.sleep(0.2)

        if not self._is_active:
            return

        self.camera_diag_text.value = "Frames: 0 sent"
        self._safe_update(self.camera_diag_text)
        capture_err_count = 0
        frames_sent_count = 0
        last_diag_time = time.perf_counter()

        while self._is_active:
            start_t = time.perf_counter()

            if getattr(self.camera_view_component, "is_camera_ready", True) and not self._is_capturing:
                self._is_capturing = True
                try:
                    raw_img = await cam.take_picture()
                    if raw_img:
                        capture_err_count = 0
                        frame_bytes = await asyncio.to_thread(
                            self._process_and_compress_frame, raw_img
                        )
                        if frame_bytes and len(frame_bytes) > 0:
                            frames_sent_count += 1
                            self.streamer.queue_frame(
                                frame_bytes,
                                is_front=False,
                                rotate=0,
                            )
                            now_diag = time.perf_counter()
                            if now_diag - last_diag_time >= 1.0:
                                last_diag_time = now_diag
                                self.camera_diag_text.value = f"Frames: {frames_sent_count} sent"
                                self._safe_update(self.camera_diag_text)
                except Exception as exc:
                    capture_err_count += 1
                    if capture_err_count % 30 == 1:
                        self.camera_diag_text.value = f"Cam Err: {type(exc).__name__}"
                        self._safe_update(self.camera_diag_text)
                finally:
                    self._is_capturing = False

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
            except Exception:
                pass

        threading.Thread(target=_tts, daemon=True).start()

    def _toggle_speech(self, e: ft.ControlEvent) -> None:
        self.is_speech_enabled = not self.is_speech_enabled
        self.speech_button.icon = (
            ft.Icons.VOLUME_UP_ROUNDED if self.is_speech_enabled else ft.Icons.VOLUME_OFF_ROUNDED
        )
        self.speech_button.icon_color = (
            ACCENT_MINT if self.is_speech_enabled else TEXT_MUTED
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

        now_ts = time.perf_counter()
        if qwen_trans:
            self.gesture_output_text.value = f'"{qwen_trans}"'
            if qwen_trans != self._last_spoken_sentence:
                self._last_spoken_sentence = qwen_trans
                self._speak_text(qwen_trans)
        elif ribbon:
            self.gesture_output_text.value = f'"{" ".join(ribbon)}"'
        elif status == "PREDICTION" and payload.get("word"):
            word = payload["word"]
            conf = payload.get("confidence", 0.0)
            self.last_pred_time = now_ts
            self.gesture_output_text.value = f'"{word}"'
            self.confidence_text.value = f"Confidence: {conf * 100:.1f}%"
            if not qwen_trans and word != self.last_commit_word:
                self.last_commit_word = word
                self._speak_text(word)
        elif not hands_present:
            if (now_ts - self.last_pred_time) >= self.HOLD_DURATION:
                self.gesture_output_text.value = '"Raise hands into camera view"'
                self.confidence_text.value = "Confidence: 0.0%"
        elif not is_signing:
            if (now_ts - self.last_pred_time) >= self.HOLD_DURATION:
                self.gesture_output_text.value = '"Lift hands into signing space"'

        self._safe_update(self.prediction_card)

    async def cleanup_async(self):
        self._is_active = False
        if hasattr(self, "_capture_task") and self._capture_task:
            self._capture_task.cancel()
        if hasattr(self, "streamer") and self.streamer:
            self.streamer.stop()

    def _on_camera_flip(self, new_direction: fc.CameraLensDirection):
        self.is_front_camera = (new_direction == fc.CameraLensDirection.FRONT)
        self.front_cam_label.value = "Front Cam" if self.is_front_camera else "Back Cam"
        self._safe_update(self.front_cam_btn)
