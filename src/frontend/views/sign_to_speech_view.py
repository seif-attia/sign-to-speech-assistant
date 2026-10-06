import asyncio
import io
import logging
import os
import threading
import time
import flet as ft
import flet_camera as fc
import flet_audio as fa


from components.camera_view import CameraView
from components.vector_view import VectorView
from services.config import (
    get_server_host,
)
from services.network_stream_service import get_global_streamer
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
        self.target_lang: str = "en"  # "en" or "ar"

        # Speech Synthesis State
        self.is_speech_enabled: bool = True
        self._last_spoken_word: str = ""
        self._last_spoken_sentence: str = ""
        self._last_spoken_time: float = 0.0

        async def close(e: ft.ControlEvent):
            await self.cleanup_async()
            await self.app_page.push_route("/")

        # Top Bar: Left '← Modes', Right: Language Dropdown + Server IP dialog + Front Cam toggle
        left_modes_btn = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ARROW_BACK_IOS_NEW, size=13, color=ACCENT_MINT),
                    ft.Text("Modes", size=13, weight=ft.FontWeight.W_500, color=ACCENT_MINT),
                ],
                spacing=4,
            ),
            on_click=close,
            padding=ft.Padding.symmetric(horizontal=8, vertical=20),
            border_radius=8,
            bgcolor="#132323",
        )

        def _on_lang_dropdown_change(e):
            self.target_lang = self.lang_dropdown.value
            logger.info(f"Sign to speech target language set to: {self.target_lang}")

        self.lang_dropdown = ft.Dropdown(
        value="en",
        options=[
            ft.dropdown.Option("en", "English"),
            ft.dropdown.Option("ar", "العربية"),
        ],
        on_select=_on_lang_dropdown_change,
        width=105,
        # Changed from ft.Padding.symmetric to lowercase ft.padding.symmetric
        content_padding=ft.Padding.symmetric(horizontal=6, vertical=2), 
        text_size=12,
        border_radius=8,
        border_color=CARD_BORDER,
        bgcolor="#132323",
        color=ACCENT_MINT,
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
                        self.lang_dropdown,
                        self.front_cam_btn,
                    ],
                    spacing=8,
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

        # 3. Persistent Global WebSocket Streamer
        self.streamer = get_global_streamer()
        self.streamer.register_result_callback(self._handle_server_prediction)
        self.streamer.register_status_callback(self._handle_server_status)

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

    def _handle_server_status(self, status: str) -> None:
        self.status_bar_text.value = f"Status: {status[:12]}"
        self._safe_update(self.prediction_card)

    def _process_and_compress_frame(self, raw_input) -> bytes:
        try:
            # If PIL is available, use it for optimal downsampling & EXIF orientation
            try:
                from PIL import Image, ImageOps
                _has_pil = True
            except ImportError:
                _has_pil = False

            if isinstance(raw_input, (bytes, bytearray)):
                if not _has_pil:
                    return bytes(raw_input)
                img = Image.open(io.BytesIO(raw_input))
            elif isinstance(raw_input, str) and os.path.exists(raw_input):
                if not _has_pil:
                    with open(raw_input, "rb") as f:
                        return f.read()
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
            if isinstance(raw_input, (bytes, bytearray)):
                return bytes(raw_input)
            elif isinstance(raw_input, str) and os.path.exists(raw_input):
                try:
                    with open(raw_input, "rb") as f:
                        return f.read()
                except Exception:
                    pass
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
                                target_lang=self.target_lang,
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

        async def _speak():
            try:
                from services.fastapi_endpoint import play_speech
                await play_speech(self.app_page, text, lang=self.target_lang)
            except Exception as e:
                logger.error(f"Error during speech playback: {e}")

        self.app_page.run_task(_speak)

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
                # Tell backend to clear sentence ribbon now that it was spoken
                self.streamer.send_command("clear_sentence")
                # Reset UI display after speaking
                async def _clear_display_after_speak():
                    await asyncio.sleep(2.0)
                    if self._is_active and self.gesture_output_text.value == f'"{qwen_trans}"':
                        self.gesture_output_text.value = '"Raise hands into camera view"'
                        self._safe_update(self.prediction_card)
                self.app_page.run_task(_clear_display_after_speak)
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
            self.streamer.unregister_result_callback(self._handle_server_prediction)
            self.streamer.unregister_status_callback(self._handle_server_status)

    def _on_camera_flip(self, new_direction: fc.CameraLensDirection):
        self.is_front_camera = (new_direction == fc.CameraLensDirection.FRONT)
        self.front_cam_label.value = "Front Cam" if self.is_front_camera else "Back Cam"
        self._safe_update(self.front_cam_btn)
