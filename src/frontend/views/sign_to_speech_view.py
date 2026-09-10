import asyncio
from pathlib import Path
import threading
import time
import flet as ft
import flet_camera as fc

from components.camera_view import CameraView
from components.vector_view import VectorView
from services.mobile_processor import MobileFrameProcessor
from services.holistic_detector import HolisticDetector
from services.sign_to_text_model import SignPredictor
from frontend.components.bottom_nav_bar import create_nav_bar

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
SIGN_MODEL_PATH = str(BASE_DIR / "sign_language_top30_model.tflite" if (BASE_DIR / "sign_language_top30_model.tflite").exists() else BASE_DIR / "sign_language_holistic_model.tflite")
LABELS_PATH = str(BASE_DIR / "labels.json" if (BASE_DIR / "labels.json").exists() else BASE_DIR / "label.json")

VIEWPORT_WIDTH = 360
VIEWPORT_HEIGHT = 380


class SignToSpeechView(ft.View):
    def __init__(self, page: ft.Page):
        self.app_page = page
        self.is_speech_enabled: bool = True
        self._last_spoken_time: float = 0.0
        self._is_active: bool = True

        # State Variables from test script
        self.sentence_ribbon: list[str] = []
        self.last_commit_word: str = ""
        self.last_commit_time: float = 0.0
        self.last_pred_time: float = 0.0

        self.HOLD_DURATION = 1.2
        self.COMMIT_THRESHOLD = 0.50
        self.COMMIT_COOLDOWN = 1.4

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
            ),
            navigation_bar=create_nav_bar(1, self.app_page),
        )

        self.camera_view_component = CameraView(
            width=VIEWPORT_WIDTH,
            height=VIEWPORT_HEIGHT,
            resolution=fc.ResolutionPreset.MEDIUM,
            lens_direction=fc.CameraLensDirection.FRONT,
            on_lens_change=self._on_camera_flip,
        )
        self.hand_display = VectorView(width=VIEWPORT_WIDTH - 30)

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
            "SENTENCE: (waiting for signs...)",
            size=12,
            weight=ft.FontWeight.W_500,
            color=ft.Colors.BLUE_GREY_800,
        )
        self.status_bar_text = ft.Text(
            "Status: Ready",
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
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.INFO_OUTLINE, size=14, color=ft.Colors.GREY_500),
                            self.status_bar_text,
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

        # Initialize Predictor and 3-Task Detector
        self.predictor = SignPredictor(
            model_path=SIGN_MODEL_PATH,
            labels_path=LABELS_PATH,
        )
        self.detector = HolisticDetector(
            base_dir=str(BASE_DIR),
            HandDisplayView=self.hand_display,
            prediction_callback=self._on_frame_detected,
            flip_horizontal=True,
        )
        self.processor = MobileFrameProcessor(
            camera_control=self.camera_view_component.camera,
            detector=self.detector,
            target_fps=30,
            rotation_angle=90,
            flip_horizontal=True,
        )

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

        self.processor.start()

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

    def _on_frame_detected(self, frame_vector, hands_present: bool, is_signing: bool) -> None:
        if not self._is_active:
            return

        res = self.predictor.process_frame(frame_vector, hands_present, is_signing)
        status = res.get("status")

        if status == "WARMING":
            fill = res["buffer_fill"]
            tot = res["total_buffer"]
            self.prediction_text.value = f"WARMING [{fill}/{tot}]"
            self.prediction_text.color = ft.Colors.ORANGE_800
            self.confidence_text.value = "Buffer filling... raise hands and sign"
            self._safe_update(self.prediction_card)
            return

        elif status == "NO_HANDS":
            # Keep displaying last detected sign for HOLD_DURATION seconds
            hold_active = (time.perf_counter() - self.last_pred_time) < self.HOLD_DURATION
            if not hold_active:
                self.prediction_text.value = "STANDBY"
                self.prediction_text.color = ft.Colors.BLUE_900
                self.confidence_text.value = "Raise hands into camera view"
                self._safe_update(self.prediction_card)
            return

        elif status == "RESTING":
            hold_active = (time.perf_counter() - self.last_pred_time) < self.HOLD_DURATION
            if not hold_active:
                self.prediction_text.value = "RESTING"
                self.prediction_text.color = ft.Colors.BLUE_700
                self.confidence_text.value = "Lift hands into signing space"
                self._safe_update(self.prediction_card)
            return

        elif status == "PREDICTION":
            word = res["word"]
            conf = res["conf"]
            top5 = res["top5"]
            now_ts = time.perf_counter()

            self.last_pred_time = now_ts
            self.prediction_text.value = f"🎯 {word.upper()}"
            self.prediction_text.color = ft.Colors.GREEN_800 if conf >= 0.50 else ft.Colors.CYAN_900
            self.confidence_text.value = f"Confidence: {conf * 100:.1f}%"

            # Top candidate diagnostics
            if len(top5) > 1:
                alt = " | ".join([f"{w} ({c * 100:.0f}%)" for w, c in top5[1:4]])
                self.status_bar_text.value = f"Top: {alt}"

            # --- Anti-Duplication & Sentence Ribbon Commitment ---
            # 1. Meets confidence threshold (no need to wait for 90%)
            # 2. Cooldown elapsed since last word commit
            # 3. Not identical to the immediately preceding committed word
            is_new_word = len(self.sentence_ribbon) == 0 or self.sentence_ribbon[-1] != word
            cooldown_passed = (now_ts - self.last_commit_time) > self.COMMIT_COOLDOWN

            if conf >= self.COMMIT_THRESHOLD and is_new_word and cooldown_passed:
                self.sentence_ribbon.append(word)
                if len(self.sentence_ribbon) > 8:
                    self.sentence_ribbon = self.sentence_ribbon[-8:]

                self.last_commit_word = word
                self.last_commit_time = now_ts

                # Update UI ribbon & synthesize speech
                self.sentence_ribbon_text.value = f"SENTENCE: {' '.join(self.sentence_ribbon)}"
                self._speak_text(word)

                # Reset buffer so the tail of the current stroke doesn't re-trigger
                self.predictor.clear()

            self._safe_update(self.prediction_card)

    def _toggle_speech(self, e: ft.ControlEvent) -> None:
        self.is_speech_enabled = not self.is_speech_enabled
        self.speech_button.icon = ft.Icons.VOLUME_UP if self.is_speech_enabled else ft.Icons.VOLUME_OFF
        self.speech_button.icon_color = ft.Colors.BLUE_700 if self.is_speech_enabled else ft.Colors.GREY_500
        self._safe_update(self.speech_button)

    def _speak_text(self, text: str) -> None:
        if not self.is_speech_enabled or not self._is_active:
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

    async def cleanup_async(self):
        self._is_active = False
        if hasattr(self, "processor") and self.processor:
            self.processor.stop()
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
        if hasattr(self, "predictor") and self.predictor:
            self.predictor.clear()

    def _on_camera_flip(self, new_direction: fc.CameraLensDirection):
        is_front = new_direction == fc.CameraLensDirection.FRONT
        if hasattr(self, "processor") and self.processor:
            self.processor.flip_horizontal = is_front
        if hasattr(self, "detector") and self.detector:
            self.detector.flip_horizontal = is_front