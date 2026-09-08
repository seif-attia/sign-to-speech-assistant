import asyncio
from pathlib import Path
import threading
import time
from typing import Optional
import flet as ft
import flet_camera as fc

from components.camera_view import CameraView
from components.vector_view import VectorView
from services.mobile_processor import MobileFrameProcessor
from services.holistic_detector import HolisticDetector
from services.sign_to_text_model import SignToTextClassifier
from frontend.components.bottom_nav_bar import create_nav_bar

# Absolute path resolution for model binaries and assets
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(BASE_DIR / "holistic_landmarker.task")
HOLISTIC_SIGN_MODEL_PATH = BASE_DIR / "sign_language_holistic_model.tflite"
SIGN_MODEL_PATH = str(
    HOLISTIC_SIGN_MODEL_PATH
    if HOLISTIC_SIGN_MODEL_PATH.exists()
    else (BASE_DIR / "sign_language_model.tflite")
)
LABELS_PATH = str(BASE_DIR / "labels.json")

# Standard layout viewport dimensions for the mobile camera container
VIEWPORT_WIDTH = 360
VIEWPORT_HEIGHT = 380


class SignToSpeechView(ft.View):
    """
    Sign language translation camera view.
    Displays real-time camera feed, MediaPipe landmark vectors,
    smooth gesture recognition with voting confidence, and speech synthesis.
    """

    def __init__(self, page: ft.Page, use_mock: bool = False):
        self.app_page = page
        self.is_speech_enabled: bool = True
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
            ),
            navigation_bar=create_nav_bar(1, self.app_page),
        )

        # --- UI Components ---
        self.camera_view_component = CameraView(
            width=VIEWPORT_WIDTH,
            height=VIEWPORT_HEIGHT,
            resolution=fc.ResolutionPreset.MEDIUM,
            lens_direction=fc.CameraLensDirection.FRONT,
        )
        self.hand_display = VectorView(width=VIEWPORT_WIDTH - 30)

        # Dedicated UI display elements (avoiding status/prediction clobbering)
        self.gesture_icon = ft.Icon(ft.Icons.FRONT_HAND, color=ft.Colors.BLUE, size=28)
        self.prediction_text = ft.Text(
            "Waiting for gesture...",
            size=18,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.BLUE_900,
        )
        self.confidence_text = ft.Text(
            "Perform sign inside camera frame",
            size=12,
            color=ft.Colors.GREY_700,
        )
        self.status_bar_text = ft.Text(
            "Status: Initializing...",
            size=11,
            color=ft.Colors.GREY_600,
            italic=True,
        )

        # TTS Speech Toggle Button
        self.speech_button = ft.IconButton(
            icon=ft.Icons.VOLUME_UP,
            icon_color=ft.Colors.BLUE_700,
            tooltip="Mute / Unmute Speech",
            on_click=self._toggle_speech,
        )

        # Mock Mode indicator badge
        self.mock_badge = ft.Container(
            content=ft.Text(
                "MOCK MODE",
                size=9,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.ORANGE_900,
            ),
            bgcolor=ft.Colors.ORANGE_100,
            border=ft.border.all(1, ft.Colors.ORANGE_300),
            border_radius=4,
            padding=ft.Padding.symmetric(horizontal=6, vertical=2),
            visible=False,
        )

        # Unified recognition feedback card
        self.prediction_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.gesture_icon,
                            ft.Column(
                                controls=[
                                    ft.Row(
                                        controls=[
                                            self.prediction_text,
                                            self.mock_badge,
                                        ],
                                        alignment=ft.MainAxisAlignment.START,
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                        spacing=6,
                                    ),
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

        # --- ML Inference Service ---
        self.classifier = SignToTextClassifier(
            model_path=SIGN_MODEL_PATH,
            labels_path=LABELS_PATH,
            threshold=0.50,
            voting_window_size=5,
            min_voting_count=3,
            cooldown_seconds=1.5,
            max_missing_frames=5,
            use_mock=use_mock,
            on_prediction=self._on_sign_recognized,
            on_status=self._on_status_update,
        )

        # Display mock badge if mock runtime was activated
        if self.classifier.is_mock:
            self.mock_badge.visible = True
            self.status_bar_text.value = "Status: Mock predictor active"

        # --- Hardware Capture & Detection Services ---
        self.detector = HolisticDetector(
            model_path=MODEL_PATH,
            HandDisplayView=self.hand_display,
            classifier=self.classifier,
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
            )
        ]

        self.processor.start()

    def _safe_update(self, control: ft.Control) -> None:
        """Thread-safe UI control update with lifecycle exception protection."""
        try:
            if self.app_page:
                control.update()
        except Exception:
            pass

    def _on_sign_recognized(self, label: str, confidence: float) -> None:
        """
        Callback invoked when a gesture passes temporal voting & cooldown.
        Updates UI and triggers speech synthesis without being wiped by status events.
        """
        self.prediction_text.value = f"🎯 {label.upper()}"
        self.prediction_text.color = ft.Colors.GREEN_800
        self.confidence_text.value = (
            f"Confidence: {confidence * 100:.1f}% (Voting Verified)"
        )
        self.status_bar_text.value = f"Recognized '{label}' • Ready for next sign"

        self._safe_update(self.prediction_card)

        # Trigger Speech Output
        self._speak_text(label)

    def _on_status_update(self, status_msg: str) -> None:
        """
        Callback invoked during buffering or hand tracking state changes.
        Updates the dedicated status bar without wiping out recognized predictions.
        """
        self.status_bar_text.value = status_msg

        # Only reset top label if no gesture has ever been recognized yet
        if self.prediction_text.value == "Waiting for gesture...":
            self.confidence_text.value = status_msg

        self._safe_update(self.prediction_card)

    def _toggle_speech(self, e: ft.ControlEvent) -> None:
        """Toggles Text-To-Speech audio feedback on or off."""
        self.is_speech_enabled = not self.is_speech_enabled
        self.speech_button.icon = (
            ft.Icons.VOLUME_UP if self.is_speech_enabled else ft.Icons.VOLUME_OFF
        )
        self.speech_button.icon_color = (
            ft.Colors.BLUE_700 if self.is_speech_enabled else ft.Colors.GREY_500
        )
        self._safe_update(self.speech_button)

    def _speak_text(self, text: str) -> None:
        """Asynchronously synthesizes speech for the recognized sign label."""
        if not self.is_speech_enabled:
            return

        now = time.time()
        if (now - self._last_spoken_time) < 1.0:
            return
        self._last_spoken_time = now

        def _tts_thread():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                # Fallback if pyttsx3 or audio drivers are not configured
                pass

        threading.Thread(target=_tts_thread, daemon=True).start()

    async def cleanup_async(self):
        """Releases all camera, hardware, and classification handles on view teardown."""
        if self.processor:
            self.processor.stop()
        if self.detector:
            self.detector.close()
        if self.classifier:
            self.classifier.close()
