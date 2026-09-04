import asyncio
from pathlib import Path
import flet as ft
import flet_camera as fc

from components.camera_view import CameraView
from components.vector_view import VectorView
from services.mobile_processor import MobileFrameProcessor
from services.hand_detector import HandDetector
from services.sign_to_text_model import SignToTextClassifier

# Absolute path resolution for model binaries and assets
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")
SIGN_MODEL_PATH = str(BASE_DIR / "sign_language_model.tflite")
LABELS_PATH = str(BASE_DIR / "labels.json")

# Standard layout viewport dimensions for the mobile camera container
VIEWPORT_WIDTH = 360
VIEWPORT_HEIGHT = 380


class SignToSpeechView(ft.View):
    def __init__(self, page: ft.Page):

        self.app_page = page
        super().__init__(
            route = '/sign-speech',
            padding = ft.Padding.only(top=15, left=15, right=15, bottom=15),
            horizontal_alignment = ft.CrossAxisAlignment.CENTER,
            vertical_alignment = ft.MainAxisAlignment.START,
            appbar = ft.AppBar(
                leading=ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda e: asyncio.create_task(self.close())),
                title=ft.Text("Sign To Speech", color= ft.Colors.WHITE),
                bgcolor=ft.Colors.BLUE
            ),
        )
      
        # --- UI Components ---
        self.camera_view_component = CameraView(
            width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT,
            resolution=fc.ResolutionPreset.MEDIUM,
            lens_direction=fc.CameraLensDirection.FRONT,
        )
        self.hand_display = VectorView(width=VIEWPORT_WIDTH - 30)

        # Recognition feedback card
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
        self.prediction_card = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.FRONT_HAND, color=ft.Colors.BLUE, size=28),
                    ft.Column(
                        controls=[
                            self.prediction_text,
                            self.confidence_text,
                        ],
                        spacing=2,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
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
            threshold=0.65,
            on_prediction=self._on_sign_recognized,
        )

        # --- Hardware Capture & Detection Services ---
        self.detector = HandDetector(
            model_path=MODEL_PATH,
            num_hands=2,
            HandDisplayView=self.hand_display,
            classifier=self.classifier,
        )
        self.processor = MobileFrameProcessor(
            camera_control=self.camera_view_component.camera,
            detector=self.detector, target_fps=12,
            rotation_angle=90, flip_horizontal=True,
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

    def _on_sign_recognized(self, label: str, confidence: float) -> None:
        """Callback invoked when high-confidence sign is inferred."""
        self.prediction_text.value = label.upper()
        self.confidence_text.value = f"Confidence: {confidence * 100:.1f}%"
        try:
            self.app_page.update()
        except Exception:
            pass

    async def close(self):
        await self.cleanup_async()
        await self.app_page.push_route("/")


    async def cleanup_async(self):
        """ Call this before destroying the view to prevent memory leaks. """
        if self.processor:
            self.processor.stop()
        if self.detector:
            self.detector.close()
        if self.classifier:
            self.classifier.close()
