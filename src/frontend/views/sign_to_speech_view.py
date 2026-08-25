import asyncio
from pathlib import Path
import flet as ft
import flet_camera as fc

from components.camera_view import CameraView
from components.vector_view import VectorView
from services.mobile_processor import MobileFrameProcessor
from services.hand_detector import HandDetector

# Absolute path resolution for the MediaPipe model binary
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")

# Standard layout viewport dimensions for the mobile camera container
VIEWPORT_WIDTH = 360
VIEWPORT_HEIGHT = 450


class SignToSpeechView(ft.View):
    def __init__(self, page: ft.Page):

        self.app_page = page
        super().__init__(
            route = '/sign-speech',
            padding = ft.Padding.only(top=15, left=15, right=15, bottom=15),
            horizontal_alignment = ft.CrossAxisAlignment.CENTER,
            vertical_alignment = ft.MainAxisAlignment.START,
            appbar = ft.AppBar(
                leading=ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda: asyncio.create_task(self.app_page.push_route("/"))),
                title=ft.Text("Sign To Speech"),
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
        
        # --- Services ---
        self.detector = HandDetector(
            model_path=MODEL_PATH, num_hands=2, HandDisplayView=self.hand_display
        )
        self.processor = MobileFrameProcessor(
            camera_control=self.camera_view_component.camera,
            detector=self.detector, target_fps=12,
            rotation_angle=90, flip_horizontal=True,
        )

        self.controls = [
            ft.Column(
                controls=[self.camera_view_component, self.hand_display],
                spacing=15,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
            )
         ]

        self.processor.start()


    async def cleanup_async(self):
        """ Call this before destroying the view to prevent memory leaks. """
        if self.processor:
            self.processor.stop()
        if self.detector:
            self.detector.close()
