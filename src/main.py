import asyncio
from pathlib import Path
import flet as ft
import flet_camera as fc
from components.camera_view import CameraView
from components.vector_view import VectorView
from services.mobile_processor import MobileFrameProcessor
from services.hand_detector import HandDetector

# Absolute path resolution for the MediaPipe model binary
BASE_DIR = Path(__file__).parent
MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")

# Standard layout viewport dimensions for the mobile camera container
VIEWPORT_WIDTH = 360
VIEWPORT_HEIGHT = 450


async def main(page: ft.Page) -> None:
    # --- Page Configuration ---
    page.title = "Mobile Hand Tracker"
    # Padding with top inset clearance for status bar/camera notch
    page.padding = ft.Padding.only(top=50, left=15, right=15, bottom=15)
    page.theme_mode = ft.ThemeMode.DARK
    page.keep_screen_on = True  # Prevents display dimming or sleeping during active camera tracking
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.START

    # --- UI Components Initialization ---
    # 1. Low-level camera view managing hardware drivers and permissions
    camera_view_component = CameraView(
        width=VIEWPORT_WIDTH,
        height=VIEWPORT_HEIGHT,
        resolution=fc.ResolutionPreset.MEDIUM,
        lens_direction=fc.CameraLensDirection.FRONT,
    )


    # 3. Textual landmark vector view component
    hand_display = VectorView(width=VIEWPORT_WIDTH - 30)

    # --- Service & Processor Instances ---
    detector = HandDetector(
        model_path=MODEL_PATH,
        num_hands=2,
        HandDisplayView= hand_display
    )

    processor = MobileFrameProcessor(
        camera_control=camera_view_component.camera,
        detector=detector,
        target_fps=12,
        rotation_angle=90,      # Corrects portrait orientation on mobile camera sensors
        flip_horizontal=True,   # Applies selfie camera mirror effect
    )
    
    # --- View Tree Mounting ---
    page.add(
        ft.Column(
            controls=[
                camera_view_component,  # Top viewport window
                hand_display,    # Bottom expanding data window
            ],
            spacing=15,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.START,
            expand=True,
        )
    )

    page.update()

    # Start non-blocking frame capture and inference loop
    processor.start()

    # Handle Cleaning

    async def on_disconnect(e) -> None:
        """Clean shutdown handler to stop processing loops and release MediaPipe memory."""
        processor.stop()
        detector.close()


    page.on_disconnect = on_disconnect

if __name__ == "__main__":
    ft.run(main)    