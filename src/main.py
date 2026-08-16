import asyncio
from pathlib import Path
import flet as ft
import flet_camera as fc
from components.camera_view import CameraView
from components.vector_view import VectorView
from mobile_processor import MobileFrameProcessor
from services.hand_detector import HandDetector

# Absolute path resolution for the MediaPipe model binary
BASE_DIR = Path(__file__).parent
MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")

# 1x1 transparent PNG Base64 string used to initialize the Image control
BLANK_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

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

    # 2. Overlay canvas that renders processed, downscaled Base64 image frames
    camera_preview = ft.Image(
        src=BLANK_DATA_URI,
        fit="cover",
        width=VIEWPORT_WIDTH,
        height=VIEWPORT_HEIGHT,
    )

    # Top Window: Stacked layout putting the canvas overlay directly over the native driver
    camera_window = ft.Container(
        content=ft.Stack(
            controls=[
                camera_view_component,  # Native hardware surface driver
                camera_preview,         # Rendered Base64 stream overlay
            ],
            width=VIEWPORT_WIDTH,
            height=VIEWPORT_HEIGHT,
        ),
        border_radius=12,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        border=ft.Border.all(1, "#333333"),
    )

    # 3. Textual landmark vector view component
    hand_display = VectorView(width=VIEWPORT_WIDTH - 30)

    # Bottom Window: Output container configured to expand vertically
    data_window = ft.Container(
        content=hand_display,
        width=VIEWPORT_WIDTH,
        padding=15,
        bgcolor="#181818",
        border_radius=12,
        border=ft.Border.all(1, "#333333"),
        expand=True,  # Automatically stretches down to fill remaining screen space
    )

    # --- Callbacks ---
    def on_hand_detected(result) -> None:
        """
        Processes MediaPipe HandLandmarker results into a fixed 126-dimensional float vector.
        Vector layout: 2 hands * 21 landmarks * 3 coordinates (x, y, z).
        Missing hands or landmarks are zero-padded to maintain consistent dimensionality.
        """
        vector = [] #<------ THE DATA LIVES IN THIS VECTOR
        num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

        for hand_idx in range(2):
            if result.hand_landmarks and hand_idx < num_hands:
                for lm in result.hand_landmarks[hand_idx]:
                    vector.extend([round(lm.x, 3), round(lm.y, 3), round(lm.z, 3)])
            else:
                # Pad missing hand slot with 63 zeros (21 landmarks * 3 coordinates)
                vector.extend([0.0] * 63)

        formatted_vals = ", ".join(f"{v:.3f}" for v in vector)
        output_text = (
            f"Hands: {num_hands} | Vector Dim: {len(vector)}\n\n"
            f"[{formatted_vals}]"
        )

        hand_display.update_data(output_text)

    def on_frame_captured(b64_str: str) -> None:
        """Updates the Base64 image source and triggers a UI frame render."""
        camera_preview.src = f"data:image/jpeg;base64,{b64_str}"
        page.update()

    # --- Service & Processor Instances ---
    detector = HandDetector(
        model_path=MODEL_PATH,
        num_hands=2,
        ui_callback=on_hand_detected,
    )

    processor = MobileFrameProcessor(
        camera_control=camera_view_component.camera,
        detector=detector,
        frame_callback=on_frame_captured,
        target_fps=12,
        rotation_angle=90,      # Corrects portrait orientation on mobile camera sensors
        flip_horizontal=True,   # Applies selfie camera mirror effect
    )

    async def on_disconnect(e) -> None:
        """Clean shutdown handler to stop processing loops and release MediaPipe memory."""
        processor.stop()
        detector.close()

    page.on_disconnect = on_disconnect

    # --- View Tree Mounting ---
    page.add(
        ft.Column(
            controls=[
                camera_window,  # Top viewport window
                data_window,    # Bottom expanding data window
            ],
            spacing=15,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.START,
            expand=True,
        )
    )

    page.update()

    # --- Initialization Wait Loop ---
    # Blocks stream execution until the underlying native camera driver reports readiness
    while camera_view_component.status_text.value != "Camera ready.":
        if (
            "Error" in camera_view_component.status_text.value
            or "denied" in camera_view_component.status_text.value
        ):
            print(
                f"[Main] Camera setup stopped: {camera_view_component.status_text.value}"
            )
            return
        await asyncio.sleep(0.5)

    # Start non-blocking frame capture and inference loop
    processor.start()


ft.app(target=main)