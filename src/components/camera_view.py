import asyncio
import time
import flet as ft
import flet_camera as fc
import flet_permission_handler as fh
from services.gesture_processor import GestureProcessor


class CameraView(ft.Column):
    def __init__(
        self,
        width: int,
        height: int,
        resolution: fc.ResolutionPreset = fc.ResolutionPreset.MEDIUM,
        lens_direction: fc.CameraLensDirection = fc.CameraLensDirection.FRONT,
    ) -> None:
        super().__init__()

        # Camera Properties
        self.camera_width: int = width
        self.camera_height: int = height
        self.camera_resolution: fc.ResolutionPreset = resolution
        self.camera_lens_direction: fc.CameraLensDirection = lens_direction

        # Gesture Processor Instance
        self.processor = GestureProcessor(sequence_length=30)
        self.is_processing = False

        # Camera UI Elements
        self.camera: fc.Camera = fc.Camera(preview_enabled=True, expand=True)
        self.permission_handler = fh.PermissionHandler()

        self.camera_container = ft.Container(
            content=self.camera,
            width=self.camera_width,
            height=self.camera_height,
            bgcolor=ft.Colors.BLACK,
            border_radius=8,
        )

        self.status_text = ft.Text(value="Initializing camera....", color=ft.Colors.GREY_400)

        # --- DEDICATED SCROLLABLE VECTOR TEST PANEL ---
        self.vector_title = ft.Text(
            value="Live Landmark Vector (126 Floats)",
            color=ft.Colors.BLUE_400,
            size=13,
            weight=ft.FontWeight.BOLD,
        )

        self.vector_text = ft.Text(
            value="Waiting for hand landmarks...",
            color=ft.Colors.GREEN_300,
            size=11,
            font_family="monospace",
        )

        self.vector_box = ft.Container(
            content=ft.Column(
                controls=[self.vector_text],
                scroll=ft.ScrollMode.ALWAYS,
            ),
            width=self.camera_width,
            height=160,  # Scrollable debug window height
            bgcolor=ft.Colors.BLACK54,
            border_radius=8,
            padding=10,
            border=ft.border.all(1, ft.Colors.BLUE_900),
        )

        # Mount UI Elements to page
        self.controls = [
            self.camera_container,
            self.status_text,
            self.vector_title,
            self.vector_box,
        ]
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.alignment = ft.MainAxisAlignment.START

    def did_mount(self) -> None:
        self.page.run_task(self._setup_camera)

    async def _setup_camera(self) -> None:
        has_permission: fh.PermissionStatus | None = self.permission_handler.request(fh.Permission.CAMERA)

        if not has_permission:
            self.status_text.value = "Camera permission denied"
            self.status_text.color = ft.Colors.RED
            self.update()
            return

        self.status_text.value = "Checking available cameras..."
        self.update()

        cameras: list[fc.CameraDescription] = await self.camera.get_available_cameras()
        if not cameras:
            self.status_text.value = "No camera detected on this device"
            self.status_text.color = ft.Colors.RED
            self.update()
            return

        target_indx = 0
        for idx, cam in enumerate(cameras):
            if cam.lens_direction == self.camera_lens_direction:
                target_indx = idx
                break

        try:
            self.status_text.value = "Initializing camera..."
            self.update()

            await self.camera.initialize(
                description=cameras[target_indx],
                resolution_preset=self.camera_resolution,
            )

            self.status_text.value = "Camera ready."
            self.status_text.color = ft.Colors.GREEN_500
            self.update()

            # Start real-time gesture extraction loop
            self.is_processing = True
            self.page.run_task(self._start_gesture_processing)

        except Exception as err:
            self.status_text.value = f"Initialization Error: {err}"
            self.status_text.color = ft.Colors.RED
            self.update()

    async def _start_gesture_processing(self) -> None:
        """Loop updating the live scrollable vector display in real time."""
        target_fps = 15
        frame_interval = 1.0 / target_fps

        while self.is_processing:
            start_time = time.time()
            try:
                image_path = await self.camera.take_picture()

                if image_path:
                    sequence_matrix = self.processor.process_image_path(image_path)
                    latest_vector = sequence_matrix[-1]  # Extract current 126 float array

                    # Check if any hand was detected (non-zero array entries)
                    is_hand_detected = any(v != 0.0 for v in latest_vector)

                    if is_hand_detected:
                        h1_vals = latest_vector[:63]
                        h2_vals = latest_vector[63:]

                        lines = ["=== HAND 1 (21 Joints x,y,z) ==="]
                        for i in range(0, 63, 3):
                            j_idx = i // 3
                            lines.append(
                                f"J{j_idx:02d}: x={h1_vals[i]:+.3f}, y={h1_vals[i+1]:+.3f}, z={h1_vals[i+2]:+.3f}"
                            )

                        lines.append("\n=== HAND 2 (21 Joints x,y,z) ===")
                        for i in range(0, 63, 3):
                            j_idx = i // 3
                            lines.append(
                                f"J{j_idx:02d}: x={h2_vals[i]:+.3f}, y={h2_vals[i+1]:+.3f}, z={h2_vals[i+2]:+.3f}"
                            )

                        self.vector_text.value = "\n".join(lines)
                        self.vector_text.color = ft.Colors.GREEN_300
                    else:
                        self.vector_text.value = "No hands detected in frame.\nVector filled with 126 zeros [0.00, ...]"
                        self.vector_text.color = ft.Colors.AMBER_300

                    self.update()

            except Exception as e:
                print(f"Frame processing error: {e}")

            elapsed = time.time() - start_time
            sleep_duration = max(0.01, frame_interval - elapsed)
            await asyncio.sleep(sleep_duration)