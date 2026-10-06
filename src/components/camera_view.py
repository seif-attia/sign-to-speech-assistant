import flet as ft
import flet_camera as fc
import flet_permission_handler as fh
from typing import Callable


class CameraView(ft.Column):
    """
    Custom UI component wrapping the flet_camera control, managing permissions,
    device camera discovery, driver initialization, and status updates.
    """

    def __init__(
        self,
        width: int,
        height: int,
        resolution: fc.ResolutionPreset = fc.ResolutionPreset.MEDIUM,
        lens_direction: fc.CameraLensDirection = fc.CameraLensDirection.FRONT,
        on_lens_change: Callable[[fc.CameraLensDirection], None] | None = None
    ) -> None:
        super().__init__()

        # Camera Configuration Settings
        self.camera_width: int = width
        self.camera_height: int = height
        self.camera_resolution: fc.ResolutionPreset = resolution
        self.camera_lens_direction: fc.CameraLensDirection = lens_direction
        self.on_lens_change = on_lens_change

        # Native Camera & Permission Handler Objects
        self.camera: fc.Camera = fc.Camera(preview_enabled=True, expand=True)
        self.permission_handler = fh.PermissionHandler()
        self.available_cameras: list[fc.CameraDescription] = []


        # Camera flip button
        self.camera_flip_btn: ft.IconButton = ft.IconButton(icon=ft.Icons.FLIP_CAMERA_ANDROID_ROUNDED, on_click= self.flip_camera)  

        # Viewport Surface Container
        self.camera_container = ft.Container(
            content=self.camera,
            width=self.camera_width,
            height=self.camera_height,
            bgcolor=ft.Colors.BLACK,
            border_radius=8,
        )

        self.is_camera_ready: bool = False

        # Status Label displaying initialization steps or permission errors
        self.status_text: ft.Text = ft.Text(
            value="Initializing camera....", color=ft.Colors.GREY_400
        )

        self.controls = [
            self.camera_container, 
            ft.Row(controls=[self.camera_flip_btn, self.status_text], alignment= ft.MainAxisAlignment.SPACE_BETWEEN)
        ]
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.alignment = ft.MainAxisAlignment.START

    def did_mount(self) -> None:
        """Lifecycle hook called when component is attached to page tree."""
        self.page.run_task(self._setup_camera)

    async def _setup_camera(self) -> None:
        """Asynchronously requests camera permissions and initializes selected camera lens."""
        self.is_camera_ready = False
        print("[CAMERA SETUP] Requesting OS camera permission...", flush=True)

        # Request OS camera permission
        try:
            has_permission: fh.PermissionStatus | None = (
                await self.permission_handler.request(fh.Permission.CAMERA)
            )
            print(f"[CAMERA SETUP] Camera permission response: {has_permission}", flush=True)
        except Exception as e:
            print(f"[CAMERA SETUP] Permission handler error: {e}", flush=True)
            has_permission = None

        if not has_permission:
            self.status_text.value = "Camera permission denied"
            self.status_text.color = ft.Colors.RED
            self.update()
            return

        self.status_text.value = "Checking available cameras..."
        self.update()
        try:
            self.available_cameras = await self.camera.get_available_cameras()
            print(f"[CAMERA SETUP] Available cameras discovered: {len(self.available_cameras)}", flush=True)
        except Exception as e:
            print(f"[CAMERA SETUP] Camera detection error: {e}", flush=True)
            self.status_text.value = f"Camera detection failed: {e}"
            self.status_text.color = ft.Colors.RED
            self.update()
            return

        if not self.available_cameras:
            print("[CAMERA SETUP] No camera detected on this device!", flush=True)
            self.status_text.value = "No camera detected on this device"
            self.status_text.color = ft.Colors.RED
            self.update()
            return

        # Select target camera matching requested lens direction (e.g. Front camera)
        target_indx = 0
        for idx, cam in enumerate(self.available_cameras):
            if cam.lens_direction == self.camera_lens_direction:
                target_indx: int = idx
                break
        try:
            self.status_text.value = "Initializing camera..."
            self.update()
            print(f"[CAMERA SETUP] Initializing camera {target_indx} ({self.available_cameras[target_indx]})...", flush=True)
            await self.camera.initialize(
                description=self.available_cameras[target_indx],
                resolution_preset=self.camera_resolution,
            )

            self.is_camera_ready = True
            self.status_text.value = "Camera ready."
            self.status_text.color = ft.Colors.GREEN_500
            self.update()
            print("[CAMERA SETUP] Camera ready!", flush=True)
        except Exception as err:
            self.is_camera_ready = False
            self.status_text.value = f"Initialization Error: {err}"
            self.status_text.color = ft.Colors.RED
            self.update()
            print(f"[CAMERA SETUP ERROR] {err}", flush=True)

    async def flip_camera(self, e=None):

        target_direction = (
            fc.CameraLensDirection.FRONT 
            if self.camera_lens_direction == fc.CameraLensDirection.BACK 
            else fc.CameraLensDirection.BACK)

        target_cam = next((cam for cam in self.available_cameras if cam.lens_direction == target_direction), None)


        self.camera_flip_btn.disabled = True
        self.is_camera_ready = False
        self.status_text.value = "Switching Camera..."
        self.status_text.color = ft.Colors.BLUE_700 
        self.update()

        try:
            await self.camera.set_description(target_cam)
            self.camera_lens_direction = target_direction
            if self.on_lens_change:
                self.on_lens_change(target_direction)
            self.is_camera_ready = True
            self.status_text.value = "Camera ready."
            self.status_text.color = ft.Colors.GREEN_500
        except Exception as err:
            self.is_camera_ready = False
            self.status_text.value = f"Couldn't switch cameras: {err}"
            self.status_text.color = ft.Colors.RED
        finally:
            self.camera_flip_btn.disabled = False
            self.update()

        