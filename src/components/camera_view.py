import flet as ft
import flet_camera as fc
import flet_permission_handler as fh


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
    ) -> None:
        super().__init__()

        # Camera Configuration Settings
        self.camera_width: int = width
        self.camera_height: int = height
        self.camera_resolution: fc.ResolutionPreset = resolution
        self.camera_lens_direction: fc.CameraLensDirection = lens_direction

        # Native Camera & Permission Handler Objects
        self.camera: fc.Camera = fc.Camera(preview_enabled=True, expand=True)
        self.permission_handler = fh.PermissionHandler()

        # Viewport Surface Container
        self.camera_container = ft.Container(
            content=self.camera,
            width=self.camera_width,
            height=self.camera_height,
            bgcolor=ft.Colors.BLACK,
            border_radius=8,
        )

        # Status Label displaying initialization steps or permission errors
        self.status_text = ft.Text(
            value="Initializing camera....", color=ft.Colors.GREY_400
        )

        self.controls = [self.camera_container, self.status_text]
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.alignment = ft.MainAxisAlignment.START

    def did_mount(self) -> None:
        """Lifecycle hook called when component is attached to page tree."""
        self.page.run_task(self._setup_camera)

    async def _setup_camera(self) -> None:
        """Asynchronously requests camera permissions and initializes selected camera lens."""

        # Request OS camera permission
        has_permission: fh.PermissionStatus | None = (
            await self.permission_handler.request(fh.Permission.CAMERA)
        )

        if not has_permission:
            self.status_text.value = "Camera permission denied"
            self.status_text.color = ft.Colors.RED
            self.update()
            return

        self.status_text.value = "Checking available cameras..."
        self.update()
        cameras: list[fc.CameraDescription] = (
            await self.camera.get_available_cameras()
        )
        if not cameras:
            self.status_text.value = "No camera detected on this device"
            self.status_text.color = ft.Colors.RED
            self.update()
            return

        # Select target camera matching requested lens direction (e.g. Front camera)
        target_indx = 0
        for idx, cam in enumerate(cameras):
            if cam.lens_direction == self.camera_lens_direction:
                target_indx: int = idx
                break
        try:
            self.status_text.value = "Initiliazing camera..."
            self.update()
            await self.camera.initialize(
                description=cameras[target_indx],
                resolution_preset=self.camera_resolution,
            )

            self.status_text.value = "Camera ready."
            self.status_text.color = ft.Colors.GREEN_500
            self.update()
        except Exception as err:
            self.status_text.value = f"Initilization Error: {err}"
            self.status_text.color = ft.Colors.RED
            self.update()