import asyncio
import base64
import io
import os
import numpy as np
from PIL import Image


class MobileFrameProcessor:
    """
    Manages background frame capture, image transformations (orientation, mirror, downscaling),
    Base64 streaming, and forwarding RGB matrices to the MediaPipe inference engine.
    """

    def __init__(
        self,
        camera_control,
        detector,
        frame_callback=None,
        target_fps: int = 12,
        rotation_angle: int = 90,
        flip_horizontal: bool = True,
    ) -> None:
        self.camera = camera_control
        self.detector = detector
        self.frame_callback = frame_callback
        self.delay = 1.0 / target_fps
        self.rotation_angle = rotation_angle
        self.flip_horizontal = flip_horizontal
        self.is_running = False
        self._is_processing = False  # Lock flag to skip frame drops if processing lags
        self._task = None

    def _process_image_data(self, image_source):
        """
        Synchronous image processing pipeline executed off the main event loop via asyncio.to_thread.
        Handles file I/O cleanup, rotation, mirroring, low-res downscaling, and Base64 JPEG encoding.
        """
        pil_img = None
        # Load from disk cache and immediately delete temporary capture file
        if isinstance(image_source, str) and os.path.exists(image_source):
            try:
                with Image.open(image_source) as img:
                    pil_img = img.convert("RGB")
            finally:
                try:
                    os.remove(image_source)
                except OSError:
                    pass
        elif isinstance(image_source, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_source)).convert("RGB")

        if pil_img is None:
            return None, None

        # 1. Rotate frame to correct physical camera sensor landscape layout to portrait
        if self.rotation_angle != 0:
            pil_img = pil_img.rotate(self.rotation_angle, expand=True)

        # 2. Mirror horizontally for intuitive front-camera selfie preview
        if self.flip_horizontal:
            try:
                pil_img = pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            except AttributeError:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 3. Downscale frame to low resolution for fast MediaPipe processing and UI rendering
        pil_img = pil_img.resize((240, 320), Image.Resampling.NEAREST)
        rgb_array = np.array(pil_img)

        # 4. Compress to low-quality JPEG to minimize Base64 string IPC overhead
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=30, optimize=False)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return rgb_array, b64_str

    async def _process_loop(self) -> None:
        """Main async loop responsible for driving frame acquisition at target FPS."""
        while self.is_running:
            if not self._is_processing:
                self._is_processing = True
                try:
                    # Request snapshot from hardware camera controller
                    image_source = self.camera.take_picture()
                    if asyncio.iscoroutine(image_source):
                        image_source = await image_source

                    if image_source:
                        # Offload CPU-heavy PIL transformations to a worker thread
                        rgb_array, b64_str = await asyncio.to_thread(
                            self._process_image_data, image_source
                        )

                        if rgb_array is not None:
                            # Pass RGB numpy array to MediaPipe detector
                            self.detector.process_frame(rgb_array)

                            # Trigger UI frame update callback with Base64 payload
                            if self.frame_callback and b64_str:
                                self.frame_callback(b64_str)

                except Exception as e:
                    err_str = str(e).lower()
                    if "not initialized" not in err_str:
                        print(f"[Processor Error]: {e}")
                finally:
                    self._is_processing = False

            await asyncio.sleep(self.delay)

    def start(self) -> None:
        """Starts the background frame processing loop."""
        if not self.is_running:
            self.is_running = True
            self._task = asyncio.create_task(self._process_loop())

    def stop(self) -> None:
        """Stops the processing loop and cancels background tasks."""
        self.is_running = False
        if self._task:
            self._task.cancel()