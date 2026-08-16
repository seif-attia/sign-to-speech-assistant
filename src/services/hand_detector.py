import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    """
    Wrapper around MediaPipe Tasks HandLandmarker using LIVE_STREAM asynchronous mode.
    """

    def __init__(
        self, model_path: str, num_hands: int = 2, ui_callback=None
    ) -> None:
        self.ui_callback = ui_callback

        # Initialize MediaPipe Task options for streaming hand detection
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            result_callback=self._internal_callback,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _internal_callback(
        self, result, output_image, timestamp_ms: int
    ) -> None:
        """Internal callback invoked by MediaPipe when hand detection completes."""
        if self.ui_callback:
            self.ui_callback(result)

    def process_frame(self, rgb_array) -> None:
        """
        Converts NumPy RGB array to mp.Image and dispatches frame asynchronously
        with a millisecond timestamp to MediaPipe.
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_array
        )
        timestamp_ms = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def close(self) -> None:
        """Releases C++ resources allocated by MediaPipe Task runner."""
        if hasattr(self, "landmarker"):
            self.landmarker.close()