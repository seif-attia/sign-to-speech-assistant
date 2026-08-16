import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from components.vector_view import VectorView
import services.data as data

""" 
    To access the raw vector values 

    Vector list form -> data.vector
    string form -> data.formatted_vals
"""

class HandDetector:
    """
    Wrapper around MediaPipe Tasks HandLandmarker using LIVE_STREAM asynchronous mode.
    """

    def __init__(
        self, model_path: str, num_hands: int = 2, HandDisplayView:VectorView = None
    ) -> None:
        self.hand_display = HandDisplayView

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
        if self.hand_display:
            self._on_hand_detected(result)
            

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

    def _on_hand_detected(self, result) -> None:
        """
        Processes MediaPipe HandLandmarker results into a fixed 126-dimensional float vector.
        Vector layout: 2 hands * 21 landmarks * 3 coordinates (x, y, z).
        Missing hands or landmarks are zero-padded to maintain consistent dimensionality.
        """
        data.vector = []
        num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

        for hand_idx in range(2):
            if result.hand_landmarks and hand_idx < num_hands:
                for lm in result.hand_landmarks[hand_idx]:
                    data.vector.extend([round(lm.x, 3), round(lm.y, 3), round(lm.z, 3)])
            else:
                # Pad missing hand slot with 63 zeros (21 landmarks * 3 coordinates)
                data.vector.extend([0.0] * 63)

      
        data.formatted_vals = ", ".join(f"{v:.3f}" for v in data.vector)
        output_text = (
            f"Hands: {num_hands} | Vector Dim: {len(data.vector)}\n\n"
            f"[{data.formatted_vals}]"
        )

        self.hand_display.update_data(output_text)