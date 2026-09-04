import time
from typing import Callable, Optional
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
        self,
        model_path: str,
        num_hands: int = 2,
        HandDisplayView: VectorView = None,
        classifier=None,
        prediction_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.hand_display = HandDisplayView
        self.classifier = classifier
        self.prediction_callback = prediction_callback

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
        Vector layout: Right Hand (63 values) + Left Hand (63 values).
        Missing hands or landmarks are zero-padded.
        Coordinates are relative to the wrist position (landmark 0) for translation invariance.
        """
        right_hand_vector = [0.0] * 63
        left_hand_vector = [0.0] * 63
        
        num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

        if result.hand_landmarks and result.handedness:
            for hand_idx, landmarks in enumerate(result.hand_landmarks):
                if hand_idx >= 2:
                    break
                
                # Identify if hand is Left or Right
                category = result.handedness[hand_idx][0]
                hand_label = category.category_name # "Right" or "Left"

                # Get wrist coordinates (landmark 0) to make others relative
                wrist_x = landmarks[0].x
                wrist_y = landmarks[0].y
                wrist_z = landmarks[0].z

                hand_coords = []
                for lm in landmarks:
                    rel_x = round(lm.x - wrist_x, 3)
                    rel_y = round(lm.y - wrist_y, 3)
                    rel_z = round(lm.z - wrist_z, 3)
                    hand_coords.extend([rel_x, rel_y, rel_z])

                # Due to camera mirroring (flip_horizontal=True), MediaPipe's handedness is inverted.
                # Physical Right Hand appears as "Left", Physical Left Hand appears as "Right".
                if hand_label == "Left":
                    right_hand_vector = hand_coords
                elif hand_label == "Right":
                    left_hand_vector = hand_coords

        # Combine into the final 126-dimensional vector
        data.vector = right_hand_vector + left_hand_vector
        data.right_hand_vector = right_hand_vector
        data.left_hand_vector = left_hand_vector

        # Select primary active hand for 63-dim sign classifier (prefer right, fallback to left)
        active_hand = right_hand_vector if any(right_hand_vector) else left_hand_vector
        prediction_info = ""

        if self.classifier and any(active_hand):
            prediction = self.classifier.process_hand_vector(active_hand)
            if prediction:
                sign_label, confidence = prediction
                data.predicted_sign = sign_label
                data.confidence = confidence
                prediction_info = f"\n\n🎯 Recognized: {sign_label.upper()} ({confidence * 100:.1f}%)"
                if self.prediction_callback:
                    self.prediction_callback(sign_label, confidence)

        data.formatted_vals = ", ".join(f"{v:.3f}" for v in data.vector)
        output_text = (
            f"Hands: {num_hands} | Vector Dim: {len(data.vector)}"
            f"{prediction_info}\n\n"
            f"[{data.formatted_vals}]"
        )

        self.hand_display.update_data(output_text)