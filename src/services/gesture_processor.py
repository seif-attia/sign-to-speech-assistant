import os
from collections import deque
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GestureProcessor:
    def __init__(self, model_path: str = "hand_landmarker.task", sequence_length: int = 30):
        # STEP 1 & 2: Configure and create HandLandmarker using the Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Sequence Buffer for dynamic gesture tracking (30 frames x 126 floats)
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)

    def process_image_path(self, image_path: str) -> np.ndarray:
        """Loads frame file into mp.Image, runs HandLandmarker detection, and returns (30, 126) matrix."""
        vector = [0.0] * 126

        if os.path.exists(image_path):
            try:
                # STEP 3: Load the input image using MediaPipe's Native Image loader
                mp_image = mp.Image.create_from_file(image_path)

                # STEP 4: Detect hand landmarks from input image
                detection_result = self.detector.detect(mp_image)

                # STEP 5: Extract & normalize vector
                vector = self._extract_landmarks(detection_result)

            except Exception as e:
                print(f"HandLandmarker detection error: {e}")

            finally:
                # Immediate storage cleanup
                try:
                    os.remove(image_path)
                except Exception:
                    pass

        # Sliding Window Buffer
        self.buffer.append(vector)

        # Pad with zero-vectors if buffer isn't full yet
        padded_buffer = list(self.buffer)
        while len(padded_buffer) < self.sequence_length:
            padded_buffer.insert(0, [0.0] * 126)

        return np.array(padded_buffer)

    def _extract_landmarks(self, detection_result) -> list[float]:
        """Extracts 21 landmarks (x, y, z) for up to 2 hands relative to the wrist."""
        vector = []

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks[:2]:
                wrist = hand_landmarks[0]  # Reference origin (wrist joint)
                for lm in hand_landmarks:
                    vector.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        # Pad missing values if fewer than 2 hands detected to guarantee 126 elements
        while len(vector) < 126:
            vector.append(0.0)

        return vector[:126]