import time
from typing import Callable, List, Optional, Tuple
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from components.vector_view import VectorView
import services.data as data

# 100 Face Landmarks in sorted order (per FLET_INTEGRATION_SPEC.md)
FACE_INDICES: List[int] = sorted([
    # Chin (10)
    18, 148, 150, 152, 175, 176, 199, 200, 377, 400,
    # Lips (40)
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146,
    178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317,
    318, 321, 324, 375, 402, 405, 409, 415,
    # Nose (8)
    1, 2, 4, 5, 6, 168, 195, 197,
    # Eyes (32)
    7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
    # Eyebrows (10)
    63, 66, 70, 105, 107, 293, 296, 300, 334, 336
])


def extract_and_normalize_holistic(
    result, flip_horizontal: bool = True
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Extracts and normalizes MediaPipe Holistic landmarks into a 525-element vector:
      [0 : 99]    - Pose (33 landmarks x 3 coords)
      [99 : 162]  - Left Hand (21 landmarks x 3 coords)
      [162 : 225] - Right Hand (21 landmarks x 3 coords)
      [225 : 525] - Face Keypoints (100 landmarks x 3 coords)

    Coordinates are centered and scaled by the Mid-Shoulder Centroid (Pose landmarks 11 & 12).
    Undetected components remain strictly 0.0.
    """
    mid_x, mid_y, mid_z = 0.0, 0.0, 0.0
    scale = 1.0

    # 1. Mid-Shoulder Centroid & Scale
    pose_lms = getattr(result, "pose_landmarks", None)
    if pose_lms and len(pose_lms) >= 13:
        ls = pose_lms[11]  # Left Shoulder
        rs = pose_lms[12]  # Right Shoulder
        mid_x = (ls.x + rs.x) / 2.0
        mid_y = (ls.y + rs.y) / 2.0
        mid_z = (ls.z + rs.z) / 2.0
        dist = np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2 + (ls.z - rs.z) ** 2)
        if dist > 1e-4:
            scale = float(dist)

    # 2. Pose (33 landmarks = 99 features)
    pose_vector = [0.0] * 99
    if pose_lms:
        for i, lm in enumerate(pose_lms[:33]):
            pose_vector[i * 3] = (lm.x - mid_x) / scale
            pose_vector[i * 3 + 1] = (lm.y - mid_y) / scale
            pose_vector[i * 3 + 2] = (lm.z - mid_z) / scale

    # 3. Left Hand & Right Hand (21 landmarks each = 63 + 63 features)
    raw_left_hand: List[float] = [0.0] * 63
    left_hand_lms = getattr(result, "left_hand_landmarks", None)
    if left_hand_lms:
        coords = []
        for lm in left_hand_lms[:21]:
            coords.extend([
                (lm.x - mid_x) / scale,
                (lm.y - mid_y) / scale,
                (lm.z - mid_z) / scale,
            ])
        if len(coords) == 63:
            raw_left_hand = coords

    raw_right_hand: List[float] = [0.0] * 63
    right_hand_lms = getattr(result, "right_hand_landmarks", None)
    if right_hand_lms:
        coords = []
        for lm in right_hand_lms[:21]:
            coords.extend([
                (lm.x - mid_x) / scale,
                (lm.y - mid_y) / scale,
                (lm.z - mid_z) / scale,
            ])
        if len(coords) == 63:
            raw_right_hand = coords

    # When camera feed is mirrored (flip_horizontal=True), MediaPipe's Left hand
    # corresponds to the user's physical Right hand, and Right to physical Left.
    if flip_horizontal:
        left_hand_vector = raw_right_hand
        right_hand_vector = raw_left_hand
    else:
        left_hand_vector = raw_left_hand
        right_hand_vector = raw_right_hand

    # 4. Face Keypoints (100 landmarks = 300 features)
    face_vector = [0.0] * 300
    face_lms = getattr(result, "face_landmarks", None)
    if face_lms:
        num_face_lms = len(face_lms)
        coords = []
        for idx in FACE_INDICES:
            if idx < num_face_lms:
                lm = face_lms[idx]
                coords.extend([
                    (lm.x - mid_x) / scale,
                    (lm.y - mid_y) / scale,
                    (lm.z - mid_z) / scale,
                ])
            else:
                coords.extend([0.0, 0.0, 0.0])
        if len(coords) == 300:
            face_vector = coords

    # 5. Full 525-feature vector
    full_vector = pose_vector + left_hand_vector + right_hand_vector + face_vector
    return full_vector, pose_vector, left_hand_vector, right_hand_vector, face_vector


class HolisticDetector:
    """
    Wrapper around MediaPipe Tasks HolisticLandmarker using LIVE_STREAM asynchronous mode.
    Generates 525-dimensional normalized landmark vectors according to FLET_INTEGRATION_SPEC.md.
    """

    def __init__(
        self,
        model_path: str,
        HandDisplayView: Optional[VectorView] = None,
        classifier=None,
        prediction_callback: Optional[Callable[[str, float], None]] = None,
        flip_horizontal: bool = True,
    ) -> None:
        self.hand_display = HandDisplayView
        self.classifier = classifier
        self.prediction_callback = prediction_callback
        self.flip_horizontal = flip_horizontal

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._internal_callback,
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)

    def _internal_callback(
        self, result, output_image, timestamp_ms: int
    ) -> None:
        """Internal callback invoked by MediaPipe Tasks when holistic detection completes."""
        if self.hand_display or self.classifier:
            self._on_holistic_detected(result)

    def process_frame(self, rgb_array: np.ndarray) -> None:
        """Dispatches an RGB frame asynchronously to the MediaPipe Holistic landmarker."""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_array
        )
        timestamp_ms = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def close(self) -> None:
        """Releases C++ resources allocated by MediaPipe Holistic landmarker."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()

    def _on_holistic_detected(self, result) -> None:
        """
        Normalizes detected landmarks into the 525-dimensional feature vector,
        updates data state, invokes classifier inference, and refreshes UI display.
        """
        full_vector, pose_vec, left_vec, right_vec, face_vec = extract_and_normalize_holistic(
            result, flip_horizontal=self.flip_horizontal
        )

        data.vector = full_vector
        data.pose_vector = pose_vec
        data.left_hand_vector = left_vec
        data.right_hand_vector = right_vec
        data.face_vector = face_vec

        prediction_info = ""
        if self.classifier:
            fn = getattr(self.classifier, "process_holistic_vector", None) or getattr(
                self.classifier, "process_hand_vector", None
            )
            if fn:
                prediction = fn(full_vector)
                if prediction:
                    sign_label, confidence = prediction
                    data.predicted_sign = sign_label
                    data.confidence = confidence
                    prediction_info = f"\n\n🎯 Recognized: {sign_label.upper()} ({confidence * 100:.1f}%)"
                    if self.prediction_callback:
                        self.prediction_callback(sign_label, confidence)

        has_pose = any(pose_vec)
        has_left = any(left_vec)
        has_right = any(right_vec)
        has_face = any(face_vec)

        data.formatted_vals = ", ".join(f"{v:.3f}" for v in full_vector[:15]) + "..."

        if self.hand_display:
            output_text = (
                f"Pose: {'✓' if has_pose else '✗'} | "
                f"L-Hand: {'✓' if has_left else '✗'} | "
                f"R-Hand: {'✓' if has_right else '✗'} | "
                f"Face: {'✓' if has_face else '✗'} | "
                f"Dim: {len(full_vector)}"
                f"{prediction_info}\n\n"
                f"Sample [0:15]: [{data.formatted_vals}]\n\n"
                f"Full 525 Vector:\n[{', '.join(f'{v:.3f}' for v in full_vector)}]"
            )
            self.hand_display.update_data(output_text)

