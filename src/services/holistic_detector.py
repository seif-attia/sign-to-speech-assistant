import os
import time
from pathlib import Path
from typing import Callable, Optional
import numpy as np

# Cross-platform edge runtime (Works in Android APK via tflite-runtime)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            raise RuntimeError("No compatible TFLite or LiteRT interpreter backend found.")

from components.vector_view import VectorView
import services.data as data

# 100 Selected Face Landmarks (matches training data & OpenCV test script exactly)
CHIN = [152, 175, 199, 200, 18, 148, 176, 150, 377, 400]
LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
NOSE = [1, 2, 4, 5, 6, 168, 195, 197]

FACE_LANDMARKS = sorted(list(set(CHIN + LIPS + LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + NOSE)))


class SubModelRunner:
    """Handles preprocessing, interpreter execution, and output tensor extraction."""

    def __init__(self, model_path: str, num_threads: int = 2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing at: {model_path}")

        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details["shape"]
        self.in_h, self.in_w = int(shape[1]), int(shape[2])
        self.in_dtype = self.input_details["dtype"]

    def run(self, rgb_crop: np.ndarray) -> list[np.ndarray]:
        src_h, src_w, _ = rgb_crop.shape
        if (src_h, src_w) != (self.in_h, self.in_w):
            # Fast bilinear resize using NumPy grid
            y = np.linspace(0, src_h - 1, self.in_h)
            x = np.linspace(0, src_w - 1, self.in_w)
            xf = np.floor(x).astype(int)
            yf = np.floor(y).astype(int)
            xc = np.minimum(xf + 1, src_w - 1)
            yc = np.minimum(yf + 1, src_h - 1)

            wa = ((xc - x) * (yc - y)[:, None])[:, :, None]
            wb = ((x - xf) * (yc - y)[:, None])[:, :, None]
            wc = ((xc - x) * (y - yf)[:, None])[:, :, None]
            wd = ((x - xf) * (y - yf)[:, None])[:, :, None]

            img = (
                wa * rgb_crop[yf[:, None], xf] +
                wb * rgb_crop[yf[:, None], xc] +
                wc * rgb_crop[yc[:, None], xf] +
                wd * rgb_crop[yc[:, None], xc]
            ).astype(rgb_crop.dtype)
        else:
            img = rgb_crop

        # Normalize pixel values
        if self.in_dtype == np.float32:
            tensor = (img.astype(np.float32) / 127.5) - 1.0
        else:
            scale, zero_point = self.input_details["quantization"]
            if scale > 0.0:
                tensor = np.round((img.astype(np.float32) / 255.0) / scale + zero_point).astype(self.in_dtype)
            else:
                tensor = img.astype(self.in_dtype)

        input_data = np.expand_dims(tensor, axis=0)
        self.interpreter.set_tensor(self.input_details["index"], input_data)
        self.interpreter.invoke()

        outputs = []
        for det in self.output_details:
            arr = self.interpreter.get_tensor(det["index"])
            scale, zero_point = det["quantization"]
            if scale > 0.0:
                arr = (arr.astype(np.float32) - zero_point) * scale
            outputs.append(arr)
        return outputs


class HolisticDetector:
    """
    Pure Python Holistic Landmark Extractor using extracted .tflite subgraphs.
    Produces the exact 525-feature vector required by Stage 11 Inception-Transformer.
    """

    def __init__(
        self,
        base_dir: str,
        HandDisplayView: Optional[VectorView] = None,
        prediction_callback: Optional[Callable[[np.ndarray, bool, bool], None]] = None,
        flip_horizontal: bool = True,
    ) -> None:
        self.hand_display = HandDisplayView
        self.prediction_callback = prediction_callback
        self.flip_horizontal = flip_horizontal

        base_path = Path(base_dir).resolve()
        pose_path = base_path / "pose_landmarker.tflite"
        hand_path = base_path / "hand_landmarker.tflite"
        face_path = base_path / "face_landmarker.tflite"

        for p in (pose_path, hand_path, face_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing TFLite model at: {p}")

        self.pose_runner = SubModelRunner(str(pose_path))
        self.hand_runner = SubModelRunner(str(hand_path))
        self.face_runner = SubModelRunner(str(face_path))

    def process_frame(self, rgb_frame: np.ndarray) -> None:
        if rgb_frame is None or rgb_frame.size == 0:
            return

        if rgb_frame.ndim == 3 and rgb_frame.shape[-1] == 4:
            rgb_frame = rgb_frame[:, :, :3]

        if rgb_frame.dtype != np.uint8:
            rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)

        if self.flip_horizontal:
            rgb_frame = np.fliplr(rgb_frame)

        h, w, _ = rgb_frame.shape

        # =========================================================================
        # 1. Pose Inference (33 Upper Body Landmarks = 99 dims)
        # =========================================================================
        pose_vector = [0.0] * 99
        pose_outs = self.pose_runner.run(rgb_frame)
        pose_lms = None

        for out in pose_outs:
            flat = out.flatten()
            # pose_landmarks_detector outputs 195 (39x5) or 165 (33x5)
            if flat.size in (195, 165):
                raw_lms = flat.reshape(-1, 5)[:33, :3].copy()

                # Scale down if coordinates are in pixel domain (0..256)
                if np.max(raw_lms[:, :2]) > 1.5:
                    raw_lms[:, 0] /= float(self.pose_runner.in_w)
                    raw_lms[:, 1] /= float(self.pose_runner.in_h)

                raw_lms[:, :2] = np.clip(raw_lms[:, :2], 0.0, 1.0)
                pose_lms = raw_lms
                pose_vector = raw_lms.flatten().tolist()
                break

        # =========================================================================
        # 2. Hand Inference (Dual Hands = 126 dims)
        # =========================================================================
        left_hand_vector = [0.0] * 63
        right_hand_vector = [0.0] * 63
        hands_present = False
        is_signing = False

        crops = []
        # Use pose wrist coordinates (15: Left Wrist, 16: Right Wrist) to create hand crops
        if pose_lms is not None:
            for wrist_idx in (15, 16):
                wx, wy = pose_lms[wrist_idx][0], pose_lms[wrist_idx][1]
                if 0.01 < wx < 0.99 and 0.01 < wy < 0.99:
                    cx, cy = int(wx * w), int(wy * h)
                    box_s = int(min(w, h) * 0.45)
                    y1, y2 = max(0, cy - box_s // 2), min(h, cy + box_s // 2)
                    x1, x2 = max(0, cx - box_s // 2), min(w, cx + box_s // 2)
                    if (y2 - y1) > 20 and (x2 - x1) > 20:
                        crops.append((rgb_frame[y1:y2, x1:x2], x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h))

        if not crops:
            crops.append((rgb_frame, 0.0, 0.0, 1.0, 1.0))

        detected_hands = []
        for crop_img, off_x, off_y, scale_w, scale_h in crops:
            hand_outs = self.hand_runner.run(crop_img)
            for out in hand_outs:
                flat = out.flatten()
                if flat.size == 63:  # 21 landmarks * 3
                    hlms = flat.reshape(21, 3).copy()

                    if np.max(hlms[:, :2]) > 1.5:
                        hlms[:, 0] /= float(self.hand_runner.in_w)
                        hlms[:, 1] /= float(self.hand_runner.in_h)

                    hlms[:, 0] = hlms[:, 0] * scale_w + off_x
                    hlms[:, 1] = hlms[:, 1] * scale_h + off_y
                    hlms[:, :2] = np.clip(hlms[:, :2], 0.0, 1.0)

                    wrist = hlms[0]
                    if wrist[1] < 0.94:
                        is_signing = True

                    detected_hands.append((wrist[0], wrist[1], hlms.flatten().tolist()))
                    break

        if detected_hands:
            hands_present = True
            if len(detected_hands) == 1:
                wx, wy, coords = detected_hands[0]
                is_right_hand = wx >= 0.50
                if pose_lms is not None:
                    l_sh = pose_lms[11]
                    r_sh = pose_lms[12]
                    d_left = (wx - l_sh[0]) ** 2 + (wy - l_sh[1]) ** 2
                    d_right = (wx - r_sh[0]) ** 2 + (wy - r_sh[1]) ** 2
                    is_right_hand = d_right < d_left

                if is_right_hand:
                    right_hand_vector = coords
                else:
                    left_hand_vector = coords
            else:
                sorted_hands = sorted(detected_hands, key=lambda h: h[0])
                left_hand_vector = sorted_hands[0][2]
                right_hand_vector = sorted_hands[1][2]

        # =========================================================================
        # 3. Face Inference (100 Selected Landmarks = 300 dims)
        # =========================================================================
        face_vector = [0.0] * 300
        face_outs = self.face_runner.run(rgb_frame)
        for out in face_outs:
            flat = out.flatten()
            if flat.size >= 1404:  # 468 or 478 landmarks * 3
                coords = flat.reshape(-1, 3).copy()
                if np.max(coords[:, :2]) > 1.5:
                    coords[:, 0] /= float(self.face_runner.in_w)
                    coords[:, 1] /= float(self.face_runner.in_h)

                coords[:, :2] = np.clip(coords[:, :2], 0.0, 1.0)
                n_mesh = len(coords)
                selected = []
                for idx in FACE_LANDMARKS:
                    if idx < n_mesh:
                        selected.extend([coords[idx][0], coords[idx][1], coords[idx][2]])
                    else:
                        selected.extend([0.0, 0.0, 0.0])
                face_vector = selected
                break

        # =========================================================================
        # 4. Concatenate Full 525 Vector
        # =========================================================================
        frame_vector = np.array(
            pose_vector + left_hand_vector + right_hand_vector + face_vector,
            dtype=np.float32,
        )

        data.vector = frame_vector.tolist()
        data.pose_vector = pose_vector
        data.left_hand_vector = left_hand_vector
        data.right_hand_vector = right_hand_vector
        data.face_vector = face_vector

        if self.prediction_callback:
            self.prediction_callback(frame_vector, hands_present, is_signing)

        if self.hand_display:
            has_pose = any(pose_vector)
            has_lh = any(left_hand_vector)
            has_rh = any(right_hand_vector)
            has_face = any(face_vector)
            output_text = (
                f"Pose: {'✓' if has_pose else '✗'} | "
                f"L-Hand: {'✓' if has_lh else '✗'} | "
                f"R-Hand: {'✓' if has_rh else '✗'} | "
                f"Face: {'✓' if has_face else '✗'}\n"
                f"Hands Present: {hands_present} | Signing: {is_signing}"
            )
            self.hand_display.update_data(output_text)

    def close(self):
        pass