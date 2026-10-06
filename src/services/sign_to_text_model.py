import json
import os
import numpy as np

# Multi-Backend Edge Runtime (matching OpenCV test code)
Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            pass


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Video-Level Median Anchor Normalization for (T, 525) inputs.
    Matches test script verbatim.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 3:
        seq = seq[0]
    T, D = seq.shape
    if D != 525:
        return seq

    pose = seq[:, :99].reshape(T, 33, 3)
    lh = seq[:, 99:162].reshape(T, 21, 3)
    rh = seq[:, 162:225].reshape(T, 21, 3)
    face = seq[:, 225:525].reshape(T, 100, 3)

    ls = pose[:, 11, :]  # Left shoulder
    rs = pose[:, 12, :]  # Right shoulder
    valid = (np.abs(ls).sum(axis=-1) > 1e-4) & (np.abs(rs).sum(axis=-1) > 1e-4)

    if np.any(valid):
        mids = (ls + rs) / 2.0
        mid_anchor = np.median(mids[valid], axis=0)
        spans = np.linalg.norm(ls - rs, axis=-1)
        scale_anchor = float(np.percentile(spans[valid], 85))
        if scale_anchor < 1e-4:
            scale_anchor = 1.0
    else:
        nose = pose[:, 0, :]
        valid_nose = np.abs(nose).sum(axis=-1) > 1e-4
        mid_anchor = np.median(nose[valid_nose], axis=0) if np.any(valid_nose) else np.zeros(3, dtype=np.float32)
        scale_anchor = 1.0

    p_m = (pose != 0.0).any(axis=-1, keepdims=True)
    lh_m = (lh != 0.0).any(axis=-1, keepdims=True)
    rh_m = (rh != 0.0).any(axis=-1, keepdims=True)
    f_m = (face != 0.0).any(axis=-1, keepdims=True)

    p_n = np.where(p_m, (pose - mid_anchor) / scale_anchor, 0.0).reshape(T, 99)
    lh_n = np.where(lh_m, (lh - mid_anchor) / scale_anchor, 0.0).reshape(T, 63)
    rh_n = np.where(rh_m, (rh - mid_anchor) / scale_anchor, 0.0).reshape(T, 63)
    f_n = np.where(f_m, (face - mid_anchor) / scale_anchor, 0.0).reshape(T, 300)

    return np.concatenate([p_n, lh_n, rh_n, f_n], axis=-1).astype(np.float32)


class SignPredictor:
    def __init__(self, model_path: str, labels_path: str):
        if Interpreter is None:
            raise RuntimeError("No suitable TFLite interpreter backend available.")

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.sequence_length = int(self.input_details[0]["shape"][1])
        self.num_classes = int(self.output_details[0]["shape"][1])

        self.labels = self._load_labels(labels_path)

        # Sliding window of raw vectors
        self.sequence = []
        self.frame_count = 0

    def _load_labels(self, path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data[str(i)] for i in range(len(data))]

    def process_frame(
        self, frame_vector: np.ndarray, hands_present: bool, is_signing: bool
    ) -> dict:
        """
        Ingests frame, maintains sliding window, runs inference every 2 frames,
        and applies Kinetic Rest & Signing-Space gating.
        """
        self.frame_count += 1
        self.sequence.append(frame_vector)
        self.sequence = self.sequence[-self.sequence_length :]

        # Check buffer warming status
        if len(self.sequence) < self.sequence_length:
            return {
                "status": "WARMING",
                "buffer_fill": len(self.sequence),
                "total_buffer": self.sequence_length,
                "hands_present": hands_present,
                "is_signing": is_signing,
            }

        # Inference cadence: every 2 frames
        if self.frame_count % 2 != 0:
            return {"status": "SKIP"}

        # Kinetic Rest Gate: If hands aren't active/elevated, suppress predictions
        if not hands_present:
            return {"status": "NO_HANDS"}
        if not is_signing:
            return {"status": "RESTING"}

        # Video-Level Median Anchor Normalization on completed window
        norm_seq = normalize_sequence(np.array(self.sequence, dtype=np.float32))
        input_data = np.expand_dims(norm_seq, axis=0)

        # Handle quantization if applicable
        scale, zero_point = self.input_details[0]["quantization"]
        if scale > 0.0:
            input_data = np.round(input_data / scale + zero_point).astype(self.input_details[0]["dtype"])

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        top5_indices = np.argsort(preds)[::-1][:5]
        top1_idx = int(top5_indices[0])
        top1_conf = float(preds[top1_idx])

        top5_candidates = [
            (self.labels[int(idx)], float(preds[int(idx)]))
            for idx in top5_indices
            if int(idx) < len(self.labels)
        ]

        return {
            "status": "PREDICTION",
            "word": self.labels[top1_idx] if top1_idx < len(self.labels) else "Unknown",
            "conf": top1_conf,
            "top5": top5_candidates,
            "hands_present": hands_present,
            "is_signing": is_signing,
        }

    def clear(self):
        self.sequence.clear()
        self.frame_count = 0