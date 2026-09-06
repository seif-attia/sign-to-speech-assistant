import collections
import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np

try:
    from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator
except ImportError:
    CompiledModel = None
    HardwareAccelerator = None

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "sign_language_model.tflite")
DEFAULT_LABELS_PATH = str(BASE_DIR / "labels.json")


class SignToTextClassifier:
    """
    Gesture classification service utilizing Google LiteRT CompiledModel
    and HardwareAccelerator (GPU/CPU) with a 20-frame temporal sliding window.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        labels_path: Optional[str] = DEFAULT_LABELS_PATH,
        labels: Optional[List[str]] = None,
        sequence_length: int = 10,
        feature_dim: int = 126,
        num_classes: int = 411,
        threshold: float = 0.30,
        prefer_hardware: Optional["HardwareAccelerator"] = None,
        on_prediction: Optional[Callable[[str, float], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.threshold = threshold
        self.on_prediction = on_prediction
        self.on_status = on_status
        self.prefer_hardware = prefer_hardware

        # Sliding window buffer of the last 20 frames
        self.frame_buffer = collections.deque(maxlen=self.sequence_length)

        # Load class labels
        self.labels: List[str] = self._load_labels(labels_path, labels)

        # LiteRT model state
        self.model: Optional[CompiledModel] = None
        self.input_buffers = None
        self.output_buffers = None
        self.is_loaded = False
        self.active_accelerator = None
        self.load_error: Optional[str] = None

        self.load_model(self.model_path, self.prefer_hardware)

    def _load_labels(
        self, labels_path: Optional[str], labels: Optional[List[str]]
    ) -> List[str]:
        """Loads labels from list or JSON file, falling back to numbered placeholders."""
        if labels:
            return labels

        if labels_path and Path(labels_path).exists():
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        return loaded
            except Exception as e:
                logger.warning(f"[SignClassifier] Failed to parse labels at {labels_path}: {e}")

        # Fallback to generic labels
        return [f"Sign_{i}" for i in range(self.num_classes)]

    def load_model(
        self,
        model_path: Optional[str] = None,
        prefer_hardware: Optional["HardwareAccelerator"] = None,
    ) -> bool:
        """
        Compiles the LiteRT model targeting the selected HardwareAccelerator.
        Attempts GPU first (or requested accelerator), with automatic fallback to CPU.
        """
        if CompiledModel is None or HardwareAccelerator is None:
            self.load_error = "ai_edge_litert is not installed or CompiledModel unavailable."
            logger.error(self.load_error)
            self.is_loaded = False
            return False

        target_path = model_path or self.model_path
        if not Path(target_path).exists():
            self.load_error = f"Model file not found at: {target_path}"
            logger.warning(self.load_error)
            self.is_loaded = False
            return False

        hardware_target = prefer_hardware or HardwareAccelerator.GPU
        accelerator_chain = [hardware_target]
        if hardware_target != HardwareAccelerator.CPU:
            accelerator_chain.append(HardwareAccelerator.CPU)

        self.close()
        last_error = None

        for accel in accelerator_chain:
            try:
                logger.info(f"[SignClassifier] Compiling LiteRT model with {accel.name} accelerator...")
                compiled = CompiledModel.from_file(target_path, hardware_accel=accel)

                self.model = compiled
                self.active_accelerator = accel.name
                self.input_buffers = self.model.create_input_buffers(0)
                self.output_buffers = self.model.create_output_buffers(0)

                # Auto-detect tensor shapes from model signature
                try:
                    sigs = self.model.get_signature_list()
                    if sigs:
                        sig_key = list(sigs.keys())[0]
                        in_details = self.model.get_input_tensor_details(sig_key)
                        out_details = self.model.get_output_tensor_details(sig_key)
                        if in_details:
                            in_shape = list(in_details.values())[0].get("shape", [])
                            if len(in_shape) >= 3:
                                self.sequence_length = int(in_shape[1])
                                self.feature_dim = int(in_shape[2])
                                self.frame_buffer = collections.deque(maxlen=self.sequence_length)
                        if out_details:
                            out_shape = list(out_details.values())[0].get("shape", [])
                            if len(out_shape) >= 2:
                                self.num_classes = int(out_shape[1])
                except Exception as shape_err:
                    logger.warning(f"[SignClassifier] Could not auto-detect tensor shapes: {shape_err}")

                self.is_loaded = True
                self.load_error = None
                logger.info(f"[SignClassifier] Successfully compiled model with {accel.name} accelerator (seq_len={self.sequence_length}, dim={self.feature_dim}, classes={self.num_classes}).")
                return True
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[SignClassifier] Could not compile with {accel.name}: {exc}"
                )

        self.is_loaded = False
        self.load_error = str(last_error)
        logger.debug(f"[SignClassifier] Model load warning: {self.load_error}")
        return False

    def process_hand_vector(
        self, hand_vector: List[float]
    ) -> Optional[Tuple[str, float]]:
        """
        Ingests a 63-dimensional landmark vector.
        When 20 consecutive valid frames have accumulated, runs LiteRT inference.
        Returns (predicted_label, confidence) if confidence exceeds threshold, else None.
        """
        # Validate vector size
        if not hand_vector or len(hand_vector) != self.feature_dim:
            return None

        self.frame_buffer.append(hand_vector)

        # Wait until we have a full temporal window of frames
        if len(self.frame_buffer) < self.sequence_length:
            if self.on_status:
                self.on_status(f"Buffering: {len(self.frame_buffer)}/{self.sequence_length} frames...")
            return None

        if not self.is_loaded or not self.model:
            return None

        try:
            # Construct contiguous [1, 20, 63] float32 tensor
            input_data = np.array(
                self.frame_buffer, dtype=np.float32
            ).reshape(1, self.sequence_length, self.feature_dim)

            # Zero-copy write directly to the accelerator input buffer
            self.input_buffers[0].write(input_data)

            # Execute compiled graph on accelerator
            self.model.run_by_index(0, self.input_buffers, self.output_buffers)

            # Read prediction tensor directly from accelerator output buffer
            predictions = self.output_buffers[0].read(self.num_classes, np.float32)

            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])
            
            raw_label = (
                self.labels[predicted_idx]
                if predicted_idx < len(self.labels)
                else f"Sign_{predicted_idx}"
            )

            # --- OpenCV-style Smoothing Logic ---
            if confidence > self.threshold:
                if not hasattr(self, 'predictions_history'):
                    self.predictions_history = []
                
                self.predictions_history.append(raw_label)
                self.predictions_history = self.predictions_history[-5:]
                
                if len(self.predictions_history) > 0:
                    most_common_pred, count = collections.Counter(self.predictions_history).most_common(1)[0]
                    if count >= 3:
                        logger.debug(f"[SignClassifier] Prediction: {most_common_pred} ({confidence * 100:.1f}%)")
                        if self.on_prediction:
                            self.on_prediction(most_common_pred, confidence)
                        return most_common_pred, confidence

            return None

        except Exception as e:
            logger.error(f"[SignClassifier] Inference error: {e}")

        return None

    def reset(self) -> None:
        """Clears the temporal window buffer."""
        self.frame_buffer.clear()

    def close(self) -> None:
        """Releases native accelerator handles."""
        if hasattr(self, "model") and self.model:
            try:
                self.model.close()
            except Exception:
                pass
            self.model = None
            self.input_buffers = None
            self.output_buffers = None
            self.is_loaded = False
