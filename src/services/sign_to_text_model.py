import collections
import json
import logging
from pathlib import Path
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple
import numpy as np

try:
    from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator
except ImportError:
    CompiledModel = None
    HardwareAccelerator = None

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
HOLISTIC_MODEL_PATH = BASE_DIR / "sign_language_holistic_model.tflite"
DEFAULT_MODEL_PATH = str(
    HOLISTIC_MODEL_PATH
    if HOLISTIC_MODEL_PATH.exists()
    else (BASE_DIR / "sign_language_model.tflite")
)
DEFAULT_LABELS_PATH = str(BASE_DIR / "labels.json")


class MockPredictionHandler:
    """
    Mock prediction provider for unit testing, CI pipelines, and UI prototyping
    without requiring compiled LiteRT runtime or actual model weights.

    Supports:
    1. Deterministic scripted queue for unit testing edge cases.
    2. Dynamic cycling through realistic gesture vocabulary.
    3. Probability vector synthesis for downstream softmax / scoring tests.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        mock_sequence: Optional[List[Tuple[str, float]]] = None,
        default_confidence: float = 0.88,
    ) -> None:
        self.labels = labels or [
            "hello",
            "thank you",
            "yes",
            "no",
            "help",
            "please",
            "water",
            "friend",
        ]
        self.mock_sequence: Deque[Tuple[str, float]] = collections.deque(
            mock_sequence or []
        )
        self.default_confidence = default_confidence
        self._inference_count = 0

    def set_sequence(self, sequence: List[Tuple[str, float]]) -> None:
        """Queues a predetermined sequence of (label, confidence) for unit tests."""
        self.mock_sequence = collections.deque(sequence)

    def predict(
        self, frame_buffer: np.ndarray, num_classes: int
    ) -> Tuple[int, float, np.ndarray]:
        """
        Simulates model inference and returns (predicted_class_index, confidence, probabilities).
        """
        # 1. Deterministic scripted queue if configured
        if self.mock_sequence:
            label, conf = self.mock_sequence.popleft()
            try:
                idx = self.labels.index(label)
            except ValueError:
                idx = 0
            probs = np.zeros(num_classes, dtype=np.float32)
            probs[idx] = conf
            if num_classes > 1:
                rem = max(0.0, 1.0 - conf) / (num_classes - 1)
                for i in range(num_classes):
                    if i != idx:
                        probs[i] = rem
            return idx, conf, probs

        # 2. Dynamic simulation cycling every 15 inference calls
        self._inference_count += 1
        active_class_count = min(len(self.labels), max(1, num_classes))
        idx = (self._inference_count // 15) % active_class_count
        conf = self.default_confidence

        probs = np.zeros(num_classes, dtype=np.float32)
        probs[idx] = conf
        if num_classes > 1:
            rem = max(0.0, 1.0 - conf) / (num_classes - 1)
            for i in range(num_classes):
                if i != idx:
                    probs[i] = rem

        return idx, conf, probs


class SignToTextClassifier:
    """
    Sign language gesture classification service with Google LiteRT CompiledModel,
    temporal sliding window, missing-hand grace period, weighted majority voting,
    and debounce/cooldown filtering.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        labels_path: Optional[str] = DEFAULT_LABELS_PATH,
        labels: Optional[List[str]] = None,
        sequence_length: int = 30,
        feature_dim: int = 525,
        num_classes: int = 2000,
        threshold: float = 0.50,
        voting_window_size: int = 5,
        min_voting_count: int = 3,
        cooldown_seconds: float = 1.5,
        max_missing_frames: int = 5,
        inference_stride: int = 1,
        use_mock: bool = False,
        mock_handler: Optional[MockPredictionHandler] = None,
        prefer_hardware: Optional["HardwareAccelerator"] = None,
        on_prediction: Optional[Callable[[str, float], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.threshold = threshold
        self.voting_window_size = voting_window_size
        self.min_voting_count = min_voting_count
        self.cooldown_seconds = cooldown_seconds
        self.max_missing_frames = max_missing_frames
        self.inference_stride = max(1, inference_stride)
        self.prefer_hardware = prefer_hardware
        self.on_prediction = on_prediction
        self.on_status = on_status

        # Temporal sequence sliding window (raw normalized landmark frames)
        self.frame_buffer: Deque[List[float]] = collections.deque(
            maxlen=self.sequence_length
        )

        # Voting window buffer: stores tuples of (label, confidence, timestamp)
        # Explicitly tracks "<IDLE>" so low-confidence frames dilute old gestures.
        self.predictions_history: Deque[Tuple[str, float, float]] = (
            collections.deque(maxlen=self.voting_window_size)
        )

        # Load class vocabulary
        self.labels: List[str] = self._load_labels(labels_path, labels)

        # Mock handler & test hooks
        self.use_mock = use_mock
        self.mock_handler: MockPredictionHandler = (
            mock_handler or MockPredictionHandler(labels=self.labels)
        )
        self.is_mock = False

        # State tracking for debounce, dropout grace periods, and stride
        self._missing_hand_counter: int = 0
        self._frame_counter: int = 0
        self._last_emitted_label: Optional[str] = None
        self._last_emitted_time: float = 0.0
        self._logged_dim_mismatch: bool = False

        # LiteRT runtime handles
        self.model: Optional[CompiledModel] = None
        self.input_buffers = None
        self.output_buffers = None
        self.is_loaded = False
        self.active_accelerator: Optional[str] = None
        self.load_error: Optional[str] = None

        if self.use_mock:
            self._activate_mock_runtime(reason="Explicitly requested via use_mock=True")
        else:
            loaded = self.load_model(self.model_path, self.prefer_hardware)
            if not loaded:
                # Graceful fallback: enable mock mode so UI and testing proceed without model file
                self._activate_mock_runtime(
                    reason=f"Model loading failed ({self.load_error}); falling back to mock mode"
                )

    def _activate_mock_runtime(self, reason: str = "") -> None:
        """Enables the internal mock predictor for zero-dependency execution."""
        self.is_mock = True
        self.is_loaded = True
        self.active_accelerator = "MOCK_CPU"
        self.load_error = None
        logger.info(f"[SignClassifier] {reason}")

    def _load_labels(
        self, labels_path: Optional[str], labels: Optional[List[str]]
    ) -> List[str]:
        """Loads labels from memory or JSON, falling back to indexed placeholders."""
        if labels:
            return labels

        if labels_path and Path(labels_path).exists():
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list) and len(loaded) > 0:
                        return loaded
            except Exception as e:
                logger.warning(
                    f"[SignClassifier] Failed to parse labels at {labels_path}: {e}"
                )

        return [f"Sign_{i}" for i in range(self.num_classes)]

    def load_model(
        self,
        model_path: Optional[str] = None,
        prefer_hardware: Optional["HardwareAccelerator"] = None,
    ) -> bool:
        """
        Compiles the LiteRT model targeting the selected HardwareAccelerator.
        Attempts GPU first, then falls back to CPU.
        """
        if CompiledModel is None or HardwareAccelerator is None:
            self.load_error = "ai_edge_litert is not installed or CompiledModel unavailable."
            logger.warning(self.load_error)
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
                logger.info(
                    f"[SignClassifier] Compiling LiteRT model with {accel.name} accelerator..."
                )
                compiled = CompiledModel.from_file(target_path, hardware_accel=accel)

                self.model = compiled
                self.active_accelerator = accel.name
                self.input_buffers = self.model.create_input_buffers(0)
                self.output_buffers = self.model.create_output_buffers(0)

                # Inspect model tensor shapes dynamically
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
                                self.frame_buffer = collections.deque(
                                    maxlen=self.sequence_length
                                )
                        if out_details:
                            out_shape = list(out_details.values())[0].get("shape", [])
                            if len(out_shape) >= 2:
                                self.num_classes = int(out_shape[1])
                except Exception as shape_err:
                    logger.warning(
                        f"[SignClassifier] Could not auto-detect tensor shapes: {shape_err}"
                    )

                self.is_loaded = True
                self.is_mock = False
                self.load_error = None
                logger.info(
                    f"[SignClassifier] Successfully compiled model with {accel.name} "
                    f"(seq_len={self.sequence_length}, dim={self.feature_dim}, classes={self.num_classes})."
                )
                return True
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[SignClassifier] Could not compile with {accel.name}: {exc}"
                )

        self.is_loaded = False
        self.load_error = str(last_error)
        return False

    def process_hand_vector(
        self, hand_vector: List[float]
    ) -> Optional[Tuple[str, float]]:
        """
        Ingests a 525-dimensional holistic landmark vector.
        Checks hand presence with grace-period dropout handling, fills sliding temporal window,
        runs model inference, and applies weighted majority voting with cooldown suppression.
        """
        # Validate dimensionality
        if not hand_vector or len(hand_vector) != self.feature_dim:
            if self.on_status and not self._logged_dim_mismatch:
                self._logged_dim_mismatch = True
                self.on_status(
                    f"Model expects {self.feature_dim} features, received {len(hand_vector) if hand_vector else 0}"
                )
            return None

        # 1. Hand Presence Checking Logic:
        # Left Hand: [99:162], Right Hand: [162:225]
        # Undetected hands in HolisticLandmarker are populated with 0.0
        left_hand_active = any(abs(x) > 1e-4 for x in hand_vector[99:162])
        right_hand_active = any(abs(x) > 1e-4 for x in hand_vector[162:225])
        hand_detected = left_hand_active or right_hand_active

        # 2. Missing-Hand Grace Period Handling:
        if hand_detected:
            self._missing_hand_counter = 0
            self.frame_buffer.append(hand_vector)
        else:
            self._missing_hand_counter += 1
            if (
                self._missing_hand_counter <= self.max_missing_frames
                and len(self.frame_buffer) > 0
            ):
                # Transient dropout: repeat last valid frame to prevent sequence interruption
                self.frame_buffer.append(self.frame_buffer[-1])
                if self.on_status and self._missing_hand_counter == 1:
                    self.on_status("Hand tracking intermittent (holding sequence)...")
            else:
                # Sustained absence: hand truly lowered or out of frame
                if len(self.frame_buffer) > 0:
                    self.frame_buffer.clear()

                # Record IDLE step to age out old predictions in history
                self.predictions_history.append(("<IDLE>", 0.0, time.time()))

                # Reset previous sign lock once hand is confirmed down
                if (time.time() - self._last_emitted_time) >= self.cooldown_seconds:
                    self._last_emitted_label = None

                if self.on_status:
                    self.on_status("No hand detected. Stand by...")
                return None

        # 3. Wait until temporal sliding sequence buffer is full
        if len(self.frame_buffer) < self.sequence_length:
            if self.on_status:
                self.on_status(
                    f"Buffering: {len(self.frame_buffer)}/{self.sequence_length} frames..."
                )
            return None

        # 4. Inference Stride check (throttle compute if configured)
        self._frame_counter += 1
        if self._frame_counter % self.inference_stride != 0:
            return None

        if not self.is_loaded:
            return None

        # 5. Execute Model or Mock Inference
        try:
            raw_label: str = ""
            confidence: float = 0.0

            if self.is_mock:
                input_data = np.array(self.frame_buffer, dtype=np.float32).reshape(
                    1, self.sequence_length, self.feature_dim
                )
                pred_idx, confidence, _ = self.mock_handler.predict(
                    input_data, self.num_classes
                )
                raw_label = (
                    self.labels[pred_idx]
                    if pred_idx < len(self.labels)
                    else f"Sign_{pred_idx}"
                )
            else:
                input_data = np.array(self.frame_buffer, dtype=np.float32).reshape(
                    1, self.sequence_length, self.feature_dim
                )
                self.input_buffers[0].write(input_data)
                self.model.run_by_index(0, self.input_buffers, self.output_buffers)
                predictions = self.output_buffers[0].read(self.num_classes, np.float32)

                # Apply Softmax if model outputs raw unnormalized logits
                if not (
                    np.all(predictions >= 0.0)
                    and np.isclose(np.sum(predictions), 1.0, atol=0.05)
                ):
                    exp_preds = np.exp(predictions - np.max(predictions))
                    predictions = exp_preds / np.sum(exp_preds)

                pred_idx = int(np.argmax(predictions))
                confidence = float(predictions[pred_idx])
                raw_label = (
                    self.labels[pred_idx]
                    if pred_idx < len(self.labels)
                    else f"Sign_{pred_idx}"
                )

            # 6. Update Voting Buffer (dilutes history with <IDLE> when confidence is below threshold)
            now = time.time()
            if confidence >= self.threshold:
                self.predictions_history.append((raw_label, confidence, now))
            else:
                self.predictions_history.append(("<IDLE>", confidence, now))

            # 7. Evaluate Majority Voting & Cooldown Suppression
            return self._evaluate_voting_window()

        except Exception as e:
            logger.error(f"[SignClassifier] Inference error: {e}")

        return None

    def _evaluate_voting_window(self) -> Optional[Tuple[str, float]]:
        """
        Calculates weighted majority voting, checks confidence thresholds,
        and applies debounce/cooldown filtering to prevent duplicate triggers.
        """
        if len(self.predictions_history) < self.min_voting_count:
            return None

        # Group candidate votes, ignoring "<IDLE>" frames
        candidates: Dict[str, List[float]] = {}
        for label, conf, _ in self.predictions_history:
            if label != "<IDLE>":
                candidates.setdefault(label, []).append(conf)

        if not candidates:
            # Entire window is idle; clear sign lock if cooldown has passed
            if (time.time() - self._last_emitted_time) >= self.cooldown_seconds:
                self._last_emitted_label = None
            return None

        # Candidate with the highest vote count (tie-broken by weighted sum of confidences)
        best_label, conf_list = max(
            candidates.items(),
            key=lambda item: (len(item[1]), sum(item[1])),
        )
        vote_count = len(conf_list)
        avg_confidence = float(np.mean(conf_list))

        # Check voting conditions
        if vote_count < self.min_voting_count or avg_confidence < self.threshold:
            return None

        # Debounce / Cooldown filter: prevent duplicate spamming while sign is held
        now = time.time()
        is_same_as_last = best_label == self._last_emitted_label
        in_cooldown = (now - self._last_emitted_time) < self.cooldown_seconds

        if is_same_as_last and in_cooldown:
            logger.debug(
                f"[SignClassifier] Debounce active: suppressed duplicate '{best_label}' "
                f"({now - self._last_emitted_time:.2f}s < {self.cooldown_seconds}s)"
            )
            return None

        # Commit prediction
        self._last_emitted_label = best_label
        self._last_emitted_time = now

        logger.info(
            f"[SignClassifier] Recognized gesture: {best_label} "
            f"(conf: {avg_confidence * 100:.1f}%, votes: {vote_count}/{len(self.predictions_history)})"
        )

        if self.on_prediction:
            self.on_prediction(best_label, avg_confidence)

        return best_label, avg_confidence

    def inject_mock_prediction(
        self, label: str, confidence: float
    ) -> Optional[Tuple[str, float]]:
        """
        Directly injects a prediction into the voting window.
        Enables deterministic testing of voting and cooldown logic without vector synthesis.
        """
        now = time.time()
        if confidence >= self.threshold:
            self.predictions_history.append((label, confidence, now))
        else:
            self.predictions_history.append(("<IDLE>", confidence, now))
        return self._evaluate_voting_window()

    def reset(self) -> None:
        """Clears buffers and resets debounce/cooldown states."""
        self.frame_buffer.clear()
        self.predictions_history.clear()
        self._missing_hand_counter = 0
        self._frame_counter = 0
        self._last_emitted_label = None
        self._last_emitted_time = 0.0

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

    # Alias for holistic vector processing
    process_holistic_vector = process_hand_vector

