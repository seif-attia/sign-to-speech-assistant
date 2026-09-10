"""
Offline Neural Text-to-Speech Service using Sherpa-ONNX (VITS / Piper).
Provides low-latency, natural human voice synthesis with graceful fallback.
"""
import logging
from pathlib import Path
import threading
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TTS_DIR = BASE_DIR / "assets" / "tts"


class SherpaTTSService:
    """
    Offline Text-to-Speech Engine using Sherpa-ONNX.
    Falls back gracefully to system TTS (pyttsx3) if neural model files are not yet present.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_TTS_DIR
        self.tts = None
        self.is_neural_ready = False
        self._lock = threading.Lock()
        self._is_speaking = False

        self._pyttsx3_engine = None
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
        except Exception:
            pass

        self._init_engine()

    def _init_engine(self):
        """Attempts to load Sherpa-ONNX VITS model from the tts assets directory."""
        try:
            import os
            if os.name == "nt":
                try:
                    import onnxruntime
                    capi = Path(onnxruntime.__file__).parent / "capi"
                    if capi.exists():
                        os.add_dll_directory(str(capi))
                except Exception:
                    pass

            import sherpa_onnx

            if not self.model_dir.exists():
                logger.info(f"[SherpaTTS] TTS directory '{self.model_dir}' not found. Using system TTS fallback.")
                return

            # Search for onnx model, tokens, and data files
            onnx_files = list(self.model_dir.glob("*.onnx"))
            tokens_files = list(self.model_dir.glob("*tokens*.txt"))
            espeak_dirs = [d for d in self.model_dir.glob("*espeak*") if d.is_dir()]

            if not onnx_files or not tokens_files:
                logger.info("[SherpaTTS] VITS model weights not yet found in assets/tts. Using fallback.")
                return

            model_path = str(onnx_files[0])
            tokens_path = str(tokens_files[0])
            data_dir = str(espeak_dirs[0]) if espeak_dirs else ""

            logger.info(f"[SherpaTTS] Initializing Sherpa-ONNX model from {model_path}...")
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_path,
                tokens=tokens_path,
                data_dir=data_dir,
            )
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(vits=vits_config),
            )

            if not config.validate():
                logger.warning("[SherpaTTS] Sherpa-ONNX config validation failed.")
                return

            self.tts = sherpa_onnx.OfflineTts(config)
            self.is_neural_ready = True
            logger.info("[SherpaTTS] Neural VITS engine successfully loaded!")

        except Exception as e:
            logger.warning(f"[SherpaTTS] Sherpa-ONNX initialization skipped: {e}")
            self.tts = None
            self.is_neural_ready = False

    def speak(
        self,
        text: str,
        lang: str = "en",
        sid: int = 0,
        speed: float = 1.0,
        on_done=None,
    ):
        """
        Asynchronously speaks the specified text in English or Arabic without blocking the UI thread.
        """
        text = text.strip()
        if not text:
            return

        self.stop()

        is_arabic = (
            lang.lower().startswith("ar")
            or any("\u0600" <= c <= "\u06FF" for c in text)
        )

        def _worker():
            with self._lock:
                self._is_speaking = True

            try:
                # 1. Neural Sherpa-ONNX Synthesis (for English when VITS model is loaded)
                if not is_arabic and self.is_neural_ready and self.tts is not None:
                    import sounddevice as sd
                    audio = self.tts.generate(text, sid=sid, speed=speed)
                    if audio and len(audio.samples) > 0:
                        sd.play(audio.samples, samplerate=audio.sample_rate)
                        sd.wait()
                        return

                # 2. Multilingual Fallback via system TTS (pyttsx3)
                try:
                    engine = self._pyttsx3_engine
                    if engine is None:
                        import pyttsx3
                        engine = pyttsx3.init()
                        self._pyttsx3_engine = engine

                    engine.setProperty("rate", int(170 * speed))

                    # Select Arabic voice if available
                    if is_arabic:
                        try:
                            voices = engine.getProperty("voices")
                            for v in voices:
                                v_id = getattr(v, "id", "").lower()
                                v_name = getattr(v, "name", "").lower()
                                if any(ar_kw in v_id or ar_kw in v_name for ar_kw in ["ar", "arabic", "hoda", "naayf", "tarik", "laila", "maged"]):
                                    engine.setProperty("voice", v.id)
                                    break
                        except Exception:
                            pass

                    engine.say(text)
                    engine.runAndWait()
                except Exception as pyttsx_err:
                    logger.debug(f"[SherpaTTS] Fallback TTS note: {pyttsx_err}")

            except Exception as e:
                logger.error(f"[SherpaTTS] Error during speech synthesis: {e}")
            finally:
                with self._lock:
                    self._is_speaking = False
                if on_done:
                    try:
                        on_done()
                    except Exception:
                        pass

        threading.Thread(target=_worker, daemon=True).start()

    def stop(self):
        """Halts current audio playback."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        try:
            if self._pyttsx3_engine is not None:
                self._pyttsx3_engine.stop()
        except Exception:
            pass
