import io
import logging
import os
import tempfile
import urllib.parse
from urllib.request import url2pathname
from typing import Callable, Optional, Tuple
import flet as ft
from flet_audio_recorder import (
    AudioRecorder,
    AudioEncoder,
    AudioRecorderConfiguration,
)
import services.data as data

logger = logging.getLogger(__name__)


AUDIO_RECORDER_PAGE_KEY = "_app_audio_recorder_instance"


import wave


class AudioRecorderService:
    """
    Android and Desktop compatible Audio Recorder service using Flet 0.86+ Service API.
    Captures live audio streams in-memory (PCM16BITS) with WAV container packaging
    and fallback to file-based recording. Outputs 16kHz mono WAV byte data directly compatible with Sherpa-ONNX.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        on_data: Optional[Callable[[bytes], None]] = None,
        on_status_change: Optional[Callable[[bool], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.on_data = on_data
        self.on_status_change = on_status_change

        self._is_recording = False
        self._output_file = os.path.join(tempfile.gettempdir(), "audio_input.wav")
        self._recorded_bytes = 0
        self._last_recorded_path: Optional[str] = None
        self._stream_chunks: list[bytes] = []

        self.recorder: Optional[AudioRecorder] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def recorded_bytes_count(self) -> int:
        return self._recorded_bytes

    @property
    def last_recorded_path(self) -> Optional[str]:
        return self._last_recorded_path

    def _handle_stream_event(self, e) -> None:
        """Accumulates incoming PCM16 audio stream chunks."""
        if hasattr(e, "chunk") and e.chunk:
            chunk = e.chunk
            # Flutter/Flet might pass chunk as base64 string or bytes
            if isinstance(chunk, str):
                import base64
                try:
                    chunk = base64.b64decode(chunk)
                except Exception:
                    chunk = chunk.encode("latin1")
            self._stream_chunks.append(chunk)
            self._recorded_bytes += len(chunk)

    def attach_to_page(self, page: ft.Page) -> None:
        """Mounts recorder service to page services once and retains it across views."""
        if not page:
            return

        # Check if already retained on page
        shared_recorder = getattr(page, AUDIO_RECORDER_PAGE_KEY, None)
        if shared_recorder is None:
            shared_recorder = AudioRecorder(
                on_stream=self._handle_stream_event
            )
            setattr(page, AUDIO_RECORDER_PAGE_KEY, shared_recorder)
        else:
            shared_recorder.on_stream = self._handle_stream_event

        self.recorder = shared_recorder

        # Ensure it is registered in page.services
        if hasattr(page, "services"):
            if self.recorder not in page.services:
                page.services.append(self.recorder)
                try:
                    page.update()
                except Exception as e:
                    logger.debug(f"Error updating page after attaching recorder: {e}")

    async def start(self) -> bool:
        """Starts hardware recording in 16kHz mono audio stream."""
        if self._is_recording:
            return True

        try:
            # Check microphone permission
            try:
                has_perm = await self.recorder.has_permission()
            except Exception as perm_err:
                logger.debug(f"has_permission check error: {perm_err}")
                has_perm = True

            self._stream_chunks.clear()
            self._recorded_bytes = 0

            if os.path.exists(self._output_file):
                try:
                    os.remove(self._output_file)
                except Exception:
                    pass

            # Primary attempt: Stream raw PCM16 audio chunks directly in-memory
            # (required when on_stream is registered)
            config = AudioRecorderConfiguration(
                encoder=AudioEncoder.PCM16BITS,
                sample_rate=self.sample_rate,
                channels=1,
            )

            try:
                await self.recorder.start_recording(
                    output_path=self._output_file,
                    configuration=config,
                )
            except Exception as stream_start_err:
                logger.debug(f"Stream start fallback to WAV: {stream_start_err}")
                # Fallback to WAV file recording if platform does not support PCM16BITS streaming
                self.recorder.on_stream = None
                config = AudioRecorderConfiguration(
                    encoder=AudioEncoder.WAV,
                    sample_rate=self.sample_rate,
                    channels=1,
                )
                await self.recorder.start_recording(
                    output_path=self._output_file,
                    configuration=config,
                )

            self._is_recording = True
            data.is_recording_audio = True

            if self.on_status_change:
                self.on_status_change(True)

            return True

        except Exception as exc:
            logger.error(f"Failed to start recording: {exc}")
            self._is_recording = False
            data.is_recording_audio = False
            if self.on_status_change:
                self.on_status_change(False)
            raise exc

    def _pack_pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """Packs raw PCM16 samples into standard 16kHz mono WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    async def stop(self) -> bytes:
        """Stops hardware recording and returns WAV bytes from stream or file."""
        if not self._is_recording:
            return getattr(data, "raw_audio_data", b"")

        try:
            output_url = await self.recorder.stop_recording()

            self._is_recording = False
            data.is_recording_audio = False

            if self.on_status_change:
                self.on_status_change(False)

            # 1. If audio stream chunks were captured in-memory, package directly into WAV
            if self._stream_chunks:
                raw_pcm = b"".join(self._stream_chunks)
                if len(raw_pcm) > 0:
                    wav_bytes = self._pack_pcm_to_wav(raw_pcm)
                    self._recorded_bytes = len(wav_bytes)
                    data.raw_audio_data = wav_bytes

                    # Save to _output_file so sherpa read_wave or cache can access it if needed
                    try:
                        with open(self._output_file, "wb") as f:
                            f.write(wav_bytes)
                        self._last_recorded_path = self._output_file
                    except Exception:
                        pass

                    print(f"[AUDIO RECORDER] Successfully assembled {len(wav_bytes):,} WAV bytes from live stream chunks.", flush=True)
                    return wav_bytes

            # 2. Fallback to file reading if streaming chunks weren't emitted
            def _clean_path(path_or_uri: Optional[str]) -> str:
                if not path_or_uri:
                    return ""
                cleaned = str(path_or_uri).strip()
                if cleaned.startswith("file://"):
                    parsed = urllib.parse.urlparse(cleaned)
                    return url2pathname(parsed.path)
                return cleaned

            resolved_path = _clean_path(output_url)

            if not resolved_path or not os.path.exists(resolved_path):
                fallback_path = _clean_path(self._output_file)
                if os.path.exists(fallback_path):
                    resolved_path = fallback_path

            self._last_recorded_path = resolved_path
            print(f"[AUDIO RECORDER] Stop result raw: '{output_url}', resolved: '{resolved_path}'", flush=True)

            if resolved_path and os.path.exists(resolved_path):
                with open(resolved_path, "rb") as f:
                    raw_bytes = f.read()

                self._recorded_bytes = len(raw_bytes)
                data.raw_audio_data = raw_bytes
                print(f"[AUDIO RECORDER] Successfully read {len(raw_bytes):,} bytes from {resolved_path}", flush=True)
                return raw_bytes

            logger.warning(f"[AUDIO RECORDER] No audio data captured.")
            return b""

        except Exception as exc:
            logger.error(f"Error stopping recording: {exc}")
            self._is_recording = False
            data.is_recording_audio = False
            if self.on_status_change:
                self.on_status_change(False)
            return b""

    def get_waveform_for_sherpa(self) -> Tuple[Optional[object], int]:
        """
        Reads the 16kHz WAV file using Sherpa-ONNX's built-in C++ wave parser.
        No soundfile/libsndfile C libraries required on Android.
        """
        if not self._last_recorded_path or not os.path.exists(self._last_recorded_path):
            return None, self.sample_rate

        try:
            import sherpa_onnx

            wave = sherpa_onnx.read_wave(self._last_recorded_path)
            return wave.samples, wave.sample_rate
        except Exception as err:
            logger.error(f"Failed to decode audio using sherpa-onnx: {err}")
            return None, self.sample_rate