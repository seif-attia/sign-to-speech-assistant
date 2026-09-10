import io
import logging
import os
import tempfile
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


class AudioRecorderService:
    """
    Android-compatible Audio Recorder service using Flet 0.86+ Service API.
    Outputs 16kHz mono WAV byte data directly compatible with Sherpa-ONNX.
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

    def attach_to_page(self, page: ft.Page) -> None:
        """Mounts recorder service to page services once and retains it across views."""
        if not page:
            return

        # Check if already retained on page
        shared_recorder = getattr(page, AUDIO_RECORDER_PAGE_KEY, None)
        if shared_recorder is None:
            shared_recorder = AudioRecorder()
            setattr(page, AUDIO_RECORDER_PAGE_KEY, shared_recorder)

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
        """Starts hardware recording in 16kHz mono WAV for Sherpa-ONNX."""
        if self._is_recording:
            return True

        try:
            # Check microphone permission (0.86 API). If False or unsupported on platform,
            # attempt to start_recording directly as mobile OS will prompt or desktop may not support has_permission()
            try:
                has_perm = await self.recorder.has_permission()
            except Exception as perm_err:
                logger.debug(f"has_permission check error: {perm_err}")
                has_perm = True

            if os.path.exists(self._output_file):
                try:
                    os.remove(self._output_file)
                except Exception:
                    pass

            # Flet 0.86 passes audio settings via AudioRecorderConfiguration
            config = AudioRecorderConfiguration(
                encoder=AudioEncoder.WAV,
                sample_rate=self.sample_rate,
                channels=1,
            )

            res = await self.recorder.start_recording(
                output_path=self._output_file,
                configuration=config,
            )

            self._is_recording = True
            self._recorded_bytes = 0
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

    async def stop(self) -> bytes:
        """Stops hardware recording and returns WAV bytes."""
        if not self._is_recording:
            return getattr(data, "raw_audio_data", b"")

        try:
            output_url = await self.recorder.stop_recording()

            self._is_recording = False
            data.is_recording_audio = False

            if self.on_status_change:
                self.on_status_change(False)

            target = output_url or self._output_file
            if target and target.startswith("file://"):
                target = target[7:]

            self._last_recorded_path = target

            if target and os.path.exists(target):
                with open(target, "rb") as f:
                    raw_bytes = f.read()

                self._recorded_bytes = len(raw_bytes)
                data.raw_audio_data = raw_bytes
                return raw_bytes

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