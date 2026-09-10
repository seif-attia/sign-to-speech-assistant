import asyncio
import logging
import os
import tempfile
from typing import Callable, Optional
import flet as ft
from flet_audio_recorder import AudioRecorder, AudioEncoder

import services.data as data

logger = logging.getLogger(__name__)


class AudioRecorderService:
    """
    Android-compatible Audio Recorder service using native Flutter platform channels.
    Outputs WAV format byte data and supports asynchronous start/stop.
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

        # Flutter AudioRecorder instance
        self.recorder = AudioRecorder(
            audio_encoder=AudioEncoder.WAV,
            sample_rate=self.sample_rate,
        )

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def recorded_bytes_count(self) -> int:
        return self._recorded_bytes

    def attach_to_page(self, page: ft.Page) -> None:
        """Mounts recorder to page overlay to enable native channel communication."""
        if page and self.recorder not in page.overlay:
            page.overlay.append(self.recorder)
            page.update()

    async def start(self) -> bool:
        """Starts hardware microphone recording."""
        if self._is_recording:
            return True

        try:
            # Check runtime permission
            has_perm = await self.recorder.has_permission_async()
            if not has_perm:
                logger.warning("Microphone permission denied.")
                return False

            if os.path.exists(self._output_file):
                os.remove(self._output_file)

            await self.recorder.start_recording_async(self._output_file)
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
            return False

    async def stop(self) -> bytes:
        """Stops hardware recording and returns all captured WAV audio bytes."""
        if not self._is_recording:
            return getattr(data, "raw_audio_data", b"")

        try:
            output_url = await self.recorder.stop_recording_async()
            self._is_recording = False
            data.is_recording_audio = False

            if self.on_status_change:
                self.on_status_change(False)

            target = output_url or self._output_file
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
            return b""