import logging
import threading
from typing import Callable, Optional
import sounddevice as sd
import services.data as data

logger = logging.getLogger(__name__)


class AudioRecorderService:
    """
    Service that captures microphone input using sounddevice, converts the stream
    into raw 16-bit PCM byte data, and maintains global state in services.data.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        on_data: Optional[Callable[[bytes], None]] = None,
        on_status_change: Optional[Callable[[bool], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.on_data = on_data
        self.on_status_change = on_status_change

        self._stream: Optional[sd.RawInputStream] = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def recorded_bytes_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback invoked by PortAudio background thread for each incoming buffer."""
        if status:
            logger.warning(f"Audio stream status flag: {status}")

        chunk = bytes(indata)
        with self._lock:
            self._buffer.extend(chunk)
            data.raw_audio_data = bytes(self._buffer)

        if self.on_data:
            try:
                self.on_data(chunk)
            except Exception as e:
                logger.error(f"Error in on_data callback: {e}")

    def start(self) -> bool:
        """
        Starts recording audio. Returns True if started successfully, False otherwise.
        """
        with self._lock:
            if self._is_recording:
                return True

            try:
                self._buffer.clear()
                data.raw_audio_data = b""
                data.is_recording_audio = True

                self._stream = sd.RawInputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_recording = True
            except Exception as exc:
                logger.error(f"Failed to start audio recording stream: {exc}")
                data.is_recording_audio = False
                self._is_recording = False
                if self._stream:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                raise exc

        if self.on_status_change:
            try:
                self.on_status_change(True)
            except Exception as e:
                logger.error(f"Error in on_status_change callback: {e}")

        return True

    def stop(self) -> bytes:
        """
        Stops audio recording and returns all captured raw byte data.
        """
        with self._lock:
            if not self._is_recording:
                return bytes(self._buffer)

            self._is_recording = False
            data.is_recording_audio = False

            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as exc:
                    logger.warning(f"Error closing audio stream: {exc}")
                finally:
                    self._stream = None

            result = bytes(self._buffer)
            data.raw_audio_data = result

        if self.on_status_change:
            try:
                self.on_status_change(False)
            except Exception as e:
                logger.error(f"Error in on_status_change callback: {e}")

        return result

    def get_raw_data(self) -> bytes:
        """Returns the current raw byte buffer."""
        with self._lock:
            return bytes(self._buffer)

    def clear(self) -> None:
        """Clears the internal buffer and global raw audio data."""
        with self._lock:
            self._buffer.clear()
            data.raw_audio_data = b""

    def close(self) -> None:
        """Cleanup resources."""
        self.stop()
