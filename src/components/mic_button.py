import asyncio
import logging
from typing import Callable, Optional
import flet as ft
import flet_permission_handler as fh
from services.audio_recorder import AudioRecorderService
import services.data as data

logger = logging.getLogger(__name__)


class MicButton(ft.Column):
    """
    Reusable Microphone Button component.
    Toggles audio recording, streams raw 16-bit PCM bytes into global data.py,
    and visually indicates active (red) vs inactive (grey/black) state.
    """

    def __init__(
        self,
        on_recorded: Optional[Callable[[bytes], None]] = None,
        button_size: int = 100,
        icon_size: int = 48,
    ):
        super().__init__()

        self.on_recorded = on_recorded
        self.button_size = button_size
        self.icon_size = icon_size
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.alignment = ft.MainAxisAlignment.CENTER
        self.spacing = 16

        # Permission handler
        self.permission_handler = fh.PermissionHandler()

        # Audio service
        self.recorder = AudioRecorderService(on_data=self._on_audio_data)

        # Reactive controls
        self.mic_icon_button = ft.IconButton(
            icon=ft.Icons.MIC_NONE,
            icon_color=ft.Colors.GREY_700,
            icon_size=self.icon_size,
            tooltip="Tap to record",
            on_click=self._toggle_recording,
        )

        self.mic_container = ft.Container(
            content=self.mic_icon_button,
            width=self.button_size,
            height=self.button_size,
            border_radius=self.button_size // 2,
            bgcolor=ft.Colors.GREY_200,
            border=ft.Border.all(2, ft.Colors.GREY_400),
            alignment=ft.Alignment.CENTER,
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        )

        self.status_text = ft.Text(
            value="Tap to record",
            size=16,
            weight=ft.FontWeight.W_600,
            color=ft.Colors.GREY_800,
            text_align=ft.TextAlign.CENTER,
        )

        self.info_text = ft.Text(
            value="Audio ready (0 bytes)",
            size=12,
            color=ft.Colors.GREY_600,
            text_align=ft.TextAlign.CENTER,
        )

        self.controls = [
            self.mic_container,
            ft.Column(
                controls=[self.status_text, self.info_text],
                spacing=4,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ]

    def _on_audio_data(self, chunk: bytes) -> None:
        """Callback from recorder thread on each audio chunk."""
        # Update text if still mounted and recording
        if self.page and self.recorder.is_recording:
            count = self.recorder.recorded_bytes_count
            self.info_text.value = f"Captured {count:,} bytes"
            try:
                self.info_text.update()
            except Exception:
                pass

    async def _toggle_recording(self, e: ft.ControlEvent) -> None:
        """Toggle recording state when mic button is pressed."""
        if not self.recorder.is_recording:
            await self._start_recording()
        else:
            await self._stop_recording()

    async def _start_recording(self) -> None:
        """Requests permission and starts recording."""
        # Request microphone permission if applicable
        try:
            has_permission = await self.permission_handler.request(
                fh.Permission.MICROPHONE
            )
            if has_permission is False:
                self.status_text.value = "Microphone permission denied"
                self.status_text.color = ft.Colors.RED
                self.update()
                return
        except Exception:
            pass  # Fallback for environments without mobile permission handler

        try:
            self.recorder.start()
            self._set_active_ui(True)
        except Exception as exc:
            logger.error(f"Error starting recording: {exc}")
            self.status_text.value = "Failed to start microphone"
            self.status_text.color = ft.Colors.RED
            self.update()

    async def _stop_recording(self) -> None:
        """Stops recording and finalizes raw audio data."""
        try:
            raw_bytes = self.recorder.stop()
            self._set_active_ui(False)
            self.info_text.value = f"Finished recording: {len(raw_bytes):,} bytes"
            self.info_text.update()

            if self.on_recorded:
                self.on_recorded(raw_bytes)
        except Exception as exc:
            logger.error(f"Error stopping recording: {exc}")
            self._set_active_ui(False)

    def _set_active_ui(self, active: bool) -> None:
        """Updates UI styling for active (recording) vs inactive (idle) states."""
        if active:
            # Active state: red mic icon, red accented ring/background
            self.mic_icon_button.icon = ft.Icons.MIC
            self.mic_icon_button.icon_color = ft.Colors.RED
            self.mic_icon_button.tooltip = "Tap to stop recording"
            self.mic_container.bgcolor = ft.Colors.RED_50
            self.mic_container.border = ft.Border.all(3, ft.Colors.RED)

            self.status_text.value = "Recording... Tap to stop"
            self.status_text.color = ft.Colors.RED_700
            self.info_text.value = "Listening..."
            self.info_text.color = ft.Colors.RED_400
        else:
            # Inactive state: black / grey mic icon, neutral background
            self.mic_icon_button.icon = ft.Icons.MIC_NONE
            self.mic_icon_button.icon_color = ft.Colors.GREY_700
            self.mic_icon_button.tooltip = "Tap to record"
            self.mic_container.bgcolor = ft.Colors.GREY_200
            self.mic_container.border = ft.Border.all(2, ft.Colors.GREY_400)

            self.status_text.value = "Tap to record"
            self.status_text.color = ft.Colors.GREY_800
            self.info_text.color = ft.Colors.GREY_600

        try:
            self.update()
        except Exception:
            pass

    async def cleanup_async(self) -> None:
        """Cleanup audio stream if the view is popped or unmounted."""
        if self.recorder.is_recording:
            self.recorder.stop()
            self._set_active_ui(False)
