import asyncio
import logging
import time
from typing import Callable, Optional
import flet as ft
from services.audio_recorder import AudioRecorderService

logger = logging.getLogger(__name__)


class MicButton(ft.Column):
    """
    Reusable Microphone Button component.
    Toggles hardware audio recording, displays live duration, and emits WAV bytes.
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

        self.recorder = AudioRecorderService()
        self._timer_task: Optional[asyncio.Task] = None
        self._record_start_time: float = 0.0

        # UI Controls
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

    def did_mount(self):
        """Called by Flet when control is attached to the page tree."""
        if self.page:
            self.recorder.attach_to_page(self.page)

    def _safe_update(self):
        """Safely updates this control only when mounted."""
        if self.page:
            try:
                self.update()
            except Exception:
                pass

    async def _update_timer_loop(self):
        """Live recording duration ticker."""
        while self.recorder.is_recording:
            elapsed = int(time.time() - self._record_start_time)
            self.info_text.value = f"Recording: {elapsed}s"
            if self.page:
                try:
                    self.info_text.update()
                except Exception:
                    break
            await asyncio.sleep(0.5)

    async def _toggle_recording(self, e: ft.ControlEvent) -> None:
        if not self.recorder.is_recording:
            await self._start_recording()
        else:
            await self._stop_recording()

    async def _start_recording(self) -> None:
        if self.page:
            self.recorder.attach_to_page(self.page)

        try:
            started = await self.recorder.start()
            if started:
                self._record_start_time = time.time()
                self._set_active_ui(True)
                self._timer_task = asyncio.create_task(self._update_timer_loop())
            else:
                self.status_text.value = "Microphone permission required"
                self.status_text.color = ft.Colors.RED
                self._safe_update()
        except Exception as exc:
            logger.error(f"Error starting recording: {exc}")
            self.status_text.value = "Failed to start microphone"
            self.status_text.color = ft.Colors.RED
            self._safe_update()

    async def _stop_recording(self) -> None:
        if self._timer_task:
            self._timer_task.cancel()

        try:
            raw_bytes = await self.recorder.stop()
            self._set_active_ui(False)
            self.info_text.value = f"Finished recording: {len(raw_bytes):,} bytes"
            self._safe_update()

            if self.on_recorded:
                if asyncio.iscoroutinefunction(self.on_recorded):
                    await self.on_recorded(raw_bytes)
                else:
                    self.on_recorded(raw_bytes)
        except Exception as exc:
            logger.error(f"Error stopping recording: {exc}")
            self._set_active_ui(False)

    def _set_active_ui(self, active: bool) -> None:
        """Toggles styling between recording (pulsing red) and standby (grey)."""
        if active:
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
            self.mic_icon_button.icon = ft.Icons.MIC_NONE
            self.mic_icon_button.icon_color = ft.Colors.GREY_700
            self.mic_icon_button.tooltip = "Tap to record"
            self.mic_container.bgcolor = ft.Colors.GREY_200
            self.mic_container.border = ft.Border.all(2, ft.Colors.GREY_400)

            self.status_text.value = "Tap to record"
            self.status_text.color = ft.Colors.GREY_800
            self.info_text.color = ft.Colors.GREY_600

        self._safe_update()

    async def cleanup_async(self) -> None:
        """Halts background timers and cleans up recording if view navigates away."""
        if self._timer_task:
            self._timer_task.cancel()
        if self.recorder.is_recording:
            await self.recorder.stop()
            self._set_active_ui(False)