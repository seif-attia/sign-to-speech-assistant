import asyncio
import logging
import time
from typing import Callable, Optional
import flet as ft
from services.audio_recorder import AudioRecorderService
from frontend.theme import (
    BG_SURFACE,
    CARD_BG,
    CARD_BORDER,
    ACCENT_EMERALD,
    ACCENT_MINT,
    ACCENT_GREEN_GLOW,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)

logger = logging.getLogger(__name__)


class MicButton(ft.Column):
    """
    Reusable Microphone Button component styled to match WESAL's
    Obsidian / Emerald Glassmorphic design system.
    Toggles hardware audio recording, displays live duration, and emits WAV bytes.
    """

    def __init__(
        self,
        on_recorded: Optional[Callable[[bytes], None]] = None,
        button_size: int = 96,
        icon_size: int = 42,
    ):
        super().__init__()

        self.on_recorded = on_recorded
        self.button_size = button_size
        self.icon_size = icon_size
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.alignment = ft.MainAxisAlignment.CENTER
        self.spacing = 14

        self.recorder = AudioRecorderService()
        self._timer_task: Optional[asyncio.Task] = None
        self._record_start_time: float = 0.0

        # UI Controls
        self.mic_icon = ft.Icon(
            ft.Icons.MIC_ROUNDED,
            color=ACCENT_MINT,
            size=self.icon_size,
        )

        self.mic_container = ft.Container(
            content=self.mic_icon,
            width=self.button_size,
            height=self.button_size,
            border_radius=self.button_size // 2,
            bgcolor=CARD_BG,
            border=ft.Border.all(2, ACCENT_EMERALD),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=16,
                color=ft.Colors.with_opacity(0.35, ACCENT_EMERALD),
                offset=ft.Offset(0, 4),
            ),
            alignment=ft.Alignment.CENTER,
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
            on_click=self._toggle_recording,
            ink=True,
        )

        self.status_text = ft.Text(
            value="Tap to record",
            size=15,
            weight=ft.FontWeight.BOLD,
            color=TEXT_PRIMARY,
            text_align=ft.TextAlign.CENTER,
        )

        self.info_text = ft.Text(
            value="Ready to listen",
            size=12,
            color=TEXT_MUTED,
            text_align=ft.TextAlign.CENTER,
        )

        self.controls = [
            self.mic_container,
            ft.Column(
                controls=[self.status_text, self.info_text],
                spacing=3,
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
                self.status_text.color = ft.Colors.RED_400
                self._safe_update()
        except Exception as exc:
            logger.error(f"Error starting recording: {exc}")
            self.status_text.value = "Failed to start microphone"
            self.status_text.color = ft.Colors.RED_400
            self._safe_update()

    async def _stop_recording(self) -> None:
        if self._timer_task:
            self._timer_task.cancel()

        try:
            self.status_text.value = "Processing audio..."
            self.status_text.color = ACCENT_MINT
            self._safe_update()

            raw_bytes = await self.recorder.stop()
            self._set_active_ui(False)

            if len(raw_bytes) > 0:
                self.info_text.value = f"Captured {len(raw_bytes):,} bytes"
                self.info_text.color = ACCENT_MINT
            else:
                self.info_text.value = "No audio detected (0 bytes)"
                self.info_text.color = ft.Colors.RED_400

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
        """Toggles styling between recording (pulsing crimson glow) and standby (emerald glow)."""
        if active:
            self.mic_icon.icon = ft.Icons.STOP_ROUNDED
            self.mic_icon.color = ft.Colors.RED_ACCENT_400
            self.mic_container.bgcolor = "#281212"
            self.mic_container.border = ft.Border.all(2, ft.Colors.RED_ACCENT_400)
            self.mic_container.shadow = ft.BoxShadow(
                spread_radius=2,
                blur_radius=22,
                color=ft.Colors.with_opacity(0.5, ft.Colors.RED_ACCENT_700),
                offset=ft.Offset(0, 4),
            )

            self.status_text.value = "Recording..."
            self.status_text.color = ft.Colors.RED_ACCENT_200
            self.info_text.value = "Tap to stop"
            self.info_text.color = ft.Colors.RED_300
        else:
            self.mic_icon.icon = ft.Icons.MIC_ROUNDED
            self.mic_icon.color = ACCENT_MINT
            self.mic_container.bgcolor = CARD_BG
            self.mic_container.border = ft.Border.all(2, ACCENT_EMERALD)
            self.mic_container.shadow = ft.BoxShadow(
                spread_radius=1,
                blur_radius=16,
                color=ft.Colors.with_opacity(0.35, ACCENT_EMERALD),
                offset=ft.Offset(0, 4),
            )

            self.status_text.value = "Tap to record"
            self.status_text.color = TEXT_PRIMARY
            self.info_text.value = "Ready to listen"
            self.info_text.color = TEXT_MUTED

        self._safe_update()

    async def cleanup_async(self) -> None:
        """Halts background timers and cleans up recording if view navigates away."""
        if self._timer_task:
            self._timer_task.cancel()
        if self.recorder.is_recording:
            await self.recorder.stop()
            self._set_active_ui(False)