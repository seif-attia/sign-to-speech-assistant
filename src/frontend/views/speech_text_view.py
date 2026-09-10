import flet as ft
from frontend.theme import (
    BG_DARK,
    CARD_BG,
    CARD_BORDER,
    ACCENT_EMERALD,
    ACCENT_MINT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    create_glass_card,
    create_header,
    create_lang_toggle,
    create_pill_badge,
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar
from components.mic_button import MicButton

class SpeechToTextView(ft.View):
    """
    Screen 6: Speech-to-Text View.
    Displays live transcribing status, active audio waveform effect,
    recognized speech buffer, and tap-to-pause microphone button.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        async def go_back(e):
            await self.cleanup_async()
            await self.app_page.push_route("/")

        super().__init__(
            route="/speech-text",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=16, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(0, self.app_page),
        )

        header = create_header(
            title="Speech to Text",
            on_back=go_back,
            right_control=create_lang_toggle("Ar ⇄ En"),
        )

        # Transcribing container card
        self.transcription_text = ft.Text(
            "Speak now to start translating...\nWaveform bars will mirror your voice modulation in real-time.",
            size=14,
            color=TEXT_SECONDARY,
            italic=True,
        )

        transcribe_card = create_glass_card(
            content=ft.Column(
                controls=[
                    create_pill_badge("LIVE TRANSCRIBING...", font_size=10),
                    ft.Container(height=8),
                    self.transcription_text,
                ],
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.START,
            ),
            padding=18,
            border_radius=18,
            width=VIEWPORT_WIDTH,
        )

        # Dedicated Mic Button Component
        self.mic_button = MicButton()

        tap_pause_label = ft.Text(
            "TAP TO PAUSE LISTENING",
            size=11,
            weight=ft.FontWeight.BOLD,
            color=TEXT_MUTED,
            letter_spacing=1.5,
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    ft.Container(height=16),
                    transcribe_card,
                    ft.Container(height=40),
                    self.mic_button,
                    ft.Container(height=8),
                    tap_pause_label,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
            )
        ]

    async def cleanup_async(self):
        if hasattr(self, "mic_button") and self.mic_button:
            await self.mic_button.cleanup_async()
