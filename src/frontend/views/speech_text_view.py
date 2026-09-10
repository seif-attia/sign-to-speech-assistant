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

        self.active_lang = "ar"  # Default listening language

        def _toggle_lang(e):
            self.active_lang = "en" if self.active_lang == "ar" else "ar"
            self.lang_mode_text.value = "Input: Arabic" if self.active_lang == "ar" else "Input: English"
            self.transcription_text.value = (
                "تحدث الآن باللغة العربية للتعرف على الصوت..." if self.active_lang == "ar"
                else "Speak now in English to start transcribing..."
            )
            self.app_page.update()

        self.lang_mode_text = ft.Text("Input: Arabic", size=11, weight=ft.FontWeight.BOLD, color=ACCENT_MINT)
        lang_toggle_btn = ft.Container(
            content=self.lang_mode_text,
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            border_radius=12,
            bgcolor="#132B26",
            border=ft.Border.all(1, "#1D473F"),
            on_click=_toggle_lang,
        )

        header = create_header(
            title="Speech to Text",
            on_back=go_back,
            right_control=lang_toggle_btn,
        )

        # Transcribing container card
        self.live_badge_text = ft.Text("READY TO LISTEN", size=10, weight=ft.FontWeight.BOLD, color=ACCENT_MINT)
        self.live_badge = ft.Container(
            content=ft.Row(
                controls=[self.live_badge_text],
                spacing=4,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            bgcolor="#103B33",
            border_radius=12,
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        )
        self.transcription_text = ft.Text(
            "تحدث الآن باللغة العربية للتعرف على الصوت...\nسيتم التقاط صوتك ومعالجته مباشرة.",
            size=14,
            color=TEXT_SECONDARY,
            italic=True,
        )

        transcribe_card = create_glass_card(
            content=ft.Column(
                controls=[
                    self.live_badge,
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

        async def _handle_recorded_audio(wav_bytes: bytes):
            if not wav_bytes or len(wav_bytes) == 0:
                self.transcription_text.value = "No audio detected. Please try speaking again."
                self.app_page.update()
                return

            self.live_badge_text.value = "PROCESSING AUDIO..."
            self.transcription_text.value = f"Captured {len(wav_bytes):,} bytes of audio. Processing..."
            self.app_page.update()

            # For demonstration & backend ASR integration:
            # We provide a prompt translation / transcription response
            sample_phrase = (
                "مرحباً بكم، تم تسجيل الصوت بنجاح وجاري المعالجة الفورية." if self.active_lang == "ar"
                else "Hello, audio recorded successfully and is being processed."
            )
            self.transcription_text.value = f'"{sample_phrase}"'
            self.live_badge_text.value = "TRANSCRIPTION COMPLETE"
            self.app_page.update()

        def on_audio_recorded(wav_bytes: bytes):
            self.app_page.run_task(_handle_recorded_audio, wav_bytes)

        # Dedicated Mic Button Component
        self.mic_button = MicButton(on_recorded=on_audio_recorded)

        tap_pause_label = ft.Text(
            "TAP TO START / STOP RECORDING",
            size=11,
            weight=ft.FontWeight.BOLD,
            color=TEXT_MUTED,
            style=ft.TextStyle(letter_spacing=1.5),
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    ft.Container(height=16),
                    transcribe_card,
                    ft.Container(height=30),
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
