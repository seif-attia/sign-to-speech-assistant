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
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar
from components.mic_button import MicButton
from services.fastapi_endpoint import translate_text, play_speech

class SpeechToSpeechView(ft.View):
    """
    Screen 5: Speech-to-Speech Mode View.
    Renders 'Speech Mode', animated soundwave visualizer,
    spoken output buffer, microphone recording, translation, and Sherpa-ONNX voice response.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page
        self.is_ar_to_en = True  # True: Ar -> En, False: En -> Ar

        async def go_back(e):
            await self.cleanup_async()
            await self.app_page.push_route("/")

        super().__init__(
            route="/speech-speech",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=16, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(0, self.app_page),
        )

        def _toggle_direction(e):
            self.is_ar_to_en = not self.is_ar_to_en
            self.lang_mode_text.value = "Ar ➔ En" if self.is_ar_to_en else "En ➔ Ar"
            self.listening_state.value = (
                "Listening in Arabic..." if self.is_ar_to_en else "Listening in English..."
            )
            self.recognized_output.value = (
                '"تحدث الآن للترجمة الفورية"' if self.is_ar_to_en else '"Speak now for live translation"'
            )
            self.app_page.update()

        self.lang_mode_text = ft.Text("Ar ➔ En", size=11, weight=ft.FontWeight.BOLD, color=ACCENT_MINT)
        lang_toggle_btn = ft.Container(
            content=self.lang_mode_text,
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            border_radius=12,
            bgcolor="#132B26",
            border=ft.Border.all(1, "#1D473F"),
            on_click=_toggle_direction,
        )

        header = create_header(
            title="Speech Mode",
            on_back=go_back,
            right_control=lang_toggle_btn,
        )

        self.listening_state = ft.Text(
            "Listening in Arabic...",
            size=14,
            color=TEXT_SECONDARY,
            weight=ft.FontWeight.W_500,
        )

        # Visualizer bars simulation
        wave_heights = [18, 32, 48, 24, 60, 42, 70, 30, 52, 22, 40, 16]
        wave_bars = ft.Row(
            controls=[
                ft.Container(
                    width=4,
                    height=h,
                    bgcolor=ACCENT_EMERALD if i % 2 == 0 else ACCENT_MINT,
                    border_radius=2,
                )
                for i, h in enumerate(wave_heights)
            ],
            spacing=5,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self.recognized_output = ft.Text(
            '"تحدث الآن للترجمة الفورية"',
            size=16,
            color=TEXT_PRIMARY,
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER,
        )

        self.translated_output = ft.Text(
            "",
            size=14,
            color=ACCENT_MINT,
            weight=ft.FontWeight.W_500,
            text_align=ft.TextAlign.CENTER,
        )

        visualizer_card = create_glass_card(
            content=ft.Column(
                controls=[
                    self.listening_state,
                    ft.Container(content=wave_bars, height=80, alignment=ft.Alignment.CENTER),
                    self.recognized_output,
                    self.translated_output,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=14,
            ),
            padding=20,
            border_radius=20,
            width=VIEWPORT_WIDTH,
        )

        async def _process_voice_input(wav_bytes: bytes):
            if not wav_bytes:
                return

            self.listening_state.value = "Translating & Synthesizing Voice..."
            self.app_page.update()

            # Process demo / live translation pair
            if self.is_ar_to_en:
                spoken = "صباح الخير، كيف حالك؟"
                translated = await translate_text(spoken, source_lang="ar", target_lang="en")
                target_lang = "en"
            else:
                spoken = "Good morning, how are you?"
                translated = await translate_text(spoken, source_lang="en", target_lang="ar")
                target_lang = "ar"

            self.recognized_output.value = f'"{spoken}"'
            self.translated_output.value = f'➔ "{translated}"'
            self.listening_state.value = "Speaking Translation via Sherpa-ONNX..."
            self.app_page.update()

            self.app_page.run_task(play_speech, self.app_page, translated, target_lang)

            self.listening_state.value = (
                "Listening in Arabic..." if self.is_ar_to_en else "Listening in English..."
            )
            self.app_page.update()

        def on_audio_recorded(wav_bytes: bytes):
            self.app_page.run_task(_process_voice_input, wav_bytes)

        self.mic_button = MicButton(on_recorded=on_audio_recorded)

        tap_pause_label = ft.Text(
            "TAP TO START / STOP LISTENING",
            size=11,
            weight=ft.FontWeight.BOLD,
            color=TEXT_MUTED,
            style=ft.TextStyle(letter_spacing=1.5),
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    ft.Container(height=12),
                    visualizer_card,
                    ft.Container(height=28),
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
