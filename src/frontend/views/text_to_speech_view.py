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
    INPUT_BG,
    create_glass_card,
    create_header,
    create_lang_toggle,
    create_pill_badge,
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar
import threading

class TextToSpeechView(ft.View):
    """
    Screen 7: Text-to-Speech View.
    Allows typing phrase, synthesizer engagement graphic, and 'Generate & Speak' button.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        async def go_back(e):
            await self.app_page.push_route("/")

        super().__init__(
            route="/text-speech",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=16, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(0, self.app_page),
        )

        header = create_header(
            title="Text to Speech",
            on_back=go_back,
            right_control=create_lang_toggle("Ar ⇄ En"),
        )

        # Synthesizer Engaged Visual
        synth_badge = create_pill_badge("SYNTHESIZER ENGAGED", font_size=10)
        speaker_icon_circle = ft.Container(
            content=ft.Icon(ft.Icons.VOLUME_UP_ROUNDED, size=40, color=ACCENT_MINT),
            width=80,
            height=80,
            border_radius=40,
            bgcolor="#132B2B",
            border=ft.Border.all(2, ACCENT_EMERALD),
            alignment=ft.Alignment.CENTER,
        )

        synth_header_card = ft.Container(
            content=ft.Column(
                controls=[
                    speaker_icon_circle,
                    synth_badge,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=12,
            ),
            padding=20,
            alignment=ft.Alignment.CENTER,
        )

        # Input Card
        self.text_input = ft.TextField(
            value="مرحباً",
            hint_text="Type text to speak...",
            hint_style=ft.TextStyle(color=TEXT_MUTED, size=15),
            text_style=ft.TextStyle(color=TEXT_PRIMARY, size=18, weight=ft.FontWeight.BOLD),
            border=ft.InputBorder.NONE,
            multiline=True,
            min_lines=3,
            max_lines=6,
            content_padding=12,
            text_align=ft.TextAlign.CENTER,
        )

        input_box_card = create_glass_card(
            content=ft.Column(
                controls=[
                    ft.Text("ENTER TRANSLATION TEXT", size=10, weight=ft.FontWeight.BOLD, color=TEXT_SECONDARY),
                    self.text_input,
                ],
                spacing=8,
            ),
            padding=16,
            border_radius=18,
            width=VIEWPORT_WIDTH,
        )

        # Action Button: > Generate & Speak
        def on_generate_and_speak(e):
            val = self.text_input.value.strip()
            if not val:
                return
            def _tts():
                try:
                    import pyttsx3
                    eng = pyttsx3.init()
                    eng.say(val)
                    eng.runAndWait()
                except Exception:
                    pass
            threading.Thread(target=_tts, daemon=True).start()

        generate_btn = ft.Container(
            content=ft.ElevatedButton(
                content=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, size=20, color=BG_DARK),
                        ft.Text("Generate & Speak", size=14, weight=ft.FontWeight.BOLD, color=BG_DARK),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=6,
                ),
                style=ft.ButtonStyle(
                    bgcolor=ACCENT_EMERALD,
                    shape=ft.RoundedRectangleBorder(radius=16),
                    padding=ft.Padding.symmetric(vertical=14),
                ),
                on_click=on_generate_and_speak,
                width=VIEWPORT_WIDTH,
            ),
            margin=ft.Margin.only(top=16),
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    synth_header_card,
                    input_box_card,
                    generate_btn,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                spacing=12,
            )
        ]
