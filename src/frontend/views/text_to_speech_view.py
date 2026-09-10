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
    BADGE_BG,
    create_glass_card,
    create_header,
    create_lang_toggle,
    create_pill_badge,
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar
from services.fastapi_endpoint import play_speech

class TextToSpeechView(ft.View):
    """
    Screen 7: Text-to-Speech View.
    Allows typing phrase, selecting English/Arabic voice, synthesizer engagement graphic,
    and 'Generate & Speak' button wired directly to Sherpa-ONNX backend.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page
        self.selected_lang = "ar"  # Default Arabic or English

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

        def _toggle_lang(e):
            self.selected_lang = "en" if self.selected_lang == "ar" else "ar"
            self.lang_btn_text.value = "Voice: Arabic" if self.selected_lang == "ar" else "Voice: English"
            if self.selected_lang == "ar":
                self.text_input.value = "مرحباً بكم في تطبيق لغة الإشارة"
                self.text_input.text_align = ft.TextAlign.RIGHT
            else:
                self.text_input.value = "Welcome to the Sign Language Assistant"
                self.text_input.text_align = ft.TextAlign.LEFT
            self.app_page.update()

        self.lang_btn_text = ft.Text("Voice: Arabic", size=11, weight=ft.FontWeight.BOLD, color=ACCENT_MINT)
        lang_toggle_btn = ft.Container(
            content=self.lang_btn_text,
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            border_radius=12,
            bgcolor="#132B26",
            border=ft.Border.all(1, "#1D473F"),
            on_click=_toggle_lang,
        )

        header = create_header(
            title="Text to Speech",
            on_back=go_back,
            right_control=lang_toggle_btn,
        )

        # Synthesizer Engaged Visual
        self.synth_badge_text = ft.Text("SHERPA-ONNX READY", size=10, weight=ft.FontWeight.BOLD, color=ACCENT_MINT)
        self.synth_badge = ft.Container(
            content=ft.Row(controls=[self.synth_badge_text], spacing=4, alignment=ft.MainAxisAlignment.CENTER),
            bgcolor=BADGE_BG,
            border_radius=12,
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        )
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
                    self.synth_badge,
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
            value="مرحباً بكم في تطبيق لغة الإشارة",
            hint_text="Type text to speak...",
            hint_style=ft.TextStyle(color=TEXT_MUTED, size=15),
            text_style=ft.TextStyle(color=TEXT_PRIMARY, size=16, weight=ft.FontWeight.BOLD),
            border=ft.InputBorder.NONE,
            multiline=True,
            min_lines=3,
            max_lines=6,
            content_padding=12,
            text_align=ft.TextAlign.RIGHT,
        )

        input_box_card = create_glass_card(
            content=ft.Column(
                controls=[
                    ft.Text("ENTER TEXT TO SYNTHESIZE", size=10, weight=ft.FontWeight.BOLD, color=TEXT_SECONDARY),
                    self.text_input,
                ],
                spacing=8,
            ),
            padding=16,
            border_radius=18,
            width=VIEWPORT_WIDTH,
        )

        # Action Button: > Generate & Speak
        async def _do_speak(val: str):
            self.synth_badge_text.value = "SYNTHESIZING..."
            self.app_page.update()
            try:
                await play_speech(self.app_page, val, lang=self.selected_lang)
            except Exception:
                pass
            self.synth_badge_text.value = "SHERPA-ONNX READY"
            self.app_page.update()

        def on_generate_and_speak(e):
            val = self.text_input.value.strip()
            if not val:
                return
            self.app_page.run_task(_do_speak, val)

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
