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
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar
from services.fastapi_endpoint import translate_text, play_speech

class TextToTextView(ft.View):
    """
    Screen 4: Text-to-Text Translation View.
    Allows typing text in English or Arabic, swapping source/target languages,
    translating with Qwen, and reading the output aloud with Sherpa-ONNX.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page
        self.is_arabic_to_english = True  # True: Ar -> En, False: En -> Ar

        async def go_back(e):
            await self.app_page.push_route("/")

        super().__init__(
            route="/text-text",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=16, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(0, self.app_page),
        )

        # Header
        self.lang_mode_text = ft.Text("Ar ⇄ En", size=11, weight=ft.FontWeight.BOLD, color=ACCENT_MINT)
        lang_badge = ft.Container(
            content=self.lang_mode_text,
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            border_radius=12,
            bgcolor="#132B26",
            border=ft.Border.all(1, "#1D473F"),
        )
        header = create_header(
            title="Translate Text",
            on_back=go_back,
            right_control=lang_badge,
        )

        self.input_card_title = ft.Text("ARABIC (INPUT)", size=11, weight=ft.FontWeight.BOLD, color=TEXT_SECONDARY)
        self.output_card_title = ft.Text("ENGLISH (TRANSLATION)", size=11, weight=ft.FontWeight.BOLD, color=ACCENT_MINT)

        # Input Card
        self.input_text_field = ft.TextField(
            hint_text="اكتب النص هنا للترجمة...",
            hint_style=ft.TextStyle(color=TEXT_MUTED, size=14),
            text_style=ft.TextStyle(color=TEXT_PRIMARY, size=15),
            border=ft.InputBorder.NONE,
            multiline=True,
            min_lines=3,
            max_lines=5,
            content_padding=0,
            text_align=ft.TextAlign.RIGHT,
        )

        def clear_input(e):
            self.input_text_field.value = ""
            self.output_text.value = "Translation will appear here..."
            self.app_page.update()

        input_card = create_glass_card(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.input_card_title,
                            ft.IconButton(
                                icon=ft.Icons.CLEAR_ROUNDED,
                                icon_size=16,
                                icon_color=TEXT_MUTED,
                                on_click=clear_input,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.input_text_field,
                ],
                spacing=8,
            ),
            padding=16,
            border_radius=18,
        )

        # Swap button in between cards
        def on_swap(e):
            self.is_arabic_to_english = not self.is_arabic_to_english
            if self.is_arabic_to_english:
                self.input_card_title.value = "ARABIC (INPUT)"
                self.output_card_title.value = "ENGLISH (TRANSLATION)"
                self.input_text_field.text_align = ft.TextAlign.RIGHT
                self.input_text_field.hint_text = "اكتب النص هنا للترجمة..."
                self.lang_mode_text.value = "Ar ➔ En"
            else:
                self.input_card_title.value = "ENGLISH (INPUT)"
                self.output_card_title.value = "ARABIC (TRANSLATION)"
                self.input_text_field.text_align = ft.TextAlign.LEFT
                self.input_text_field.hint_text = "Type text here to translate..."
                self.lang_mode_text.value = "En ➔ Ar"
            self.app_page.update()

        swap_btn = ft.Container(
            content=ft.IconButton(
                icon=ft.Icons.SWAP_VERT_ROUNDED,
                icon_color=ACCENT_MINT,
                icon_size=24,
                tooltip="Swap Translation Languages",
                on_click=on_swap,
            ),
            bgcolor="#152B2B",
            border=ft.Border.all(1, CARD_BORDER),
            border_radius=24,
            padding=2,
            margin=ft.Margin.symmetric(vertical=4),
        )

        # Output Card
        self.output_text = ft.Text(
            "Translation will appear here...",
            size=15,
            color=TEXT_PRIMARY,
            weight=ft.FontWeight.W_500,
        )

        async def speak_translation(e):
            text = self.output_text.value.strip()
            if not text or text.startswith("Translation will"):
                return
            target_lang = "en" if self.is_arabic_to_english else "ar"
            self.app_page.run_task(play_speech, self.app_page, text, target_lang)

        output_card = create_glass_card(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.output_card_title,
                            ft.IconButton(
                                icon=ft.Icons.VOLUME_UP_ROUNDED,
                                icon_size=20,
                                icon_color=ACCENT_MINT,
                                tooltip="Listen with Sherpa-ONNX Voice",
                                on_click=speak_translation,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.output_text,
                ],
                spacing=8,
            ),
            padding=16,
            border_radius=18,
            bg_color="#102525",
        )

        # Translate Action Button
        async def on_translate_click(e):
            in_text = self.input_text_field.value.strip()
            if not in_text:
                return
            src = "ar" if self.is_arabic_to_english else "en"
            tgt = "en" if self.is_arabic_to_english else "ar"
            self.output_text.value = "Translating with Qwen..."
            self.app_page.update()

            translated = await translate_text(in_text, source_lang=src, target_lang=tgt)
            self.output_text.value = translated if translated else "(No translation generated)"
            self.app_page.update()

        translate_btn = ft.Container(
            content=ft.ElevatedButton(
                content=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.TRANSLATE_ROUNDED, size=18, color=BG_DARK),
                        ft.Text("Translate", size=14, weight=ft.FontWeight.BOLD, color=BG_DARK),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=8,
                ),
                style=ft.ButtonStyle(
                    bgcolor=ACCENT_EMERALD,
                    shape=ft.RoundedRectangleBorder(radius=16),
                    padding=ft.Padding.symmetric(vertical=12),
                ),
                on_click=on_translate_click,
                width=VIEWPORT_WIDTH,
            ),
            margin=ft.Margin.only(top=6),
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    ft.Container(height=8),
                    input_card,
                    ft.Row(controls=[swap_btn], alignment=ft.MainAxisAlignment.CENTER),
                    output_card,
                    translate_btn,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                spacing=4,
            )
        ]
