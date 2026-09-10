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
import threading

class TextToTextView(ft.View):
    """
    Screen 4: Text-to-Text Translation View.
    Allows typing Arabic text and getting English translation with audio readout.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

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
        header = create_header(
            title="Translate Text",
            on_back=go_back,
            right_control=create_lang_toggle("Ar ⇄ En"),
        )

        # Input Card (Arabic)
        self.input_text_field = ft.TextField(
            hint_text="صباح الخير، كيف يمكنني مساعدتك اليوم؟",
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
                            ft.Text("ARABIC (INPUT)", size=11, weight=ft.FontWeight.BOLD, color=TEXT_SECONDARY),
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
            # In a full deployment, this swaps source and target languages
            pass

        swap_btn = ft.Container(
            content=ft.IconButton(
                icon=ft.Icons.SWAP_VERT_ROUNDED,
                icon_color=ACCENT_MINT,
                icon_size=24,
                on_click=on_swap,
            ),
            bgcolor="#152B2B",
            border=ft.Border.all(1, CARD_BORDER),
            border_radius=24,
            padding=2,
            margin=ft.Margin.symmetric(vertical=4),
        )

        # Output Card (English)
        self.output_text = ft.Text(
            "Good morning, how can I help you today?",
            size=15,
            color=TEXT_PRIMARY,
            weight=ft.FontWeight.W_500,
        )

        def speak_translation(e):
            text = self.output_text.value
            if not text:
                return
            def _tts():
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
            threading.Thread(target=_tts, daemon=True).start()

        output_card = create_glass_card(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text("ENGLISH (TRANSLATION)", size=11, weight=ft.FontWeight.BOLD, color=ACCENT_MINT),
                            ft.IconButton(
                                icon=ft.Icons.VOLUME_UP_ROUNDED,
                                icon_size=20,
                                icon_color=ACCENT_MINT,
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

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    ft.Container(height=12),
                    input_card,
                    ft.Row(controls=[swap_btn], alignment=ft.MainAxisAlignment.CENTER),
                    output_card,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                spacing=4,
            )
        ]
