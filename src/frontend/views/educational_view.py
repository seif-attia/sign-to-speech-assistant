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
    create_pill_badge,
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar

class EducationalView(ft.View):
    """
    Screen 8: Educational Mode View (/learn).
    Features XP badge, Search bar, Category filter pills, and a 2x2 grid of Frequent Phrases.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        super().__init__(
            route="/learn",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=50, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(1, self.app_page),
        )

        # Header with XP counter
        header = ft.Row(
            controls=[
                ft.Text("Learn", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                create_pill_badge("320 XP", bg_color="#12382F", text_color=ACCENT_MINT, font_size=12),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            width=VIEWPORT_WIDTH,
        )

        # Search Bar
        search_field = ft.TextField(
            hint_text="Search words or signs...",
            hint_style=ft.TextStyle(color=TEXT_MUTED, size=13),
            text_style=ft.TextStyle(color=TEXT_PRIMARY, size=14),
            prefix_icon=ft.Icons.SEARCH,
            border=ft.InputBorder.NONE,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        )
        search_card = ft.Container(
            content=search_field,
            bgcolor=CARD_BG,
            border=ft.Border.all(1, CARD_BORDER),
            border_radius=14,
            width=VIEWPORT_WIDTH,
            padding=ft.Padding.symmetric(horizontal=4),
        )

        # Categories horizontal list
        categories = ["Greetings", "Numbers", "Colors", "Food"]
        selected_cat = "Greetings"

        def build_category_chip(name: str, is_active: bool):
            return ft.Container(
                content=ft.Text(
                    name,
                    size=12,
                    weight=ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL,
                    color=BG_DARK if is_active else TEXT_SECONDARY,
                ),
                bgcolor=ACCENT_EMERALD if is_active else CARD_BG,
                border=ft.Border.all(1, ACCENT_EMERALD if is_active else CARD_BORDER),
                border_radius=20,
                padding=ft.Padding.symmetric(horizontal=16, vertical=8),
            )

        cat_row = ft.Row(
            controls=[
                build_category_chip(cat, cat == selected_cat)
                for cat in categories
            ],
            scroll=ft.ScrollMode.HIDDEN,
            spacing=8,
            width=VIEWPORT_WIDTH,
        )

        # Frequent Phrases Grid
        phrases_header = ft.Container(
            content=ft.Text(
                "FREQUENT PHRASES",
                size=11,
                weight=ft.FontWeight.BOLD,
                color=TEXT_MUTED,
                style=ft.TextStyle(letter_spacing=1.2),
            ),
            width=VIEWPORT_WIDTH,
            margin=ft.Margin.only(top=8),
        )

        phrases = [
            {"ar": "مرحباً", "en": "Hello"},
            {"ar": "شكراً لك", "en": "Thank you"},
            {"ar": "كيف حالك؟", "en": "How are you?"},
            {"ar": "مع السلامة", "en": "Goodbye"},
        ]

        from services.fastapi_endpoint import play_speech

        def build_phrase_card(ar: str, en: str):
            def _on_card_click(e):
                self.app_page.run_task(play_speech, self.app_page, ar, "ar")

            return ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(ar, size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                        ft.Text(en, size=12, color=TEXT_SECONDARY),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=6,
                ),
                width=(VIEWPORT_WIDTH - 12) // 2,
                height=90,
                bgcolor=CARD_BG,
                border=ft.Border.all(1, CARD_BORDER),
                border_radius=16,
                alignment=ft.Alignment.CENTER,
                ink=True,
                tooltip="Tap to speak in Arabic",
                on_click=_on_card_click,
            )

        grid_row_1 = ft.Row(
            controls=[
                build_phrase_card(phrases[0]["ar"], phrases[0]["en"]),
                build_phrase_card(phrases[1]["ar"], phrases[1]["en"]),
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        grid_row_2 = ft.Row(
            controls=[
                build_phrase_card(phrases[2]["ar"], phrases[2]["en"]),
                build_phrase_card(phrases[3]["ar"], phrases[3]["en"]),
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    search_card,
                    ft.Container(
                        content=ft.Text("CATEGORIES", size=11, weight=ft.FontWeight.BOLD, color=TEXT_MUTED, style=ft.TextStyle(letter_spacing=1.2)),
                        width=VIEWPORT_WIDTH,
                    ),
                    cat_row,
                    phrases_header,
                    grid_row_1,
                    grid_row_2,
                ],
                spacing=14,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                scroll=ft.ScrollMode.ALWAYS,
            )
        ]
