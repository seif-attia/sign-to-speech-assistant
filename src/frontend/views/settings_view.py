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
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar

class SettingsView(ft.View):
    """
    Screen 10: Settings Mode View (/settings).
    Displays Profile card (Ahmed A.) and settings menu entries.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        super().__init__(
            route="/settings",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=24, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(3, self.app_page),
        )

        header = ft.Container(
            content=ft.Text("Settings", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            width=VIEWPORT_WIDTH,
            margin=ft.Margin.only(bottom=8),
        )

        # Profile Card
        avatar = ft.Container(
            content=ft.Text("AA", size=18, weight=ft.FontWeight.BOLD, color=BG_DARK),
            width=50,
            height=50,
            border_radius=25,
            bgcolor=ACCENT_MINT,
            alignment=ft.Alignment.CENTER,
        )

        profile_info = ft.Column(
            controls=[
                ft.Text("Ahmed A.", size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                ft.Text("ahmed24@gmail.com", size=12, color=TEXT_SECONDARY),
            ],
            spacing=2,
            expand=True,
        )

        profile_card = create_glass_card(
            content=ft.Row(
                controls=[
                    avatar,
                    profile_info,
                    ft.Icon(ft.Icons.EDIT_OUTLINED, size=18, color=TEXT_MUTED),
                ],
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=16,
            border_radius=18,
            width=VIEWPORT_WIDTH,
        )

        # Settings Group
        section_label = ft.Container(
            content=ft.Text("APP SETTINGS", size=11, weight=ft.FontWeight.BOLD, color=TEXT_MUTED, style=ft.TextStyle(letter_spacing=1.2)),
            width=VIEWPORT_WIDTH,
            margin=ft.Margin.only(top=16, bottom=4),
        )

        def build_settings_row(title: str, trailing: ft.Control):
            return ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Text(title, size=14, color=TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        trailing,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                padding=ft.Padding.symmetric(vertical=14, horizontal=16),
                bgcolor=CARD_BG,
                border_radius=14,
                border=ft.Border.all(1, CARD_BORDER),
                width=VIEWPORT_WIDTH,
            )

        def _open_lang_dialog(e):
            def set_lang(chosen):
                dialog.open = False
                self.app_page.update()

            dialog = ft.AlertDialog(
                title=ft.Text("Language Preferences"),
                content=ft.Column(
                    controls=[
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.LANGUAGE, color=ACCENT_MINT),
                            title=ft.Text("English (US)"),
                            subtitle=ft.Text("Piper VITS Neural TTS"),
                            on_click=lambda ev: set_lang("en"),
                        ),
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.LANGUAGE, color=ACCENT_MINT),
                            title=ft.Text("العربية (Arabic)"),
                            subtitle=ft.Text("Nabra Kokoro Neural TTS"),
                            on_click=lambda ev: set_lang("ar"),
                        ),
                    ],
                    tight=True,
                ),
            )
            self.app_page.overlay.append(dialog)
            dialog.open = True
            self.app_page.update()

        row_language = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text("Language Preferences (English / العربية)", size=14, color=TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                    ft.Icon(ft.Icons.CHEVRON_RIGHT, color=TEXT_MUTED, size=20),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.Padding.symmetric(vertical=14, horizontal=16),
            bgcolor=CARD_BG,
            border_radius=14,
            border=ft.Border.all(1, CARD_BORDER),
            width=VIEWPORT_WIDTH,
            on_click=_open_lang_dialog,
            ink=True,
        )
        row_help = build_settings_row(
            "Help & Feedback",
            ft.Icon(ft.Icons.CHEVRON_RIGHT, color=TEXT_MUTED, size=20),
        )
        row_version = build_settings_row(
            "App Version",
            ft.Text("v1.1.0", size=13, color=TEXT_SECONDARY, weight=ft.FontWeight.W_500),
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    profile_card,
                    section_label,
                    row_language,
                    row_help,
                    row_version,
                ],
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                scroll=ft.ScrollMode.ALWAYS,
            )
        ]
