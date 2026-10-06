import flet as ft
from frontend.theme import (
    BG_DARK,
    CARD_BG,
    ACCENT_EMERALD,
    ACCENT_MINT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    VIEWPORT_WIDTH,
)

class IntroView(ft.View):
    """
    Screen 1: Intro / Main Brand Splash Screen.
    Displays WESAL with stylized branding and entrance button.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        async def get_started(e):
            await self.app_page.push_route("/")

        super().__init__(
            route="/intro",
            bgcolor=BG_DARK,
            padding=24,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.CENTER,
        )

        # Stylized brand logo emblem
        logo_icon = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Icon(ft.Icons.ALL_INCLUSIVE_ROUNDED, size=90, color=TEXT_PRIMARY),
                    ft.Text(
                        "WESAL",
                        size=34,
                        weight=ft.FontWeight.BOLD,
                        color=TEXT_PRIMARY,
                        letter_spacing=6,
                    ),
                    ft.Text(
                        "Sign Language & Speech Assistant",
                        size=13,
                        color=ACCENT_MINT,
                        weight=ft.FontWeight.W_500,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=12,
            ),
            alignment=ft.Alignment.CENTER,
            expand=True,
        )

        enter_btn = ft.Container(
            content=ft.ElevatedButton(
                content=ft.Row(
                    controls=[
                        ft.Text("Get Started", size=15, weight=ft.FontWeight.BOLD, color=BG_DARK),
                        ft.Icon(ft.Icons.ARROW_FORWARD, size=18, color=BG_DARK),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=8,
                ),
                style=ft.ButtonStyle(
                    bgcolor=ACCENT_EMERALD,
                    shape=ft.RoundedRectangleBorder(radius=16),
                    padding=ft.Padding.symmetric(vertical=14),
                ),
                on_click=get_started,
                width=VIEWPORT_WIDTH,
            ),
            margin=ft.Margin.only(bottom=24),
        )

        self.controls = [
            ft.Column(
                controls=[
                    logo_icon,
                    enter_btn,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                expand=True,
            )
        ]
