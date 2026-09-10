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


class HomeView(ft.View):
    """
    Communication Mode View (Screen 2).
    Displays 'Choose Mode', a 2x2 grid of translation modes,
    and a prominent 'Sign-to-Speech' Live Camera banner card.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        super().__init__(
            route="/",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=24, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(0, self.app_page),
        )

        # Header Title
        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text(
                        "Choose Mode",
                        size=24,
                        weight=ft.FontWeight.BOLD,
                        color=TEXT_PRIMARY,
                    ),
                    ft.Container(
                        content=ft.Icon(ft.Icons.AUTO_AWESOME, color=ACCENT_MINT, size=20),
                        padding=ft.Padding.all(6),
                        border_radius=10,
                        bgcolor="#162828",
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            width=VIEWPORT_WIDTH,
            margin=ft.Margin.only(bottom=16),
        )

        # 2x2 Mode Grid items
        def build_mode_tile(title: str, icon_name: str, route_target: str):
            async def on_click(e):
                await self.app_page.push_route(route_target)

            return ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Container(
                            content=ft.Icon(icon_name, size=26, color=ACCENT_MINT),
                            padding=ft.Padding.all(10),
                            border_radius=12,
                            bgcolor="#132B27",
                        ),
                        ft.Text(
                            title,
                            size=13,
                            weight=ft.FontWeight.W_600,
                            color=TEXT_PRIMARY,
                        ),
                    ],
                    spacing=12,
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                ),
                width=(VIEWPORT_WIDTH - 12) // 2,
                height=110,
                padding=14,
                border_radius=16,
                bgcolor=CARD_BG,
                border=ft.Border.all(1, CARD_BORDER),
                on_click=on_click,
                ink=True,
            )

        grid_row_1 = ft.Row(
            controls=[
                build_mode_tile("Text-to-Text", ft.Icons.KEYBOARD_ALT_OUTLINED, "/text-text"),
                build_mode_tile("Speech-to-Text", ft.Icons.GRAPHIC_EQ_ROUNDED, "/speech-text"),
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        grid_row_2 = ft.Row(
            controls=[
                build_mode_tile("Speech-to-Speech", ft.Icons.FORUM_OUTLINED, "/speech-speech"),
                build_mode_tile("Text-to-Speech", ft.Icons.VOLUME_UP_ROUNDED, "/text-speech"),
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        # Sign-to-Speech Big Banner
        async def on_launch_camera(e):
            await self.app_page.push_route("/sign-speech")

        sign_banner = ft.Container(
            content=ft.Column(
                controls=[
                    create_pill_badge("LIVE CAMERA DETECT", font_size=10),
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.FRONT_HAND, size=24, color=ACCENT_MINT),
                            ft.Text(
                                "Sign-to-Speech",
                                size=17,
                                weight=ft.FontWeight.BOLD,
                                color=TEXT_PRIMARY,
                            ),
                        ],
                        spacing=8,
                    ),
                    ft.Text(
                        "Translate sign language in real-time using on-device computer vision & AI.",
                        size=12,
                        color=TEXT_SECONDARY,
                    ),
                    ft.Container(
                        content=ft.ElevatedButton(
                            content=ft.Row(
                                controls=[
                                    ft.Text("Launch Sign Camera", size=13, weight=ft.FontWeight.BOLD, color=BG_DARK),
                                    ft.Icon(ft.Icons.NORTH_EAST_ROUNDED, size=16, color=BG_DARK),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=6,
                            ),
                            style=ft.ButtonStyle(
                                bgcolor=ACCENT_EMERALD,
                                shape=ft.RoundedRectangleBorder(radius=12),
                                padding=ft.Padding.symmetric(vertical=12, horizontal=16),
                            ),
                            on_click=on_launch_camera,
                            width=VIEWPORT_WIDTH - 48,
                        ),
                        margin=ft.Margin.only(top=6),
                    ),
                ],
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.START,
            ),
            padding=18,
            border_radius=20,
            bgcolor="#132626",
            border=ft.Border.all(1, "#264848"),
            width=VIEWPORT_WIDTH,
            margin=ft.Margin.only(top=16),
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    grid_row_1,
                    grid_row_2,
                    sign_banner,
                ],
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                scroll=ft.ScrollMode.ALWAYS,
            )
        ]
