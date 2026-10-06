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
    create_pill_badge,
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar

class ContributionView(ft.View):
    """
    Screen 9: Contribution Mode View (/contribute).
    Enables volunteers to record and submit gestures to train AI models.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        super().__init__(
            route="/contribute",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=50, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(2, self.app_page),
        )

        header = ft.Row(
            controls=[
                ft.Text("Contribute", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                create_pill_badge("CAM ACTIVE", font_size=11),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            width=VIEWPORT_WIDTH,
        )

        subtitle = ft.Text(
            "Help train the gesture models. Fill in the sign label, aim your camera, and record a short video clip.",
            size=12,
            color=TEXT_SECONDARY,
            width=VIEWPORT_WIDTH,
        )

        # Camera Viewfinder Box with crosshairs
        viewfinder = ft.Container(
            content=ft.Stack(
                controls=[
                    ft.Container(
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(ft.Icons.VIDEOCAM, size=48, color="#223E3E"),
                    ),
                    # Corner markings simulation
                    ft.Container(
                        content=ft.Icon(ft.Icons.CROP_FREE, size=40, color=ACCENT_MINT),
                        alignment=ft.Alignment.CENTER,
                    ),
                ]
            ),
            width=VIEWPORT_WIDTH,
            height=200,
            bgcolor="#0E1B1B",
            border_radius=20,
            border=ft.Border.all(1, CARD_BORDER),
        )

        # Form Inputs: Sign Name and Category
        sign_name_input = ft.TextField(
            label="SIGN NAME",
            label_style=ft.TextStyle(size=11, color=TEXT_MUTED, weight=ft.FontWeight.BOLD),
            value="Hello",
            text_style=ft.TextStyle(color=TEXT_PRIMARY, size=14),
            border=ft.InputBorder.OUTLINE,
            border_color=CARD_BORDER,
            focused_border_color=ACCENT_EMERALD,
            bgcolor=INPUT_BG,
            border_radius=12,
            content_padding=ft.Padding.symmetric(horizontal=14, vertical=10),
        )

        category_input = ft.TextField(
            label="CATEGORY",
            label_style=ft.TextStyle(size=11, color=TEXT_MUTED, weight=ft.FontWeight.BOLD),
            value="Conversational",
            text_style=ft.TextStyle(color=TEXT_PRIMARY, size=14),
            border=ft.InputBorder.OUTLINE,
            border_color=CARD_BORDER,
            focused_border_color=ACCENT_EMERALD,
            bgcolor=INPUT_BG,
            border_radius=12,
            content_padding=ft.Padding.symmetric(horizontal=14, vertical=10),
        )

        # Record Action Button
        def on_record_click(e):
            pass

        record_btn = ft.Container(
            content=ft.ElevatedButton(
                content=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.CIRCLE, size=14, color=ft.Colors.RED_500),
                        ft.Text("Record a Sign", size=14, weight=ft.FontWeight.BOLD, color=BG_DARK),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=8,
                ),
                style=ft.ButtonStyle(
                    bgcolor=ACCENT_EMERALD,
                    shape=ft.RoundedRectangleBorder(radius=16),
                    padding=ft.Padding.symmetric(vertical=14),
                ),
                on_click=on_record_click,
                width=VIEWPORT_WIDTH,
            ),
            margin=ft.Margin.only(top=10),
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    subtitle,
                    viewfinder,
                    sign_name_input,
                    category_input,
                    record_btn,
                ],
                spacing=14,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                scroll=ft.ScrollMode.ALWAYS,
            )
        ]
