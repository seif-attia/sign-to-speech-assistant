import asyncio
import flet as ft

from frontend.components.bottom_nav_bar import create_nav_bar

class SpeechToTextView(ft.View):
    def __init__(self, page: ft.Page):
        self.app_page = page

        async def close(e: ft.ControlEvent):
            await self.app_page.push_route("/")

        super().__init__(
            route= '/speech-text',
            horizontal_alignment= ft.CrossAxisAlignment.CENTER,
            vertical_alignment= ft.MainAxisAlignment.START,
            appbar = ft.AppBar(
                            leading=ft.IconButton(ft.Icons.ARROW_BACK, on_click= close),
                            title=ft.Text("Speech to Text", color= ft.Colors.BLACK),
                            bgcolor=ft.Colors.SURFACE
                        ),
            navigation_bar= create_nav_bar(2, self.app_page)
        )

        self.controls = [
            ft.Column(
                controls=[ft.Text("Speech to Text")]
            )
        ]