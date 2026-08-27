import asyncio
import flet as ft

class SpeechToTextView(ft.View):
    def __init__(self, page: ft.Page):
        self.app_page = page

        super().__init__(
            route= '/speech-text',
            horizontal_alignment= ft.CrossAxisAlignment.CENTER,
            vertical_alignment= ft.MainAxisAlignment.START,
            appbar = ft.AppBar(
                            leading=ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda e: asyncio.create_task(self.app_page.push_route("/"))),
                            title=ft.Text("Speech to Text", color= ft.Colors.WHITE),
                            bgcolor=ft.Colors.BLUE
                        ),
        )

        self.controls = [
            ft.Column(
                controls=[ft.Text("Speech to Text")]
            )
        ]