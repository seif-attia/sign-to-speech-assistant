import asyncio
import flet as ft

from components.mic_button import MicButton
from frontend.components.bottom_nav_bar import create_nav_bar


class SpeechToSpeechView(ft.View):
    def __init__(self, page: ft.Page):
        self.app_page = page

        async def close(e: ft.ControlEvent):
            await self.cleanup_async()
            await self.app_page.push_route("/")

        self.mic_button = MicButton()

        super().__init__(
            route="/speech-speech",
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.CENTER,
            appbar=ft.AppBar(
                leading=ft.IconButton(ft.Icons.ARROW_BACK, on_click=close),
                title=ft.Text("Speech to Speech", color=ft.Colors.BLACK),
                bgcolor=ft.Colors.SURFACE,
            ),
            navigation_bar=create_nav_bar(2, self.app_page),
        )

        self.controls = [
            ft.Container(
                content=self.mic_button,
                alignment=ft.Alignment.CENTER,
                expand=True,
            )
        ]

    async def cleanup_async(self):
        """Clean up background audio recording streams on view destruction."""
        if hasattr(self, "mic_button") and self.mic_button:
            await self.mic_button.cleanup_async()
