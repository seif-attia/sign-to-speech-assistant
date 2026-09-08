import flet as ft
from frontend.components.bottom_nav_bar import create_nav_bar

class HomeView(ft.View):
    def __init__(self, page: ft.Page):
        self.app_page = page

        async def go_sign_speech(e: ft.ControlEvent):
            await self.app_page.push_route("/sign-speech")

        async def go_speech_text(e: ft.ControlEvent):
            await self.app_page.push_route("/speech-text")

        async def go_speech_speech(e: ft.ControlEvent):
            await self.app_page.push_route("/speech-speech")

        super().__init__(  
          route = '/',
          horizontal_alignment = ft.CrossAxisAlignment.CENTER,
          vertical_alignment = ft.MainAxisAlignment.CENTER,
          appbar=ft.AppBar(title=ft.Text("Sign to Speech Assistant", color= ft.Colors.BLACK),bgcolor=ft.Colors.SURFACE ),
          navigation_bar= create_nav_bar(0, self.app_page)
        )

        self.controls = [
            ft.Column(controls=
            [
                ft.Button("Sign to Speech",
                icon=ft.Icons.CAMERA_ALT,
                on_click=go_sign_speech
                ),
                ft.Button("Speech To Text",
                icon=ft.Icons.MESSAGE,
                on_click=go_speech_text
                ),
                ft.Button("Speech to Speech",
                icon=ft.Icons.MIC,
                on_click=go_speech_speech
                ),
            ])
        ]

    
