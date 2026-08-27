import asyncio
import flet as ft

class HomeView(ft.View):
    def __init__(self, page: ft.Page):
        self.app_page = page
        super().__init__(  
          route = '/',
          horizontal_alignment = ft.CrossAxisAlignment.CENTER,
          vertical_alignment = ft.MainAxisAlignment.CENTER,
          appbar=ft.AppBar(title=ft.Text("Sign to Speech Assistant", color= ft.Colors.WHITE),bgcolor=ft.Colors.BLUE )
        )

        self.controls = [
            ft.Column(controls=
            [
                ft.Button("Sign to Speech",
                icon=ft.Icons.CAMERA_ALT,
                on_click=lambda e: asyncio.create_task(self.app_page.push_route("/sign-speech"))
                ),
                ft.Button("Speech To Text",
                icon=ft.Icons.MESSAGE,
                on_click=lambda e: asyncio.create_task(self.app_page.push_route("/speech-text"))
                ),

            ])
        ]

    
