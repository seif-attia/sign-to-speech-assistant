import asyncio
import flet as ft

class HomeView(ft.View):
    def __init__(self, page: ft.Page):
        self.app_page = page
        super().__init__(  
          route = '/',
          horizontal_alignment = ft.CrossAxisAlignment.CENTER,
         vertical_alignment = ft.MainAxisAlignment.CENTER,
        )

        self.controls = [
                    ft.Icon(ft.Icons.SIGN_LANGUAGE, size=100, color = ft.Colors.BLUE),
                    ft.Text("Sign Language Assistant", size=30, weight=ft.FontWeight.BOLD),
                    ft.Button("Sign to Speech",
                    icon=ft.Icons.CAMERA_ALT,
                    on_click=lambda: asyncio.create_task(self.app_page.push_route("/sign-speech"))
                    )
                ]

    
