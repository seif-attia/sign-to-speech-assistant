import flet as ft
import flet_camera as fc
import flet_permission_handler as fh
from components.camera_view import CameraView

def main(page: ft.Page) -> None:

    page.title = "Flet Camera" 
    page.theme_mode = ft.ThemeMode.LIGHT
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    camera_feed: CameraView = CameraView(width=320, height=420, lens_direction=fc.CameraLensDirection.BACK)

    page.add(ft.Column(
        alignment=ft.MainAxisAlignment.START,
        controls= camera_feed
    ))

   
if __name__ == "__main__":
    ft.run(main)
