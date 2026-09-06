import flet as ft

def create_nav_bar(selected_index, page: ft.Page):

    routes = ["/", "/sign-speech", "/speech-text"]

    async def on_nav_change(e: ft.ControlEvent):
            target = routes[e.control.selected_index]
            if page.route != target:
                await page.push_route(target)

    return ft.NavigationBar(
        selected_index= selected_index,
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.HOME, label="Home"),
            ft.NavigationBarDestination(icon=ft.Icons.BOOK, label="Learn"),
            ft.NavigationBarDestination(icon=ft.Icons.SETTINGS, label="Settings")
        ],
        on_change= on_nav_change
    )
