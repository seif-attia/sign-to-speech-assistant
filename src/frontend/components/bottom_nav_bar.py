import flet as ft
from frontend.theme import BG_DARK, CARD_BG, CARD_BORDER, ACCENT_EMERALD, TEXT_PRIMARY, TEXT_MUTED

def create_nav_bar(selected_index: int, page: ft.Page):
    """
    Bottom navigation dock matching the 4 primary modes:
    0: Communication Mode (/)
    1: Educational Mode (/learn)
    2: Contribution Mode (/contribute)
    3: Settings Mode (/settings)
    """
    routes = ["/", "/learn", "/contribute", "/settings"]

    async def on_nav_change(e: ft.ControlEvent):
        target = routes[e.control.selected_index]
        if page.route != target:
            await page.push_route(target)

    return ft.Container(
        content=ft.NavigationBar(
            selected_index=selected_index,
            bgcolor="transparent",
            indicator_color=ACCENT_EMERALD,
            destinations=[
                ft.NavigationBarDestination(
                    icon=ft.Icons.HOME_OUTLINED,
                    selected_icon=ft.Icons.HOME,
                    label="Home",
                ),
                ft.NavigationBarDestination(
                    icon=ft.Icons.BOOK_OUTLINED,
                    selected_icon=ft.Icons.BOOK,
                    label="Learn",
                ),
                ft.NavigationBarDestination(
                    icon=ft.Icons.VIDEOCAM_OUTLINED,
                    selected_icon=ft.Icons.VIDEOCAM,
                    label="Contribute",
                ),
                ft.NavigationBarDestination(
                    icon=ft.Icons.PERSON_OUTLINED,
                    selected_icon=ft.Icons.PERSON,
                    label="Settings",
                ),
            ],
            on_change=on_nav_change,
        ),
        bgcolor=CARD_BG,
        border=ft.Border(top=ft.BorderSide(1, CARD_BORDER)),
        padding=ft.Padding.only(bottom=8),
    )
