import asyncio
import flet as ft
from services.config import (
    check_server_health,
    get_server_host,
    get_server_port,
    get_server_ws_url,
    set_server_host,
    set_server_port,
)
from services.network_stream_service import get_global_streamer
from frontend.theme import (
    BG_DARK,
    CARD_BG,
    CARD_BORDER,
    ACCENT_EMERALD,
    ACCENT_MINT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    create_glass_card,
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar

class SettingsView(ft.View):
    """
    Screen 10: Settings Mode View (/settings).
    Displays Profile card (Ahmed A.) and settings menu entries.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        super().__init__(
            route="/settings",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=50, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(3, self.app_page),
        )

        def _open_server_settings_dialog(e):
            """Allows dynamically updating server host IP from Settings view."""
            host_input = ft.TextField(
                label="FastAPI Server IP / Host",
                value=get_server_host(),
                hint_text="e.g. 192.168.1.43 or 127.0.0.1",
                width=280,
            )
            port_input = ft.TextField(
                label="Port",
                value=str(get_server_port()),
                hint_text="8000",
                width=280,
            )
            dialog_status = ft.Text("", size=12, italic=True)

            async def save_and_reconnect(ev):
                new_host = host_input.value.strip()
                try:
                    new_port = int(port_input.value.strip())
                    set_server_port(new_port)
                except ValueError:
                    pass

                if new_host:
                    set_server_host(new_host)

                dialog_status.value = "Testing connection..."
                dialog_status.color = ft.Colors.BLUE_700
                self.app_page.update()

                health = await check_server_health(timeout=2.0)
                if health:
                    dialog_status.value = f"Success! Backend online ({health.get('classes_count', '?')} classes)"
                    dialog_status.color = ft.Colors.GREEN_700
                else:
                    dialog_status.value = "Warning: Could not reach /health. Reconnecting WS anyway..."
                    dialog_status.color = ft.Colors.ORANGE_800

                # Reconnect the persistent global websocket client to the new host
                global_streamer = get_global_streamer()
                global_streamer.update_url(get_server_ws_url())
                self.app_page.update()
                await asyncio.sleep(1.0)
                settings_dialog.open = False
                self.app_page.update()

            def cancel_dialog(ev):
                settings_dialog.open = False
                self.app_page.update()

            settings_dialog = ft.AlertDialog(
                title=ft.Text("Backend Connection Settings"),
                content=ft.Column(
                    controls=[
                        ft.Text("Enter workstation IP where FastAPI server is running:"),
                        host_input,
                        port_input,
                        dialog_status,
                    ],
                    spacing=8,
                    tight=True,
                ),
                actions=[
                    ft.TextButton("Cancel", on_click=cancel_dialog),
                    ft.ElevatedButton("Save & Connect", on_click=save_and_reconnect),
                ],
            )
            self.app_page.overlay.append(settings_dialog)
            settings_dialog.open = True
            self.app_page.update()

        # Top Bar with Settings title on the left and Server IP button on the top right
        server_ip_btn = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.SETTINGS_ETHERNET, size=15, color=ACCENT_MINT),
                    ft.Text("Server IP", size=12, color=ACCENT_MINT, weight=ft.FontWeight.W_500),
                ],
                spacing=5,
            ),
            padding=ft.Padding.symmetric(horizontal=10, vertical=6),
            border_radius=10,
            bgcolor="#132323",
            on_click=_open_server_settings_dialog,
            tooltip="Configure Server IP / Host",
        )

        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text("Settings", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                    server_ip_btn,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            width=VIEWPORT_WIDTH,
            margin=ft.Margin.only(bottom=8),
        )

        # Profile Card
        avatar = ft.Container(
            content=ft.Text("AA", size=18, weight=ft.FontWeight.BOLD, color=BG_DARK),
            width=50,
            height=50,
            border_radius=25,
            bgcolor=ACCENT_MINT,
            alignment=ft.Alignment.CENTER,
        )

        profile_info = ft.Column(
            controls=[
                ft.Text("Ahmed A.", size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                ft.Text("ahmed24@gmail.com", size=12, color=TEXT_SECONDARY),
            ],
            spacing=2,
            expand=True,
        )

        profile_card = create_glass_card(
            content=ft.Row(
                controls=[
                    avatar,
                    profile_info,
                    ft.Icon(ft.Icons.EDIT_OUTLINED, size=18, color=TEXT_MUTED),
                ],
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=16,
            border_radius=18,
            width=VIEWPORT_WIDTH,
        )

        # Settings Group
        section_label = ft.Container(
            content=ft.Text("APP SETTINGS", size=11, weight=ft.FontWeight.BOLD, color=TEXT_MUTED, style=ft.TextStyle(letter_spacing=1.2)),
            width=VIEWPORT_WIDTH,
            margin=ft.Margin.only(top=16, bottom=4),
        )

        def build_settings_row(title: str, trailing: ft.Control):
            return ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Text(title, size=14, color=TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        trailing,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                padding=ft.Padding.symmetric(vertical=14, horizontal=16),
                bgcolor=CARD_BG,
                border_radius=14,
                border=ft.Border.all(1, CARD_BORDER),
                width=VIEWPORT_WIDTH,
            )

        def _open_lang_dialog(e):
            def set_lang(chosen):
                dialog.open = False
                self.app_page.update()

            dialog = ft.AlertDialog(
                title=ft.Text("Language Preferences"),
                content=ft.Column(
                    controls=[
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.LANGUAGE, color=ACCENT_MINT),
                            title=ft.Text("English (US)"),
                            subtitle=ft.Text("Piper VITS Neural TTS"),
                            on_click=lambda ev: set_lang("en"),
                        ),
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.LANGUAGE, color=ACCENT_MINT),
                            title=ft.Text("العربية (Arabic)"),
                            subtitle=ft.Text("Nabra Kokoro Neural TTS"),
                            on_click=lambda ev: set_lang("ar"),
                        ),
                    ],
                    tight=True,
                ),
            )
            self.app_page.overlay.append(dialog)
            dialog.open = True
            self.app_page.update()

        row_language = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text("Language Preferences (English / العربية)", size=14, color=TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                    ft.Icon(ft.Icons.CHEVRON_RIGHT, color=TEXT_MUTED, size=20),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.Padding.symmetric(vertical=14, horizontal=16),
            bgcolor=CARD_BG,
            border_radius=14,
            border=ft.Border.all(1, CARD_BORDER),
            width=VIEWPORT_WIDTH,
            on_click=_open_lang_dialog,
            ink=True,
        )
        row_help = build_settings_row(
            "Help & Feedback",
            ft.Icon(ft.Icons.CHEVRON_RIGHT, color=TEXT_MUTED, size=20),
        )
        row_version = build_settings_row(
            "App Version",
            ft.Text("v1.1.0", size=13, color=TEXT_SECONDARY, weight=ft.FontWeight.W_500),
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    profile_card,
                    section_label,
                    row_language,
                    row_help,
                    row_version,
                ],
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
                scroll=ft.ScrollMode.ALWAYS,
            )
        ]
