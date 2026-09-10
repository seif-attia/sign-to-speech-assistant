import flet as ft
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
    create_header,
    create_lang_toggle,
    VIEWPORT_WIDTH,
)
from frontend.components.bottom_nav_bar import create_nav_bar
from components.mic_button import MicButton

class SpeechToSpeechView(ft.View):
    """
    Screen 5: Speech-to-Speech Mode View.
    Renders 'Speech Mode', animated soundwave visualizer,
    spoken output buffer, and microphone toggle button.
    """

    def __init__(self, page: ft.Page):
        self.app_page = page

        async def go_back(e):
            await self.cleanup_async()
            await self.app_page.push_route("/")

        super().__init__(
            route="/speech-speech",
            bgcolor=BG_DARK,
            padding=ft.Padding.only(top=16, left=16, right=16, bottom=16),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.START,
            navigation_bar=create_nav_bar(0, self.app_page),
        )

        header = create_header(
            title="Speech Mode",
            on_back=go_back,
            right_control=create_lang_toggle("Ar ⇄ En"),
        )

        listening_state = ft.Text(
            "Listening in Arabic...",
            size=14,
            color=TEXT_SECONDARY,
            weight=ft.FontWeight.W_500,
        )

        # Visualizer bars simulation
        wave_heights = [18, 32, 48, 24, 60, 42, 70, 30, 52, 22, 40, 16]
        wave_bars = ft.Row(
            controls=[
                ft.Container(
                    width=4,
                    height=h,
                    bgcolor=ACCENT_EMERALD if i % 2 == 0 else ACCENT_MINT,
                    border_radius=2,
                )
                for i, h in enumerate(wave_heights)
            ],
            spacing=5,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self.recognized_output = ft.Text(
            '"صباح الخير كيف حالك؟"',
            size=16,
            color=TEXT_PRIMARY,
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER,
        )

        visualizer_card = create_glass_card(
            content=ft.Column(
                controls=[
                    listening_state,
                    ft.Container(content=wave_bars, height=80, alignment=ft.Alignment.CENTER),
                    self.recognized_output,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=16,
            ),
            padding=24,
            border_radius=20,
            width=VIEWPORT_WIDTH,
        )

        self.mic_button = MicButton()

        tap_pause_label = ft.Text(
            "TAP TO PAUSE LISTENING",
            size=11,
            weight=ft.FontWeight.BOLD,
            color=TEXT_MUTED,
            letter_spacing=1.5,
        )

        self.controls = [
            ft.Column(
                controls=[
                    header,
                    ft.Container(height=16),
                    visualizer_card,
                    ft.Container(height=36),
                    self.mic_button,
                    ft.Container(height=8),
                    tap_pause_label,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.START,
                expand=True,
            )
        ]

    async def cleanup_async(self):
        if hasattr(self, "mic_button") and self.mic_button:
            await self.mic_button.cleanup_async()
