"""
WESAL Design System Tokens & Theme Helpers
Obsidian dark theme with emerald & teal accents matching the Figma/PDF design specifications.
"""
import flet as ft

# Colors
BG_DARK = "#081010"            # Deep obsidian background
BG_SURFACE = "#111C1C"         # Surface container background
CARD_BG = "#152424"            # Glass-morphism card fill
CARD_BG_LIGHT = "#1C3030"      # Elevated card fill
CARD_BORDER = "#223E3E"        # Card outline border
ACCENT_EMERALD = "#10B981"     # Vibrant emerald green
ACCENT_TEAL = "#059669"        # Darker teal accent
ACCENT_MINT = "#6EE7B7"        # Bright mint for icons/indicators
ACCENT_GREEN_GLOW = "#00DF81"  # Neon glow accent
BADGE_BG = "#103B33"           # Dark teal pill background
TEXT_PRIMARY = "#FFFFFF"       # White text
TEXT_SECONDARY = "#9CA3AF"     # Muted grey text
TEXT_MUTED = "#6B7280"         # Darker muted grey text
TEXT_EMERALD = "#34D399"       # Emerald highlight text
INPUT_BG = "#101D1D"           # Textbox fill background

VIEWPORT_WIDTH = 370           # Standard mobile frame width

def create_glass_card(content: ft.Control, padding: int = 16, border_color: str = CARD_BORDER, bg_color: str = CARD_BG, border_radius: int = 16, width: int = VIEWPORT_WIDTH) -> ft.Container:
    """Helper to generate glass-morphism style card containers."""
    return ft.Container(
        content=content,
        padding=padding,
        border_radius=border_radius,
        bgcolor=bg_color,
        border=ft.Border.all(1, border_color),
        width=width,
    )

def create_pill_badge(text: str, bg_color: str = BADGE_BG, text_color: str = ACCENT_MINT, font_size: int = 11, icon: ft.Icon = None) -> ft.Container:
    """Helper to generate badge pills like 'CAM ACTIVE', 'LIVE TRANSCRIBING', '320 XP'."""
    row_controls = []
    if icon:
        row_controls.append(icon)
    row_controls.append(
        ft.Text(text, size=font_size, weight=ft.FontWeight.BOLD, color=text_color)
    )
    return ft.Container(
        content=ft.Row(controls=row_controls, spacing=4, alignment=ft.MainAxisAlignment.CENTER),
        bgcolor=bg_color,
        border_radius=12,
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
    )

def create_header(title: str, on_back=None, right_control: ft.Control = None) -> ft.Row:
    """Standard header for inner mode pages with '← Modes' or back arrow."""
    left_control = (
        ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ARROW_BACK_IOS_NEW, size=14, color=ACCENT_MINT),
                    ft.Text("Modes", size=13, weight=ft.FontWeight.W_500, color=ACCENT_MINT),
                ],
                spacing=4,
            ),
            on_click=on_back,
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
            border_radius=8,
            bgcolor="#132323",
        )
        if on_back
        else ft.Container(width=40)
    )

    return ft.Row(
        controls=[
            left_control,
            ft.Text(title, size=17, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            right_control if right_control else ft.Container(width=60),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

def create_lang_toggle(current_lang="Ar ⇄ En") -> ft.Container:
    """Pill button to toggle between Arabic and English."""
    return ft.Container(
        content=ft.Row(
            controls=[
                ft.Text(current_lang, size=12, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=4,
        ),
        bgcolor=CARD_BG,
        border=ft.Border.all(1, CARD_BORDER),
        border_radius=12,
        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
    )
