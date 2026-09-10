import flet as ft
from frontend.views.intro_view import IntroView
from frontend.views.home_view import HomeView
from frontend.views.sign_to_speech_view import SignToSpeechView
from frontend.views.text_to_text_view import TextToTextView
from frontend.views.speech_text_view import SpeechToTextView
from frontend.views.speech_to_speech_view import SpeechToSpeechView
from frontend.views.text_to_speech_view import TextToSpeechView
from frontend.views.educational_view import EducationalView
from frontend.views.contribution_view import ContributionView
from frontend.views.settings_view import SettingsView
from frontend.theme import BG_DARK


async def main(page: ft.Page) -> None:
    # --- Page Configuration ---
    page.title = "WESAL - Sign & Speech Assistant"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = BG_DARK
    page.keep_screen_on = True  # Prevents display dimming or sleeping during active camera tracking

    async def route_change():
        # 1. Cleanup the current view's hardware loops before switching
        if page.views:
            current_view = page.views[-1]
            if hasattr(current_view, "cleanup_async"):
                await current_view.cleanup_async()

        # 2. Rebuild the view stack
        page.views.clear()

        # Always mount Home underneath if navigating deeper
        if page.route == "/intro":
            page.views.append(IntroView(page))
        elif page.route == "/":
            page.views.append(HomeView(page))
        else:
            page.views.append(HomeView(page))
            if page.route == "/sign-speech":
                page.views.append(SignToSpeechView(page))
            elif page.route == "/text-text":
                page.views.append(TextToTextView(page))
            elif page.route == "/speech-text":
                page.views.append(SpeechToTextView(page))
            elif page.route == "/speech-speech":
                page.views.append(SpeechToSpeechView(page))
            elif page.route == "/text-speech":
                page.views.append(TextToSpeechView(page))
            elif page.route == "/learn":
                page.views.append(EducationalView(page))
            elif page.route == "/contribute":
                page.views.append(ContributionView(page))
            elif page.route == "/settings":
                page.views.append(SettingsView(page))

        page.update()

    async def view_pop(e: ft.ViewPopEvent):
        """Handles physical back buttons / Swipe gestures."""
        top_view = page.views[-1]
        if hasattr(top_view, "cleanup_async"):
            await top_view.cleanup_async()

        page.views.pop()
        top_view = page.views[-1]
        await page.push_route(top_view.route)

    async def on_disconnect(e):
        """Handle forced app closures."""
        if page.views:
            current_view = page.views[-1]
            if hasattr(current_view, "cleanup_async"):
                await current_view.cleanup_async()

    # --- Router Configuration ---
    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.on_disconnect = on_disconnect

    # Initialize app at the home route
    await route_change()


if __name__ == "__main__":
    ft.run(main)
