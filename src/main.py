import flet as ft
from flet.messaging.session import Session

# --- Gracefully handle late invoke_method responses from disposed controls ---
_original_handle_invoke = Session.handle_invoke_method_results

def _safe_handle_invoke_method_results(self, control_id: int, call_id: str, result, error):
    """
    Prevents crashing when asynchronous invoke_method responses (e.g. from camera,
    audio, recorder) return from Flutter after a control was unmounted during navigation.
    """
    try:
        _original_handle_invoke(self, control_id, call_id, result, error)
    except RuntimeError as exc:
        if "is not registered" in str(exc):
            # Control was unmounted/disposed while Flutter bridge call was in flight.
            # Safely resolve any pending future so callers don't hang, without throwing.
            method_calls = getattr(self, "_Session__method_calls", {})
            method_call_results = getattr(self, "_Session__method_call_results", {})
            evt = method_calls.pop(call_id, None)
            if evt is not None:
                method_call_results[evt] = (result, error)
                evt.set()
        else:
            raise

Session.handle_invoke_method_results = _safe_handle_invoke_method_results

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
import flet_audio as fa
from services.network_stream_service import get_global_streamer, stop_global_streamer


async def main(page: ft.Page) -> None:
    # Initialize persistent WebSocket connection to backend from the moment the app starts
    get_global_streamer()

    # Register persistent Audio player in page.services so TTS playback is ready across all views
    tts_player = fa.Audio(autoplay=False, volume=1.0)
    setattr(page, "_app_tts_audio_service", tts_player)
    if hasattr(page, "services"):
        page.services.append(tts_player)

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
        stop_global_streamer()

    # --- Router Configuration ---
    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.on_disconnect = on_disconnect

    # Initialize app at the home route
    await route_change()


if __name__ == "__main__":
    ft.run(main)
