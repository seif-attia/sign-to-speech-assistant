import asyncio
import flet as ft
from frontend.views.home_view import HomeView
from frontend.views.sign_to_speech_view import SignToSpeechView


async def main(page: ft.Page) -> None:
    # --- Page Configuration ---
    page.title = "Sign Language Assistant"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.keep_screen_on = True  # Prevents display dimming or sleeping during active camera tracking

    async def route_change():
        # 1. Cleanup the current view's hardware loops before switching
        if page.views:
            current_view = page.views[-1]
            if hasattr(current_view, 'cleanup_async'):
                await current_view.cleanup_async()

        # 2. Rebuild the view stack
        page.views.clear()
        
        # Always mount Home underneath
        page.views.append(HomeView(page))
        
        # Stack the target page on top
        if page.route == "/sign-speech":
            page.views.append(SignToSpeechView(page))
            
        page.update()

    async def view_pop(e: ft.ViewPopEvent):
        """Handles physical back buttons / Swipe """
        top_view = page.views[-1]
        if hasattr(top_view, 'cleanup_async'):
            await top_view.cleanup_async()
            
        page.views.pop()
        top_view = page.views[-1]
        await page.push_route(top_view.route)

    async def on_disconnect(e):
        """Handle forced app closures (e.g., user closes the browser/app window)."""
        if page.views:
            current_view = page.views[-1]
            if hasattr(current_view, 'cleanup_async'):
                await current_view.cleanup_async()

    # --- Router Configuration ---
    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.on_disconnect = on_disconnect
    # Initialize app at the home route
    await route_change()

if __name__ == "__main__":
    ft.run(main)    
