import asyncio
import logging
import os
import tempfile
import time
import httpx
import flet as ft
import flet_audio as fa
from services.config import get_server_base_url

logger = logging.getLogger(__name__)

AUDIO_SERVICE_KEY = "_app_tts_audio_service"
_current_tts_player = None

# ========================================================
#            THE ABSTRACTED CLIENT FUNCTIONS
# ========================================================

async def send_prompt(prompt_text: str, target_lang: str = "en") -> str:
    """Sends the gloss prompt to the server and returns a job_id string."""
    base_url = get_server_base_url()
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.post(
            f"{base_url}/send_prompt",
            json={"prompt": prompt_text, "target_lang": target_lang},
        )
        return res.json().get("job_id", "")


async def get_response(job_id: str) -> dict:
    """Polls the server for job status. Returns dict with status and response."""
    base_url = get_server_base_url()
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.get(f"{base_url}/get_response/{job_id}")
        return res.json()


async def generate_arabic_translation(gloss_tokens: list[str]) -> str:
    """Combines send and poll into a single awaitable call (Arabic)."""
    prompt_str = f"Gloss: {' '.join(gloss_tokens)}"
    job_id = await send_prompt(prompt_str, target_lang="ar")

    while True:
        data = await get_response(job_id)
        if data.get("status") == "completed":
            return data.get("response", "")
        elif data.get("status") in ("failed", "not_found"):
            return f"Error: {data.get('response', 'Unknown error')}"
        await asyncio.sleep(0.4)


async def translate_text(text: str, source_lang: str = "ar", target_lang: str = "en") -> str:
    """Direct translation between Arabic and English via FastAPI /translate."""
    if not text.strip():
        return ""
    base_url = get_server_base_url()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(
                f"{base_url}/translate",
                json={"text": text, "source_lang": source_lang, "target_lang": target_lang},
            )
            if res.status_code == 200:
                data = res.json()
                return data.get("translation", "")
            return f"Error ({res.status_code})"
    except Exception as e:
        logger.error(f"Translation request error: {e}")
        return f"Connection error: {e}"


async def transcribe_audio(wav_bytes: bytes, lang: str = "ar") -> dict:
    """Sends recorded WAV bytes to backend /transcribe endpoint for speech recognition."""
    if not wav_bytes or len(wav_bytes) < 100:
        return {"text": "", "error": "No audio recorded"}

    base_url = get_server_base_url()
    b64_audio = base64.b64encode(wav_bytes).decode("ascii")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                f"{base_url}/transcribe",
                json={"audio_b64": b64_audio, "lang": lang},
            )
            if res.status_code == 200:
                return res.json()
            return {"text": "", "error": f"Server error ({res.status_code})"}
    except Exception as e:
        logger.error(f"Transcription network error: {e}")
        return {"text": "", "error": f"Connection error: {e}"}


async def synthesize_speech_bytes(text: str, lang: str = "en", speed: float = 1.0) -> bytes:
    """Synthesizes text using backend Sherpa-ONNX and returns WAV bytes."""
    if not text.strip():
        return b""
    base_url = get_server_base_url()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                f"{base_url}/tts",
                json={"text": text, "lang": lang, "speed": speed},
            )
            if res.status_code == 200:
                return res.content
            logger.warning(f"TTS endpoint returned status {res.status_code}")
            return b""
    except Exception as e:
        logger.error(f"TTS request error: {e}")
        return b""


async def get_speech_url(text: str, lang: str = "en", speed: float = 1.0) -> str:
    """Gets a direct HTTP audio stream URL from backend Sherpa-ONNX."""
    if not text.strip():
        return ""
    base_url = get_server_base_url()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                f"{base_url}/tts_url",
                json={"text": text, "lang": lang, "speed": speed},
            )
            if res.status_code == 200:
                data = res.json()
                url = data.get("url", "")
                if url.startswith("http://") or url.startswith("https://"):
                    return url
                elif url:
                    return f"{base_url}{url}"
    except Exception as e:
        logger.error(f"TTS URL request error: {e}")
    return ""

import base64
import urllib.parse


def _get_or_create_audio_service(page: ft.Page) -> fa.Audio:
    """Retrieves or registers fa.Audio as a page service."""
    if hasattr(page, "services"):
        for svc in page.services:
            if isinstance(svc, fa.Audio):
                return svc

    player = getattr(page, AUDIO_SERVICE_KEY, None)
    if player is None:
        player = fa.Audio(autoplay=False, volume=1.0)
        setattr(page, AUDIO_SERVICE_KEY, player)
        if hasattr(page, "services"):
            page.services.append(player)
        page.update()

    return player


async def play_speech(page: ft.Page, text: str, lang: str = "en") -> None:
    """
    Fetches TTS audio from the FastAPI server and plays it directly as an in-memory
    audio data stream across all platforms (Android, iOS, Desktop) with no WAV disk files.
    """
    if not text or not page:
        return

    clean_text = text.strip()
    if not clean_text:
        return

    try:
        # 1. Fetch raw audio byte stream directly from backend
        wav_bytes = await synthesize_speech_bytes(text=clean_text, lang=lang, speed=1.0)

        # 2. Convert binary audio stream into an in-memory data URI stream
        # Flutter audioplayers natively handles data URIs in-memory as a stream
        if wav_bytes and len(wav_bytes) > 44:
            b64_data = base64.b64encode(wav_bytes).decode("ascii")
            stream_src = f"data:audio/wav;base64,{b64_data}"
            print(f"[TTS STREAM] Streamed {len(wav_bytes)} audio bytes in-memory.", flush=True)
        else:
            # Fallback directly to backend HTTP audio streaming URL
            base_url = get_server_base_url()
            encoded_text = urllib.parse.quote(clean_text)
            stream_src = f"{base_url}/tts_stream?text={encoded_text}&lang={lang}&speed=1.0"
            print(f"[TTS STREAM] Streaming directly via URL: {stream_src}", flush=True)

        player = _get_or_create_audio_service(page)

        # Release any active playback before starting new stream
        try:
            await player.release()
        except Exception:
            pass

        # Set stream source and trigger playback
        player.src = stream_src
        player.autoplay = True
        page.update()

        try:
            await player.play()
        except Exception as play_err:
            logger.debug(f"[TTS STREAM] player.play() call: {play_err}")

    except Exception as exc:
        logger.error(f"[TTS STREAM ERROR] {exc}")