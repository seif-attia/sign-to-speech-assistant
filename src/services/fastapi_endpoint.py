import asyncio
import httpx
from services.config import get_server_base_url


# ========================================================
#            THE 2 ABSTRACTED CLIENT FUNCTIONS
# ========================================================

async def send_prompt(prompt_text: str) -> str:
    """Sends the gloss prompt to the server and returns a job_id string."""
    base_url = get_server_base_url()
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.post(
            f"{base_url}/send_prompt", json={"prompt": prompt_text}
        )
        return res.json()["job_id"]


async def get_response(job_id: str) -> dict:
    """Polls the server for job status. Returns dict with status and response."""
    base_url = get_server_base_url()
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.get(f"{base_url}/get_response/{job_id}")
        return res.json()


async def generate_arabic_translation(gloss_tokens: list[str]) -> str:
    """Higher-level helper combining send and poll into a single awaitable call."""
    prompt_str = f"Gloss: {' '.join(gloss_tokens)}"
    job_id = await send_prompt(prompt_str)

    # Poll until ready
    while True:
        data = await get_response(job_id)
        if data["status"] == "completed":
            return data["response"]
        elif data["status"] in ("failed", "not_found"):
            return f"Error: {data.get('response', 'Unknown error')}"

        await asyncio.sleep(0.4)