import httpx
import anthropic
from google import genai

async def fetch_models(provider: str, api_key: str) -> list[str]:
    if provider == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        models = client.models.list()
        return [m.id for m in models.data]

    elif provider == "gemini":
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        return [m.name.replace("models/", "") for m in models if "generateContent" in (m.supported_actions or [])]

    raise ValueError(f"Unknown provider: {provider}")
