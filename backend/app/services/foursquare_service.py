"""Foursquare Places API (2025-06-17), not the retired legacy v3 API."""
import httpx
from app.core.config import FOURSQUARE_RADIUS_METERS, FOURSQUARE_SEARCH_LIMIT, settings

BASE_URL = "https://places-api.foursquare.com"
API_VERSION = "2025-06-17"

def get_foursquare_headers() -> dict:
    if not settings.foursquare_api_key:
        raise RuntimeError("FOURSQUARE_API_KEY is not set.")
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.foursquare_api_key}",
        "X-Places-Api-Version": API_VERSION,
    }

def search_restaurants(query: str, latitude: float, longitude: float, max_price: int | None = None) -> list[dict]:
    params = {
        "query": query,
        "ll": f"{latitude},{longitude}",
        "radius": FOURSQUARE_RADIUS_METERS,
        "limit": min(FOURSQUARE_SEARCH_LIMIT, 50),
        "sort": "RELEVANCE",
    }
    if max_price is not None:
        params["max_price"] = max_price
    response = httpx.get(
        f"{BASE_URL}/places/search",
        headers=get_foursquare_headers(),
        params=params,
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("results", [])
