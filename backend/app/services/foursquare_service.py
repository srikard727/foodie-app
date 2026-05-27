import httpx

from app.core.config import (
    FOURSQUARE_RESTAURANT_CATEGORY_ID,
    FOURSQUARE_RADIUS_METERS,
    FOURSQUARE_SEARCH_LIMIT,
    FOURSQUARE_TIPS_LIMIT,
    settings,
)

FOURSQUARE_API_VERSION = "1970-01-01"
FOURSQUARE_PLACE_FIELDS = ",".join(
    [
        "fsq_id",
        "name",
        "categories",
        "description",
        "location",
        "geocodes",
        "distance",
        "rating",
        "price",
        "photos",
        "tips",
        "tastes",
        "website",
    ]
)
FOURSQUARE_PLACE_URL = "https://api.foursquare.com/v3/places/{fsq_id}"
FOURSQUARE_SEARCH_URL = "https://api.foursquare.com/v3/places/search"
FOURSQUARE_TIPS_URL = "https://api.foursquare.com/v3/places/{fsq_id}/tips"


def get_foursquare_headers() -> dict:
    api_key = settings.foursquare_api_key

    if not api_key:
        raise RuntimeError("FOURSQUARE_API_KEY is not set.")

    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Places-Api-Version": FOURSQUARE_API_VERSION,
    }


def search_restaurants(message: str, latitude: float, longitude: float) -> list[dict]:
    params = {
        "query": message,
        "ll": f"{latitude},{longitude}",
        "radius": FOURSQUARE_RADIUS_METERS,
        "categories": FOURSQUARE_RESTAURANT_CATEGORY_ID,
        "limit": FOURSQUARE_SEARCH_LIMIT,
        "sort": "RELEVANCE",
        "fields": FOURSQUARE_PLACE_FIELDS,
    }

    response = httpx.get(
        FOURSQUARE_SEARCH_URL,
        headers=get_foursquare_headers(),
        params=params,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    return data.get("results", [])


def get_place_details(fsq_id: str) -> dict:
    response = httpx.get(
        FOURSQUARE_PLACE_URL.format(fsq_id=fsq_id),
        headers=get_foursquare_headers(),
        params={"fields": FOURSQUARE_PLACE_FIELDS},
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        return data

    return {}


def get_place_tips(fsq_id: str) -> list[dict]:
    response = httpx.get(
        FOURSQUARE_TIPS_URL.format(fsq_id=fsq_id),
        headers=get_foursquare_headers(),
        params={
            "limit": FOURSQUARE_TIPS_LIMIT,
            "sort": "POPULAR",
        },
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list):
        return data

    return data.get("results", [])
