import httpx
from sentence_transformers import SentenceTransformer, util

from app.core.config import FOURSQUARE_DETAIL_LIMIT, RECOMMENDATION_LIMIT
from app.services.foursquare_service import (
    get_place_details,
    get_place_tips,
    search_restaurants,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_restaurant_text(restaurant: dict) -> str:
    categories = " ".join(
        category.get("name", "") for category in restaurant.get("categories", [])
    )
    address = restaurant.get("location", {}).get("formatted_address", "")
    tastes = " ".join(restaurant.get("tastes", []))
    tips = " ".join(tip.get("text", "") for tip in restaurant.get("tips", []))
    rating = restaurant.get("rating")

    return " ".join(
        str(part)
        for part in [
            restaurant.get("name", ""),
            restaurant.get("description", ""),
            categories,
            tastes,
            tips,
            f"price {restaurant.get('price')}" if restaurant.get("price") else "",
            f"rating {rating}" if rating else "",
            address,
        ]
        if part
    )


def format_address(restaurant: dict) -> str | None:
    formatted_address = restaurant.get("location", {}).get("formatted_address")

    if formatted_address:
        return formatted_address

    address_parts = [
        restaurant.get("location", {}).get("address"),
        restaurant.get("location", {}).get("locality"),
        restaurant.get("location", {}).get("region"),
        restaurant.get("location", {}).get("postcode"),
    ]
    address = ", ".join(part for part in address_parts if part)

    return address or None


def build_photo_url(restaurant: dict) -> str | None:
    photos = restaurant.get("photos", [])

    if not photos:
        return None

    photo = photos[0]
    prefix = photo.get("prefix")
    suffix = photo.get("suffix")

    if not prefix or not suffix:
        return None

    return f"{prefix}original{suffix}"


def normalize_restaurant(
    restaurant: dict,
    restaurant_text: str,
    score: float,
) -> dict:
    coordinates = restaurant.get("geocodes", {}).get("main", {})

    return {
        "id": restaurant.get("fsq_id"),
        "name": restaurant["name"],
        "description": restaurant_text,
        "latitude": coordinates.get("latitude", restaurant.get("latitude")),
        "longitude": coordinates.get("longitude", restaurant.get("longitude")),
        "rating": restaurant.get("rating"),
        "price": restaurant.get("price"),
        "address": format_address(restaurant),
        "image_url": build_photo_url(restaurant),
        "score": score,
    }


def score_restaurants(message: str, restaurants: list[dict]) -> list[dict]:
    if not restaurants:
        return []

    query_embedding = model.encode(message, convert_to_tensor=True)

    restaurant_texts = [build_restaurant_text(restaurant) for restaurant in restaurants]
    restaurant_embeddings = model.encode(restaurant_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, restaurant_embeddings)[0]

    ranked_results = sorted(
        zip(restaurants, restaurant_texts, scores),
        key=lambda item: float(item[2]),
        reverse=True,
    )

    return [
        normalize_restaurant(restaurant, restaurant_text, float(score))
        for restaurant, restaurant_text, score in ranked_results
    ]


def get_detailed_restaurants(restaurants: list[dict]) -> list[dict]:
    detailed_restaurants = []

    for restaurant in restaurants[:FOURSQUARE_DETAIL_LIMIT]:
        fsq_id = restaurant.get("id")

        if not fsq_id:
            detailed_restaurants.append(restaurant)
            continue

        try:
            details = get_place_details(fsq_id)
            tips = get_place_tips(fsq_id)
        except httpx.HTTPError:
            detailed_restaurants.append(restaurant)
            continue

        detailed_restaurants.append({**restaurant, **details, "tips": tips})

    return detailed_restaurants


def rank_restaurants(
    message: str,
    latitude: float,
    longitude: float,
    limit: int = RECOMMENDATION_LIMIT,
) -> list[dict]:
    restaurants = search_restaurants(message, latitude, longitude)
    scored_restaurants = score_restaurants(message, restaurants)
    detailed_restaurants = get_detailed_restaurants(scored_restaurants)

    return score_restaurants(message, detailed_restaurants)[:limit]
