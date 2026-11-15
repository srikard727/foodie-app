"""Retrieve candidates, rank by semantic relevance and grounded venue attributes."""
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from sentence_transformers import SentenceTransformer
from app.core.config import RECOMMENDATION_LIMIT
from app.services.foursquare_service import search_restaurants
from app.services.query_understanding import understand_query

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_model():
    # Lazy loading keeps health checks and API startup independent of model downloads.
    return SentenceTransformer(MODEL_NAME)

def build_restaurant_text(place: dict) -> str:
    categories = " ".join(c.get("name", "") for c in place.get("categories", []))
    tastes = " ".join(place.get("tastes") or [])
    tips = " ".join(t.get("text", "") for t in place.get("tips") or [])
    return " ".join(str(x) for x in [place.get("name"), categories, place.get("description"), tastes, tips] if x)

def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    a = sin(radians(lat2-lat1)/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(radians(lon2-lon1)/2)**2
    return 2*r*asin(sqrt(a))

def normalize_restaurant(place: dict, score: float, semantic_score: float, distance: float | None, reasons: list[str]) -> dict:
    location = place.get("location") or {}
    address = location.get("formatted_address") or ", ".join(str(location[k]) for k in ("address", "locality", "region") if location.get(k)) or None
    photos = place.get("photos") or []
    photo = photos[0] if photos else {}
    image_url = f"{photo['prefix']}original{photo['suffix']}" if photo.get("prefix") and photo.get("suffix") else None
    return {
        "id": place.get("fsq_place_id") or place.get("fsq_id"),
        "name": place["name"],
        "latitude": place.get("latitude") or (place.get("geocodes") or {}).get("main", {}).get("latitude"),
        "longitude": place.get("longitude") or (place.get("geocodes") or {}).get("main", {}).get("longitude"),
        "rating": place.get("rating"),
        "price": place.get("price"),
        "address": address,
        "image_url": image_url,
        "score": round(score, 4),
        "semantic_score": round(semantic_score, 4),
        "distance_km": round(distance, 2) if distance is not None else None,
        "reasons": reasons,
    }

def score_restaurants(message: str, restaurants: list[dict], latitude: float, longitude: float, max_price: int | None = None) -> list[dict]:
    places = [p for p in restaurants if p.get("name")]
    if not places:
        return []
    model = get_model()
    texts = [build_restaurant_text(p) for p in places]
    query_embedding = model.encode([message], normalize_embeddings=True)
    place_embeddings = model.encode(texts, normalize_embeddings=True)
    # Normalized dot product is cosine similarity. Clamp to [0,1] for hybrid weighting.
    similarities = (place_embeddings @ query_embedding[0]).tolist()
    results = []
    for place, raw_similarity in zip(places, similarities):
        semantic = max(0.0, min(1.0, float(raw_similarity)))
        lat = place.get("latitude") or (place.get("geocodes") or {}).get("main", {}).get("latitude")
        lon = place.get("longitude") or (place.get("geocodes") or {}).get("main", {}).get("longitude")
        distance = distance_km(latitude, longitude, float(lat), float(lon)) if lat is not None and lon is not None else None
        proximity = max(0.0, 1.0 - distance / 8.0) if distance is not None else 0.5
        rating = place.get("rating")
        quality = max(0.0, min(1.0, float(rating)/10)) if rating is not None else 0.5
        price = place.get("price")
        # Unknown prices are neutral rather than assumed affordable.
        price_fit = 1.0 if max_price is None or (price is not None and int(price) <= max_price) else (0.5 if price is None else 0.0)
        score = 0.70*semantic + 0.15*proximity + 0.10*quality + 0.05*price_fit
        reasons = ["Matches your food request"] if semantic >= 0.45 else ["Related to your food request"]
        if distance is not None and distance <= 2:
            reasons.append("Nearby")
        if max_price is not None and price is not None and int(price) <= max_price:
            reasons.append("Within your price preference")
        results.append(normalize_restaurant(place, score, semantic, distance, reasons))
    return sorted(results, key=lambda p: p["score"], reverse=True)

def rank_restaurants(message: str, latitude: float, longitude: float, limit: int = RECOMMENDATION_LIMIT) -> list[dict]:
    intent = understand_query(message)
    candidates = search_restaurants(intent.search_query, latitude, longitude, intent.max_price)
    return score_restaurants(message, candidates, latitude, longitude, intent.max_price)[:limit]
