"""Small, deterministic query parser; keep user text for semantic ranking."""
import re
from dataclasses import dataclass

PRICE_TERMS = {"cheap": 1, "budget": 1, "affordable": 2, "inexpensive": 2, "expensive": 4, "fancy": 4, "upscale": 4}
FOOD_TERMS = ("ramen", "sushi", "pizza", "burger", "burgers", "tacos", "thai", "indian", "italian", "chinese", "korean", "japanese", "vietnamese", "pho", "noodles", "curry", "bbq", "sandwich", "seafood", "vegetarian", "vegan", "breakfast", "brunch", "coffee", "dessert")
STOPWORDS = {"i", "want", "need", "some", "a", "an", "the", "please", "find", "me", "near", "nearby", "around", "here", "for", "tonight", "today", "something", "food", "restaurant", "restaurants", "to", "eat", "with", "and", "that", "is", "are", "good", "best"}

@dataclass(frozen=True)
class FoodIntent:
    search_query: str
    max_price: int | None
    preferences: tuple[str, ...]

def understand_query(message: str) -> FoodIntent:
    words = re.findall(r"[a-z]+", message.lower())
    price = next((PRICE_TERMS[word] for word in words if word in PRICE_TERMS), None)
    food = [term for term in FOOD_TERMS if term in message.lower().split() or term in message.lower()]
    # Prefer a short dish/cuisine query for provider retrieval; preserve original text for embedding.
    query = " ".join(food[:2]) if food else " ".join(w for w in words if w not in STOPWORDS and w not in PRICE_TERMS)[:70]
    return FoodIntent(search_query=query or "restaurant", max_price=price, preferences=tuple(w for w in words if w not in STOPWORDS))
