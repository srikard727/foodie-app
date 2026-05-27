from fastapi import FastAPI, HTTPException
from app.schemas.recommendation import RecommendationRequest
from app.services.recommendation_service import rank_restaurants

app = FastAPI()


def build_recommendation_response(restaurant: dict) -> dict:
    return {
        "name": restaurant["name"],
        "latitude": restaurant["latitude"],
        "longitude": restaurant["longitude"],
        "rating": restaurant["rating"],
        "price": restaurant["price"],
        "address": restaurant["address"],
        "image_url": restaurant["image_url"],
        "score": restaurant["score"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommendations")
def give_food_recommendation(request: RecommendationRequest):
    try:
        top_results = rank_restaurants(
            request.message,
            request.latitude,
            request.longitude,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if not top_results:
        raise HTTPException(status_code=404, detail="No restaurant recommendations found.")

    best = top_results[0]
    alternatives = top_results[1:]

    return {
        "recommendation": {
            **build_recommendation_response(best),
            "message": f"Try {best['name']} tonight.",
        },
        "alternatives": [
            {
                **build_recommendation_response(food),
                "reason": "Also matches your food request.",
            }
            for food in alternatives
        ],
    }
