from fastapi import FastAPI
from app.schemas.recommendation import RecommendationRequest
from app.services.recommendation_service import rank_food_options

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommendations")
def give_food_recommendation(request: RecommendationRequest):
    top_results = rank_food_options(request.message)

    best = top_results[0]
    alternatives = top_results[1:]

    return {
        "recommendation": {
            "name": best["name"],
            "message": f"Try {best['name']} tonight.",
            "score": best["score"],
        },
        "alternatives": [
            {
                "name": food["name"],
                "reason": "Also matches your food request.",
                "score": food["score"],
            }
            for food in alternatives
        ],
    }