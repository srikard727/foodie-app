import httpx
from fastapi import FastAPI, HTTPException
from app.schemas.recommendation import RecommendationRequest
from app.services.recommendation_service import rank_restaurants

app = FastAPI(title="Foodie Recommendation API", version="0.2.0")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommendations")
def give_food_recommendation(request: RecommendationRequest):
    try:
        results = rank_restaurants(request.message, request.latitude, request.longitude)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail="Restaurant recommendations are not configured.") from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=502, detail="Restaurant provider returned an error.") from error
    except httpx.RequestError as error:
        raise HTTPException(status_code=502, detail="Restaurant provider is temporarily unavailable.") from error

    if not results:
        raise HTTPException(status_code=404, detail="No matching restaurants found.")

    best, *alternatives = results
    return {
        "recommendation": {
            **best,
            "message": f"Try {best['name']}. " + "; ".join(best["reasons"]) + ".",
        },
        "alternatives": [
            {**place, "reason": "; ".join(place["reasons"]) + "."}
            for place in alternatives
        ],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
