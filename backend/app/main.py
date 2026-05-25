from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class RecommendationRequest(BaseModel):
    message: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommendations")
def give_food_recommendation(request: RecommendationRequest):
    top_results = rank_food_options(request)

    best_food, best_score = top_results[0]
    alternatives = top_results[1:]

    return {
        "recommendation": {
            "name": best_food["name"],
            "message": f"Try {best_food['name']} tonight.",
            "score": float(best_score),
        },
        "alternatives": [
            {
                "name": food["name"],
                "reason": "Also matches your food request.",
                "score": float(score),
            }
            for food, score in alternatives
        ],
    }

def rank_food_options(request: RecommendationRequest):
    query_embedding = model.encode(request.message, convert_to_tensor=True)

    food_options = [
        {
            "name": "Thai Curry",
            "description": "spicy warm coconut curry flavorful dinner affordable",
        },
        {
            "name": "Burrito Bowl",
            "description": "cheap filling quick rice beans protein casual",
        },
        {
            "name": "Sushi",
            "description": "light fresh seafood rice date night expensive",
        },
        {
            "name": "Ramen",
            "description": "warm noodles broth comfort food savory late night",
        },
    ]

    food_texts = [food["description"] for food in food_options]
    food_embeddings = model.encode(food_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, food_embeddings)[0]

    ranked_results = sorted(
        zip(food_options, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    top_three_results = ranked_results[:3]
    return top_three_results
