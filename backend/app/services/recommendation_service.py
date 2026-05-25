from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


FOOD_OPTIONS = [
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


def rank_food_options(message: str, limit: int = 3) -> list[dict]:
    query_embedding = model.encode(message, convert_to_tensor=True)

    food_texts = [food["description"] for food in FOOD_OPTIONS]
    food_embeddings = model.encode(food_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, food_embeddings)[0]

    ranked_results = sorted(
        zip(FOOD_OPTIONS, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    top_results = ranked_results[:limit]

    return [
        {
            "name": food["name"],
            "description": food["description"],
            "score": float(score),
        }
        for food, score in top_results
    ]