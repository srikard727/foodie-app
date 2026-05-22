from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

user_request = "I want spicy cheap food"
user_tags = ["spicy", "cheap", "quick"]

query_text = user_request + " " + " ".join(user_tags)

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

query_embedding = model.encode(query_text, convert_to_tensor=True)

food_texts = [food["description"] for food in food_options]
food_embeddings = model.encode(food_texts, convert_to_tensor=True)
    
scores = util.cos_sim(query_embedding, food_embeddings)[0]

ranked_results = sorted(
    zip(food_options, scores),
    key=lambda item: item[1],
    reverse=True,
)

for food, score in ranked_results:
    print(food["name"], float(score))