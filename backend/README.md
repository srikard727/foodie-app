# Foodie recommendation backend

FastAPI service returning a recommended restaurant plus alternatives for a natural-language food request and coordinates.

## API

`GET /health` checks the API.

`POST /recommendations`:
```json
{"message":"I want affordable spicy ramen nearby","latitude":37.77,"longitude":-122.42}
```

Returns `recommendation`, `alternatives`, and the embedding `model` name. Each venue includes a ranking score, semantic score, distance (when available), and short, **attribute-based** reasons. Reasons are heuristics, not verified claims about individual menu items.

## Pipeline

1. Deterministically extract food keywords and approximate price preference from the request.
2. Retrieve candidates through the **current Foursquare Places API**, using coordinates and optional maximum price.
3. Encode the original full request and candidate descriptions with pretrained MiniLM.
4. Rank using a transparent baseline: 70% semantic similarity, 15% proximity, 10% provider rating, 5% price fit. Missing attributes are treated neutrally.
5. Return the best result and alternatives. The embedding model loads lazily.

This is an initial hybrid-ranking baseline, **not a measured accuracy improvement**. Next: build a labeled query/venue evaluation set, compare BGE against MiniLM using NDCG@5, then tune weights. No LLM is needed to invent venue descriptions.

## Local development

From `backend/`:

```bash
uv sync
# set FOURSQUARE_API_KEY in backend/.env
uv run fastapi dev app/main.py
```

Run offline tests (pytest must be installed): `uv run --with pytest pytest tests -q`.

Foursquare's retired `api.foursquare.com/v3` routes were replaced by `https://places-api.foursquare.com/places/search`. Verify your Foursquare API key has access to the new Places API and review provider pricing before live testing. A live request and model quality benchmark have not been verified in this PR.
