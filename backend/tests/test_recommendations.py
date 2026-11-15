"""Offline API and ranking tests: no provider credentials or model download required."""
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.query_understanding import understand_query

client = TestClient(app)
PLACE = {"fsq_place_id": "123", "name": "Ramen House", "latitude": 37.77, "longitude": -122.42, "rating": 8.5, "price": 2}

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_invalid_coordinates():
    response = client.post("/recommendations", json={"message": "ramen", "latitude": 100, "longitude": 0})
    assert response.status_code == 422

def test_blank_message():
    response = client.post("/recommendations", json={"message": "   ", "latitude": 0, "longitude": 0})
    assert response.status_code == 422

def test_intent():
    intent = understand_query("I want cheap spicy ramen nearby")
    assert intent.search_query == "ramen"
    assert intent.max_price == 1

def test_recommendation_contract():
    ranked = [{**PLACE, "score": 0.8, "semantic_score": 0.8, "distance_km": 1.0, "address": None, "image_url": None, "reasons": ["Matches your food request"]}]
    with patch("app.main.rank_restaurants", return_value=ranked):
        response = client.post("/recommendations", json={"message": "ramen", "latitude": 37.77, "longitude": -122.42})
    assert response.status_code == 200
    data = response.json()
    assert data["recommendation"]["name"] == "Ramen House"
    assert data["alternatives"] == []

def test_no_results():
    with patch("app.main.rank_restaurants", return_value=[]):
        response = client.post("/recommendations", json={"message": "ramen", "latitude": 0, "longitude": 0})
    assert response.status_code == 404
