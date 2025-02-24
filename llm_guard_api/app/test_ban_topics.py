import pytest
from fastapi.testclient import TestClient
from .guard import app  
client = TestClient(app)

def test_ban_topics_valid():
    data = {
        "prompt": "This is a safe prompt without any banned topics.",
        "topics": ["Sports", "Technology"],  # No banned topics here
        "threshold": 0.5
    }
    response = client.post("/ban-topics", json=data)
    assert response.status_code == 200
    assert response.json() == {"message": "Prompt is valid", "risk_score": 0.0}

def test_ban_topics_invalid_banned_topic():
    data = {
        "prompt": "This prompt contains violence.",
        "topics": ["Violence", "Racism"],
        "threshold": 0.5
    }
    response = client.post("/ban-topics", json=data)
    assert response.status_code == 400
    assert response.json() == {"detail": "Prompt contains banned topics: Violence, Racism"}

def test_ban_topics_no_banned_topics():
    data = {
        "prompt": "Let's talk about the latest tech trends.",
        "topics": ["Technology", "Science"],
        "threshold": 0.5
    }
    response = client.post("/ban-topics", json=data)
    assert response.status_code == 200
    assert response.json() == {"message": "Prompt is valid", "risk_score": 0.0}
