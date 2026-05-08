"""
Smoke tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_search():
    response = client.post("/api/search", json={
        "query": "graph neural networks for recommendation",
        "k": 2,
        "use_gnn": True
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query_time_ms" in data
    assert "model_used" in data
    assert len(data["results"]) <= 2

def test_suggest():
    response = client.post("/api/suggest", json={
        "project_description": "We are building a recommender system using GNNs.",
        "k": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "retrieved_papers" in data
    assert "reasoning_paths" in data
    assert "suggestions" in data
    assert "grounding_note" in data

def test_graph():
    response = client.get("/api/graph")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert "stats" in data

def test_stats():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "R-GCN (Contrastive)" in data["models"]
