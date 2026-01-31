"""
Unit tests for the Purchase Prediction API

Run with: pytest test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from deployment.api.main import app, model_manager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestGeneralEndpoints:
    """Test general API endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "E-Commerce Purchase Prediction API"
        assert data["status"] == "running"
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert data["status"] in ["healthy", "degraded"]


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_predict_no_model(self, client):
        """Test prediction when no model is loaded."""
        # This test assumes no model is loaded initially
        if not model_manager.is_loaded():
            response = client.post("/predict", json={
                "features": [{
                    "user_id": "test_user",
                    "item_id": "test_item",
                    "item_price": 100.0,
                    "item_base_price": 100.0,
                    "item_discount_pct": 0.0,
                    "item_category": "Test"
                }],
                "return_probabilities": True
            })
            assert response.status_code == 503
    
    def test_predict_invalid_request(self, client):
        """Test prediction with invalid request."""
        response = client.post("/predict", json={
            "features": [],  # Empty features list
            "return_probabilities": True
        })
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_price(self, client):
        """Test prediction with invalid price."""
        response = client.post("/predict", json={
            "features": [{
                "user_id": "test_user",
                "item_id": "test_item",
                "item_price": -10.0,  # Invalid negative price
                "item_base_price": 100.0,
                "item_discount_pct": 0.0,
                "item_category": "Test"
            }],
            "return_probabilities": True
        })
        assert response.status_code == 422  # Validation error


class TestModelManagement:
    """Test model management endpoints."""
    
    def test_model_info_no_model(self, client):
        """Test model info when no model is loaded."""
        if not model_manager.is_loaded():
            response = client.get("/model-info")
            assert response.status_code == 404
    
    def test_load_model_invalid_path(self, client):
        """Test loading model with invalid path."""
        response = client.post("/load-model", params={
            "model_path": "/nonexistent/model.pkl"
        })
        assert response.status_code == 404


class TestValidation:
    """Test input validation."""
    
    def test_batch_size_limit(self, client):
        """Test batch size limit."""
        # Create request with too many features
        features = [{
            "user_id": f"user_{i}",
            "item_id": f"item_{i}",
            "item_price": 100.0,
            "item_base_price": 100.0,
            "item_discount_pct": 0.0,
            "item_category": "Test"
        } for i in range(1001)]  # Exceeds max of 1000
        
        response = client.post("/predict", json={
            "features": features,
            "return_probabilities": True
        })
        assert response.status_code == 422  # Validation error
    
    def test_threshold_validation(self, client):
        """Test threshold validation."""
        # Test invalid threshold > 1
        response = client.post("/predict", json={
            "features": [{
                "user_id": "test_user",
                "item_id": "test_item",
                "item_price": 100.0,
                "item_base_price": 100.0,
                "item_discount_pct": 0.0,
                "item_category": "Test"
            }],
            "return_probabilities": True,
            "threshold": 1.5  # Invalid
        })
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
