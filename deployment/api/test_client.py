"""
Test client for the Purchase Prediction API

Demonstrates how to interact with the API and validate responses.
"""

import requests
import time
from typing import Dict, List


class PredictionAPIClient:
    """Client for interacting with the Purchase Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """
        Check API health status.
        
        Returns:
            Health check response
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_single(
        self,
        user_id: str,
        item_id: str,
        features: Dict,
        threshold: float = 0.5
    ) -> Dict:
        """
        Make a single prediction.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            features: Feature dictionary
            threshold: Classification threshold
            
        Returns:
            Prediction response
        """
        payload = {
            "features": [{
                "user_id": user_id,
                "item_id": item_id,
                **features
            }],
            "return_probabilities": True,
            "threshold": threshold
        }
        
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_batch(
        self,
        predictions: List[Dict],
        threshold: float = 0.5
    ) -> Dict:
        """
        Make batch predictions.
        
        Args:
            predictions: List of feature dictionaries with user_id and item_id
            threshold: Classification threshold
            
        Returns:
            Batch prediction response
        """
        payload = {
            "features": predictions,
            "return_probabilities": True,
            "threshold": threshold
        }
        
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Model information
        """
        response = self.session.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_path: str) -> Dict:
        """
        Load a specific model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Load status response
        """
        response = self.session.post(
            f"{self.base_url}/load-model",
            params={"model_path": model_path}
        )
        response.raise_for_status()
        return response.json()


def main():
    """Test the API with sample requests."""
    
    print("=" * 80)
    print("🧪 Testing Purchase Prediction API")
    print("=" * 80)
    
    # Initialize client
    client = PredictionAPIClient()
    
    # 1. Health Check
    print("\n1️⃣  Health Check")
    print("-" * 40)
    try:
        health = client.health_check()
        print(f"✅ Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        if health['model_loaded']:
            print(f"   Model version: {health['model_version']}")
        print(f"   Uptime: {health['uptime_seconds']:.2f}s")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {str(e)}")
        return
    
    # Check if model is loaded
    if not health['model_loaded']:
        print("\n⚠️  No model loaded. Please train a model first.")
        print("   Run: python run_pipeline.py --full")
        return
    
    # 2. Model Info
    print("\n2️⃣  Model Information")
    print("-" * 40)
    try:
        model_info = client.get_model_info()
        print(f"✅ Model version: {model_info['model_version']}")
        print(f"   Model type: {model_info['model_type']}")
        print(f"   Features: {model_info.get('num_features', 'N/A')}")
        print(f"   Loaded at: {model_info['loaded_at']}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get model info: {str(e)}")
    
    # 3. Single Prediction
    print("\n3️⃣  Single Prediction")
    print("-" * 40)
    
    # Example features for a high-intent user
    high_intent_features = {
        "days_since_last_view": 1.0,
        "days_since_last_item_view": 2.0,
        "days_since_last_cart": 3.0,
        "view_count_7d": 20,
        "view_count_30d": 50,
        "cart_count_7d": 5,
        "cart_count_30d": 10,
        "purchase_count_7d": 1,
        "purchase_count_30d": 3,
        "cart_to_purchase_ratio": 0.3,
        "avg_session_duration": 15.0,
        "session_item_count": 10,
        "session_duration_minutes": 20.0,
        "item_view_count": 5,
        "item_cart_count": 2,
        "item_price": 79.99,
        "item_base_price": 129.99,
        "item_discount_pct": 38.5,
        "item_category": "Electronics"
    }
    
    try:
        start = time.time()
        result = client.predict_single(
            user_id="user_test_001",
            item_id="item_test_001",
            features=high_intent_features,
            threshold=0.5
        )
        latency = (time.time() - start) * 1000
        
        pred = result['predictions'][0]
        print(f"✅ User: {pred['user_id']}")
        print(f"   Item: {pred['item_id']}")
        print(f"   Prediction: {'WILL PURCHASE' if pred['prediction'] == 1 else 'NO PURCHASE'}")
        print(f"   Probability: {pred['probability']:.2%}")
        print(f"   Confidence: {pred['confidence']}")
        print(f"   Latency: {latency:.2f}ms (API: {result['latency_ms']:.2f}ms)")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction failed: {str(e)}")
    
    # 4. Batch Prediction
    print("\n4️⃣  Batch Prediction")
    print("-" * 40)
    
    # Create batch of predictions with varying intent levels
    batch_features = []
    
    # High intent user
    batch_features.append({
        "user_id": "user_batch_001",
        "item_id": "item_batch_001",
        "view_count_7d": 25,
        "cart_count_7d": 5,
        "item_price": 89.99,
        "item_base_price": 129.99,
        "item_discount_pct": 30.0,
        "item_category": "Electronics"
    })
    
    # Medium intent user
    batch_features.append({
        "user_id": "user_batch_002",
        "item_id": "item_batch_002",
        "view_count_7d": 10,
        "cart_count_7d": 1,
        "item_price": 49.99,
        "item_base_price": 59.99,
        "item_discount_pct": 16.7,
        "item_category": "Clothing"
    })
    
    # Low intent user
    batch_features.append({
        "user_id": "user_batch_003",
        "item_id": "item_batch_003",
        "view_count_7d": 2,
        "cart_count_7d": 0,
        "item_price": 199.99,
        "item_base_price": 199.99,
        "item_discount_pct": 0.0,
        "item_category": "Home & Kitchen"
    })
    
    try:
        start = time.time()
        result = client.predict_batch(batch_features, threshold=0.5)
        latency = (time.time() - start) * 1000
        
        print(f"✅ Processed {len(result['predictions'])} predictions")
        print(f"   Total latency: {latency:.2f}ms")
        print(f"   Avg per item: {latency/len(result['predictions']):.2f}ms")
        print()
        
        for i, pred in enumerate(result['predictions'], 1):
            print(f"   [{i}] {pred['user_id']} + {pred['item_id']}")
            print(f"       → {'PURCHASE' if pred['prediction'] == 1 else 'NO PURCHASE'} "
                  f"(p={pred['probability']:.2%}, {pred['confidence']} confidence)")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction failed: {str(e)}")
    
    # 5. Threshold Testing
    print("\n5️⃣  Threshold Testing")
    print("-" * 40)
    
    thresholds = [0.3, 0.5, 0.7]
    
    try:
        for threshold in thresholds:
            result = client.predict_single(
                user_id="user_threshold_test",
                item_id="item_threshold_test",
                features=high_intent_features,
                threshold=threshold
            )
            pred = result['predictions'][0]
            print(f"   Threshold {threshold:.1f}: "
                  f"{'PURCHASE' if pred['prediction'] == 1 else 'NO PURCHASE'} "
                  f"(p={pred['probability']:.2%})")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Threshold testing failed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("✅ API Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
