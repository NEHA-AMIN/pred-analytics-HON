"""
Final Model Evaluation on Test Set
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt
from pathlib import Path


def load_model_and_preprocessor(model_path: str, preprocessor_path: str):
    """Load trained model and preprocessor."""
    # Load preprocessor
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(model_path)
    
    return model, preprocessor


def evaluate_on_test_set():
    """Full evaluation on held-out test set."""
    print("=" * 70)
    print("FINAL EVALUATION: Test Set")
    print("=" * 70)
    
    # Load test data
    print("\n📂 Loading test data...")
    test_df = pd.read_parquet('data/processed/test_features.parquet')
    print(f"   Test examples: {len(test_df):,}")
    print(f"   Positive rate: {test_df['label'].mean():.4%}")
    print(f"   Positive examples: {test_df['label'].sum()}")
    
    # Load model
    print("\n🤖 Loading XGBoost model...")
    model, preprocessor = load_model_and_preprocessor(
        'models/xgboost_v1_model.json',
        'models/xgboost_v1_preprocessor.pkl'
    )
    
    # Prepare features
    drop_cols = ['user_id', 'item_id', 'session_id', 'session_end_time', 'label']
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['label']
    
    X_test_processed = preprocessor.transform(X_test)
    dtest = xgb.DMatrix(X_test_processed)
    
    # Get predictions
    print("\n🔮 Generating predictions...")
    y_pred_proba = model.predict(dtest)
    
    # Comprehensive metrics
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE")
    print("=" * 70)
    
    # 1. PR-AUC (primary metric)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"\n📊 PR-AUC: {pr_auc:.4f}")
    
    # 2. ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"📊 ROC-AUC: {roc_auc:.4f}")
    
    # 3. Precision @ K
    print(f"\n📊 Precision @ K:")
    for k_pct in [1, 5, 10, 20]:
        k = int(len(y_test) * k_pct / 100)
        top_k_indices = np.argsort(y_pred_proba)[-k:]
        precision_at_k = y_test.iloc[top_k_indices].mean()
        positives_found = y_test.iloc[top_k_indices].sum()
        print(f"   Top {k_pct:2d}% ({k:5d} items): "
              f"Precision = {precision_at_k:.4f} "
              f"({int(positives_found)}/{int(y_test.sum())} positives found)")
    
    # 4. Score distribution
    print(f"\n📊 Prediction Score Distribution:")
    print(f"   Min:    {y_pred_proba.min():.6f}")
    print(f"   25%:    {np.percentile(y_pred_proba, 25):.6f}")
    print(f"   Median: {np.percentile(y_pred_proba, 50):.6f}")
    print(f"   75%:    {np.percentile(y_pred_proba, 75):.6f}")
    print(f"   95%:    {np.percentile(y_pred_proba, 95):.6f}")
    print(f"   99%:    {np.percentile(y_pred_proba, 99):.6f}")
    print(f"   Max:    {y_pred_proba.max():.6f}")
    
    # 5. Analyze top predictions
    print(f"\n📊 Top 20 Predictions:")
    top_20_indices = np.argsort(y_pred_proba)[-20:][::-1]
    top_20_df = test_df.iloc[top_20_indices][
        ['user_id', 'item_id', 'item_category', 'item_current_price', 'label']
    ].copy()
    top_20_df['predicted_score'] = y_pred_proba[top_20_indices]
    print(top_20_df.to_string(index=False))
    
    # 6. Feature importance
    print(f"\n📊 Top 10 Most Important Features:")
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in importance.items()
    ]).sort_values('importance', ascending=False).head(10)
    print(importance_df.to_string(index=False))
    
    # 7. Baseline comparison
    baseline_precision = y_test.mean()
    print(f"\n📊 Baseline Comparison:")
    print(f"   Random baseline:     {baseline_precision:.4f}")
    print(f"   Model @ top 1%:      {y_test.iloc[np.argsort(y_pred_proba)[-int(len(y_test)*0.01):]].mean():.4f}")
    lift = (y_test.iloc[np.argsort(y_pred_proba)[-int(len(y_test)*0.01):]].mean() / baseline_precision) if baseline_precision > 0 else 0
    print(f"   Lift:                {lift:.1f}x")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'top_predictions': top_20_df
    }


if __name__ == "__main__":
    results = evaluate_on_test_set()