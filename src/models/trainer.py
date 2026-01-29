"""
Model Training Pipeline

Trains models with extreme class imbalance handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, Tuple
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available - skipping experiment tracking")


class ModelTrainer:
    """
    Trains models for extreme imbalance scenarios.
    
    Key strategies:
    - Class weighting (scale_pos_weight)
    - Precision-focused metrics
    - Threshold tuning
    """
    
    def __init__(self, config_path: str = 'configs/model_training.yaml'):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = None
        self.preprocessor = None
        self.model = None
        self.best_threshold = 0.5
        
    def load_data(
        self,
        train_path: str,
        val_path: str,
        test_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/val/test feature sets."""
        print("📂 Loading feature sets...")
        
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
        
        print(f"   Train: {len(train_df):,} examples ({train_df['label'].mean():.4%} positive)")
        print(f"   Val:   {len(val_df):,} examples ({val_df['label'].mean():.4%} positive)")
        print(f"   Test:  {len(test_df):,} examples ({test_df['label'].mean():.4%} positive)")
        
        return train_df, val_df, test_df
    
    def prepare_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple:
        """
        Prepare features for training.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\n🔧 Preparing features...")
        
        # Get feature config
        drop_cols = self.config['features']['drop_features']
        target_col = self.config['features']['target_column']
        
        # Separate features and labels
        X_train = train_df.drop(columns=drop_cols + [target_col])
        X_val = val_df.drop(columns=drop_cols + [target_col])
        X_test = test_df.drop(columns=drop_cols + [target_col])
        
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        y_test = test_df[target_col]
        
        # Handle categorical features (item_category)
        categorical_features = ['item_category']
        numeric_features = [col for col in X_train.columns if col not in categorical_features]
        
        print(f"   Numeric features: {len(numeric_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        
        # Fit and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        print(f"   Final feature dimension: {X_train_processed.shape[1]}")
        
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
    
    def train_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """Train baseline Logistic Regression."""
        print("\n" + "="*70)
        print("TRAINING BASELINE: Logistic Regression")
        print("="*70)
        
        config = self.config['baseline']
        
        # Train model
        self.model = LogisticRegression(
            class_weight=config['class_weight'],
            max_iter=config['max_iter'],
            random_state=config['random_state']
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self._evaluate_model(X_val, y_val, "Validation")
        
        return metrics
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """Train XGBoost with extreme imbalance handling."""
        print("\n" + "="*70)
        print("TRAINING XGBoost (Extreme Imbalance Mode)")
        print("="*70)
        
        config = self.config['xgboost']
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Training parameters
        params = {
            'objective': config['objective'],
            'eval_metric': config['eval_metric'],
            'max_depth': config['max_depth'],
            'learning_rate': config['learning_rate'],
            'min_child_weight': config['min_child_weight'],
            'gamma': config['gamma'],
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample_bytree'],
            'reg_alpha': config['reg_alpha'],
            'reg_lambda': config['reg_lambda'],
            'scale_pos_weight': config['scale_pos_weight'],
            'tree_method': config['tree_method'],
            'random_state': config['random_state']
        }
        
        print(f"\n⚙️  Key parameters:")
        print(f"   scale_pos_weight: {params['scale_pos_weight']} (handling 0.01% positive rate)")
        print(f"   learning_rate: {params['learning_rate']}")
        print(f"   max_depth: {params['max_depth']}")
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=config['n_estimators'],
            evals=evals,
            early_stopping_rounds=config['early_stopping_rounds'],
            verbose_eval=20
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Best iteration: {self.model.best_iteration}")
        
        # Evaluate
        metrics = self._evaluate_model_xgb(dval, y_val, "Validation")
        
        return metrics
    
    def _evaluate_model(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        split_name: str
    ) -> Dict:
        """Evaluate sklearn-style model."""
        # Get predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        return self._compute_metrics(y_true, y_pred_proba, split_name)
    
    def _evaluate_model_xgb(
        self,
        dmatrix: xgb.DMatrix,
        y_true: np.ndarray,
        split_name: str
    ) -> Dict:
        """Evaluate XGBoost model."""
        # Get predictions
        y_pred_proba = self.model.predict(dmatrix)
        
        return self._compute_metrics(y_true, y_pred_proba, split_name)
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        split_name: str
    ) -> Dict:
        """Compute comprehensive metrics for extreme imbalance."""
        print(f"\n📊 {split_name} Set Metrics:")
        print("-" * 70)
        
        metrics = {}
        
        # 1. PR-AUC (Primary metric for imbalanced data)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        metrics['pr_auc'] = pr_auc
        print(f"   PR-AUC: {pr_auc:.4f} {'🎯' if pr_auc > 0.01 else '⚠️'}")
        
        # 2. ROC-AUC (Less meaningful but included for completeness)
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            metrics['roc_auc'] = roc_auc
            print(f"   ROC-AUC: {roc_auc:.4f}")
        
        # 3. Precision @ K (Most important for extreme imbalance)
        for k_pct in [1, 5, 10]:
            k = int(len(y_true) * k_pct / 100)
            top_k_indices = np.argsort(y_pred_proba)[-k:]
            precision_at_k = y_true.iloc[top_k_indices].mean() if hasattr(y_true, 'iloc') else y_true[top_k_indices].mean()
            metrics[f'precision_at_{k_pct}_pct'] = precision_at_k
            print(f"   Precision@{k_pct}%: {precision_at_k:.4f} ({int(precision_at_k * k)}/{k} positives)")
        
        # 4. Prediction distribution
        print(f"\n   Prediction score distribution:")
        print(f"     Min:  {y_pred_proba.min():.6f}")
        print(f"     25%:  {np.percentile(y_pred_proba, 25):.6f}")
        print(f"     50%:  {np.percentile(y_pred_proba, 50):.6f}")
        print(f"     75%:  {np.percentile(y_pred_proba, 75):.6f}")
        print(f"     Max:  {y_pred_proba.max():.6f}")
        
        # 5. Baseline comparison
        baseline_precision = y_true.mean()
        print(f"\n   Baseline (random): {baseline_precision:.4f}")
        print(f"   Lift @ 1%: {metrics['precision_at_1_pct'] / baseline_precision:.2f}x")
        
        return metrics
    
    def save_model(self, output_dir: Path, model_name: str):
        """Save trained model and preprocessor."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor
        preprocessor_path = output_dir / f'{model_name}_preprocessor.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save model
        if isinstance(self.model, xgb.Booster):
            model_path = output_dir / f'{model_name}_model.json'
            self.model.save_model(str(model_path))
        else:
            model_path = output_dir / f'{model_name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        print(f"\n💾 Saved model to: {model_path}")
        print(f"💾 Saved preprocessor to: {preprocessor_path}")


def main():
    """Train models on full dataset."""
    print("=" * 70)
    print("MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    train_df, val_df, test_df = trainer.load_data(
        train_path='data/processed/train_features.parquet',
        val_path='data/processed/val_features.parquet',
        test_path='data/processed/test_features.parquet'
    )
    
    # Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_features(
        train_df, val_df, test_df
    )
    
    # Train baseline
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Baseline Model")
    print("=" * 70)
    baseline_metrics = trainer.train_baseline(X_train, y_train, X_val, y_val)
    
    # Save baseline
    trainer.save_model(Path('models'), 'baseline_logreg')
    
    # Train XGBoost
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: XGBoost")
    print("=" * 70)
    xgb_metrics = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    
    # Save XGBoost
    trainer.save_model(Path('models'), 'xgboost_v1')
    
    # Compare models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print("\nValidation Set Performance:")
    print(f"{'Metric':<25} {'Baseline':<15} {'XGBoost':<15}")
    print("-" * 55)
    for metric in ['pr_auc', 'precision_at_1_pct', 'precision_at_5_pct']:
        baseline_val = baseline_metrics.get(metric, 0)
        xgb_val = xgb_metrics.get(metric, 0)
        print(f"{metric:<25} {baseline_val:<15.4f} {xgb_val:<15.4f}")
    
    print("\n✅ Training complete!")
    print("\nNext steps:")
    print("   1. Analyze feature importance")
    print("   2. Tune hyperparameters")
    print("   3. Evaluate on test set")


if __name__ == "__main__":
    main()