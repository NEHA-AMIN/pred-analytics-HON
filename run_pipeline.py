#!/usr/bin/env python3
"""
End-to-End ML Pipeline Orchestrator

This script orchestrates the complete machine learning pipeline from data generation
to model training. It connects all the isolated scripts in src/ in the correct order.

Usage:
    # Run full pipeline
    python run_pipeline.py --full

    # Run specific stages
    python run_pipeline.py --data-gen --sessionize
    python run_pipeline.py --features --train

    # Skip data generation (use existing data)
    python run_pipeline.py --skip-data-gen --full

Pipeline Stages:
    1. Data Generation (users, catalog, clickstream)
    2. Sessionization (time-based session grouping)
    3. Temporal Split (train/val/test with no leakage)
    4. Feature Engineering (point-in-time features)
    5. Model Training (XGBoost with imbalance handling)
"""

import argparse
import logging
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Import pipeline components
from src.data.catalog import ItemCatalogGenerator
from src.data.users import UserBehaviorSimulator
from src.data.clickstream import ClickstreamGenerator
from src.data.sessionization import TimeBasedSessionizer, create_temporal_split
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the end-to-end ML pipeline.
    
    Design Principles:
    - Each stage is idempotent (can be re-run safely)
    - Intermediate results are saved to disk
    - Clear logging at each stage
    - Graceful error handling with cleanup
    """
    
    def __init__(self, config_path: str = "configs/data_generation.yaml"):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup directories
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
        self.models_dir = Path("models")
        
        self._create_directories()
        
        logger.info("=" * 80)
        logger.info("🚀 E-Commerce Purchase Prediction Pipeline")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded configuration from {self.config_path}")
        return config
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.raw_dir, self.processed_dir, self.features_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Created/verified directory structure")
    
    def stage_1_generate_data(self) -> Dict[str, Path]:
        """
        Stage 1: Generate synthetic data (users, catalog, clickstream).
        
        Returns:
            Dictionary with paths to generated files
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 STAGE 1: Data Generation")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 1.1: Generate Users
        logger.info("\n🎭 Step 1.1: Generating user profiles...")
        user_simulator = UserBehaviorSimulator(
            config=self.config['user_personas'],
            random_seed=self.config['dataset']['random_seed']
        )
        users_df = user_simulator.generate_users(
            num_users=self.config['dataset']['num_users']
        )
        users_path = self.raw_dir / "users.parquet"
        users_df.to_parquet(users_path, compression='snappy')
        logger.info(f"💾 Saved {len(users_df)} users to {users_path}")
        
        # 1.2: Generate Item Catalog
        logger.info("\n📦 Step 1.2: Generating item catalog...")
        catalog_generator = ItemCatalogGenerator(
            config=self.config['item_catalog'],
            random_seed=self.config['dataset']['random_seed']
        )
        catalog_df = catalog_generator.generate_catalog()
        catalog_path = self.raw_dir / "item_catalog.parquet"
        catalog_df.to_parquet(catalog_path, compression='snappy')
        logger.info(f"💾 Saved {len(catalog_df)} items to {catalog_path}")
        
        # 1.3: Generate Clickstream Events
        logger.info("\n🖱️  Step 1.3: Generating clickstream events...")
        clickstream_generator = ClickstreamGenerator(
            users_df=users_df,
            catalog_df=catalog_df,
            config=self.config,
            random_seed=self.config['dataset']['random_seed']
        )
        clickstream_df = clickstream_generator.generate_clickstream(
            start_date=self.config['dataset']['start_date'],
            num_days=self.config['dataset']['num_days']
        )
        clickstream_path = self.raw_dir / "clickstream_events.parquet"
        clickstream_df.to_parquet(clickstream_path, compression='snappy')
        logger.info(f"💾 Saved {len(clickstream_df)} events to {clickstream_path}")
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Stage 1 completed in {elapsed:.2f} seconds")
        
        return {
            'users': users_path,
            'catalog': catalog_path,
            'clickstream': clickstream_path
        }
    
    def stage_2_sessionize(self, clickstream_path: Optional[Path] = None) -> Path:
        """
        Stage 2: Sessionize clickstream events.
        
        Args:
            clickstream_path: Path to clickstream events (if None, loads from default)
            
        Returns:
            Path to sessionized events
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔗 STAGE 2: Sessionization")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load clickstream events
        if clickstream_path is None:
            clickstream_path = self.raw_dir / "clickstream_events.parquet"
        
        logger.info(f"📂 Loading clickstream from {clickstream_path}")
        events_df = pd.read_parquet(clickstream_path)
        logger.info(f"   Loaded {len(events_df)} events")
        
        # Sessionize using time-based logic
        logger.info("\n⏱️  Applying time-based sessionization...")
        sessionizer = TimeBasedSessionizer(
            inactivity_threshold_minutes=self.config['session']['inactivity_threshold_minutes']
        )
        sessionized_df = sessionizer.sessionize(events_df)
        
        # Save sessionized events
        sessionized_path = self.processed_dir / "sessionized_events.parquet"
        sessionized_df.to_parquet(sessionized_path, compression='snappy')
        logger.info(f"💾 Saved sessionized events to {sessionized_path}")
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Stage 2 completed in {elapsed:.2f} seconds")
        
        return sessionized_path
    
    def stage_3_temporal_split(self, sessionized_path: Optional[Path] = None) -> Dict[str, Path]:
        """
        Stage 3: Create temporal train/val/test splits.
        
        Args:
            sessionized_path: Path to sessionized events (if None, loads from default)
            
        Returns:
            Dictionary with paths to train/val/test splits
        """
        logger.info("\n" + "=" * 80)
        logger.info("📅 STAGE 3: Temporal Split (No Data Leakage)")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load sessionized events
        if sessionized_path is None:
            sessionized_path = self.processed_dir / "sessionized_events.parquet"
        
        logger.info(f"📂 Loading sessionized events from {sessionized_path}")
        events_df = pd.read_parquet(sessionized_path)
        
        # Calculate split dates based on dataset configuration
        start_date = pd.Timestamp(self.config['dataset']['start_date'])
        num_days = self.config['dataset']['num_days']
        
        # Use 70/15/15 split
        train_days = int(num_days * 0.70)
        val_days = int(num_days * 0.15)
        
        train_end = start_date + pd.Timedelta(days=train_days)
        val_end = train_end + pd.Timedelta(days=val_days)
        test_end = start_date + pd.Timedelta(days=num_days)
        
        logger.info(f"\n📊 Split configuration:")
        logger.info(f"   Train: {start_date.date()} to {train_end.date()} ({train_days} days)")
        logger.info(f"   Val:   {train_end.date()} to {val_end.date()} ({val_days} days)")
        logger.info(f"   Test:  {val_end.date()} to {test_end.date()} ({num_days - train_days - val_days} days)")
        
        # Create temporal split
        train_df, val_df, test_df = create_temporal_split(
            events_df=events_df,
            train_end_date=train_end.strftime('%Y-%m-%d'),
            val_end_date=val_end.strftime('%Y-%m-%d'),
            test_end_date=test_end.strftime('%Y-%m-%d')
        )
        
        # Save splits
        train_path = self.processed_dir / "train_events.parquet"
        val_path = self.processed_dir / "val_events.parquet"
        test_path = self.processed_dir / "test_events.parquet"
        
        train_df.to_parquet(train_path, compression='snappy')
        val_df.to_parquet(val_path, compression='snappy')
        test_df.to_parquet(test_path, compression='snappy')
        
        logger.info(f"\n💾 Saved splits:")
        logger.info(f"   Train: {train_path} ({len(train_df)} events)")
        logger.info(f"   Val:   {val_path} ({len(val_df)} events)")
        logger.info(f"   Test:  {test_path} ({len(test_df)} events)")
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Stage 3 completed in {elapsed:.2f} seconds")
        
        return {
            'train': train_path,
            'val': val_path,
            'test': test_path
        }
    
    def stage_4_feature_engineering(
        self,
        split_paths: Optional[Dict[str, Path]] = None,
        catalog_path: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Stage 4: Engineer features with point-in-time correctness.
        
        Args:
            split_paths: Dictionary with train/val/test event paths
            catalog_path: Path to item catalog
            
        Returns:
            Dictionary with paths to feature sets
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔧 STAGE 4: Feature Engineering")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load data
        if split_paths is None:
            split_paths = {
                'train': self.processed_dir / "train_events.parquet",
                'val': self.processed_dir / "val_events.parquet",
                'test': self.processed_dir / "test_events.parquet"
            }
        
        if catalog_path is None:
            catalog_path = self.raw_dir / "item_catalog.parquet"
        
        logger.info(f"📂 Loading catalog from {catalog_path}")
        catalog_df = pd.read_parquet(catalog_path)
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(config_path='configs/feature_engineering.yaml')
        
        feature_paths = {}
        
        # Process each split
        for split_name, events_path in split_paths.items():
            logger.info(f"\n🔨 Processing {split_name.upper()} split...")
            logger.info(f"   Loading events from {events_path}")
            
            events_df = pd.read_parquet(events_path)
            logger.info(f"   Loaded {len(events_df)} events")
            
            # Create training examples with features
            logger.info(f"   Extracting features...")
            features_df = feature_engineer.create_training_examples(
                events_df=events_df,
                catalog_df=catalog_df
            )
            
            # Save features
            feature_path = self.features_dir / f"{split_name}_features.parquet"
            features_df.to_parquet(feature_path, compression='snappy')
            feature_paths[split_name] = feature_path
            
            logger.info(f"   💾 Saved {len(features_df)} examples to {feature_path}")
            logger.info(f"   Features: {len(features_df.columns)} columns")
            logger.info(f"   Positive rate: {features_df['label'].mean():.2%}")
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Stage 4 completed in {elapsed:.2f} seconds")
        
        return feature_paths
    
    def stage_5_train_models(self, feature_paths: Optional[Dict[str, Path]] = None) -> Path:
        """
        Stage 5: Train models with imbalance handling.
        
        Args:
            feature_paths: Dictionary with train/val/test feature paths
            
        Returns:
            Path to saved model
        """
        logger.info("\n" + "=" * 80)
        logger.info("🤖 STAGE 5: Model Training")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load feature sets
        if feature_paths is None:
            feature_paths = {
                'train': self.features_dir / "train_features.parquet",
                'val': self.features_dir / "val_features.parquet",
                'test': self.features_dir / "test_features.parquet"
            }
        
        logger.info("📂 Loading feature sets...")
        train_df = pd.read_parquet(feature_paths['train'])
        val_df = pd.read_parquet(feature_paths['val'])
        test_df = pd.read_parquet(feature_paths['test'])
        
        logger.info(f"   Train: {len(train_df)} examples")
        logger.info(f"   Val:   {len(val_df)} examples")
        logger.info(f"   Test:  {len(test_df)} examples")
        
        # Initialize trainer
        trainer = ModelTrainer(config_path='configs/model_training.yaml')
        
        # Load data into trainer
        trainer.load_data(
            train_path=str(feature_paths['train']),
            val_path=str(feature_paths['val']),
            test_path=str(feature_paths['test'])
        )
        
        # Prepare features
        logger.info("\n🔧 Preparing features...")
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_features(
            train_df, val_df, test_df
        )
        
        # Train baseline model
        logger.info("\n📊 Training baseline (Logistic Regression)...")
        trainer.train_baseline(X_train, y_train, X_val, y_val)
        
        # Train XGBoost model
        logger.info("\n🚀 Training XGBoost model...")
        trainer.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        logger.info("\n📈 Evaluating on test set...")
        trainer._evaluate_model_xgb(
            trainer.model.DMatrix(X_test, label=y_test),
            y_test,
            'test'
        )
        
        # Save model
        logger.info("\n💾 Saving model...")
        model_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trainer.save_model(self.models_dir, model_name)
        model_path = self.models_dir / f"{model_name}.pkl"
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Stage 5 completed in {elapsed:.2f} seconds")
        
        return model_path
    
    def run_full_pipeline(self, skip_data_gen: bool = False):
        """
        Run the complete end-to-end pipeline.
        
        Args:
            skip_data_gen: If True, skip data generation and use existing data
        """
        pipeline_start = time.time()
        
        try:
            # Stage 1: Data Generation
            if not skip_data_gen:
                data_paths = self.stage_1_generate_data()
                clickstream_path = data_paths['clickstream']
            else:
                logger.info("\n⏭️  Skipping data generation (using existing data)")
                clickstream_path = None
            
            # Stage 2: Sessionization
            sessionized_path = self.stage_2_sessionize(clickstream_path)
            
            # Stage 3: Temporal Split
            split_paths = self.stage_3_temporal_split(sessionized_path)
            
            # Stage 4: Feature Engineering
            feature_paths = self.stage_4_feature_engineering(split_paths)
            
            # Stage 5: Model Training
            model_path = self.stage_5_train_models(feature_paths)
            
            # Pipeline summary
            total_elapsed = time.time() - pipeline_start
            logger.info("\n" + "=" * 80)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed with error: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point for pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description='E-Commerce Purchase Prediction Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py --full

  # Run specific stages
  python run_pipeline.py --data-gen --sessionize
  python run_pipeline.py --features --train

  # Skip data generation (use existing data)
  python run_pipeline.py --skip-data-gen --full

  # Use custom config
  python run_pipeline.py --config configs/custom.yaml --full
        """
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_generation.yaml',
        help='Path to configuration file (default: configs/data_generation.yaml)'
    )
    
    # Pipeline stages
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--data-gen', action='store_true', help='Run data generation stage')
    parser.add_argument('--sessionize', action='store_true', help='Run sessionization stage')
    parser.add_argument('--split', action='store_true', help='Run temporal split stage')
    parser.add_argument('--features', action='store_true', help='Run feature engineering stage')
    parser.add_argument('--train', action='store_true', help='Run model training stage')
    
    # Options
    parser.add_argument(
        '--skip-data-gen',
        action='store_true',
        help='Skip data generation (use existing data)'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config_path=args.config)
    
    # Run pipeline based on arguments
    if args.full:
        orchestrator.run_full_pipeline(skip_data_gen=args.skip_data_gen)
    else:
        # Run individual stages
        if args.data_gen:
            orchestrator.stage_1_generate_data()
        
        if args.sessionize:
            orchestrator.stage_2_sessionize()
        
        if args.split:
            orchestrator.stage_3_temporal_split()
        
        if args.features:
            orchestrator.stage_4_feature_engineering()
        
        if args.train:
            orchestrator.stage_5_train_models()
        
        # If no stages specified, show help
        if not any([args.data_gen, args.sessionize, args.split, args.features, args.train]):
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
