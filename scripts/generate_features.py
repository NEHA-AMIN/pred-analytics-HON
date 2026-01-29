"""
Generate features for full train/val/test sets.

This script:
1. Loads train/val/test event splits
2. Extracts features maintaining point-in-time correctness
3. Saves feature matrices for model training
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.engineering import FeatureEngineer


def generate_features_for_split(
    events_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    split_name: str,
    output_dir: Path
):
    """
    Generate features for a data split.
    
    Args:
        events_df: Events for this split
        catalog_df: Item catalog
        split_name: 'train', 'val', or 'test'
        output_dir: Where to save features
    """
    print(f"\n{'='*70}")
    print(f"GENERATING FEATURES: {split_name.upper()}")
    print(f"{'='*70}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Extract features
    features_df = engineer.create_training_examples(
        events_df=events_df,
        catalog_df=catalog_df
    )
    
    # Save features
    output_path = output_dir / f'{split_name}_features.parquet'
    features_df.to_parquet(output_path, compression='snappy')
    
    print(f"\n💾 Saved features to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Feature statistics
    print(f"\n📊 Feature Statistics:")
    print(f"   Examples: {len(features_df):,}")
    print(f"   Features: {len(features_df.columns) - 5}")
    print(f"   Positive rate: {features_df['label'].mean():.2%}")
    print(f"   Users: {features_df['user_id'].nunique():,}")
    print(f"   Items: {features_df['item_id'].nunique():,}")
    
    # Check for nulls
    null_cols = features_df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if len(null_cols) > 0:
        print(f"\n⚠️  Null values detected:")
        for col, count in null_cols.items():
            print(f"     {col}: {count} ({count/len(features_df):.2%})")
    else:
        print(f"\n✅ No null values - all features complete!")
    
    return features_df


def main():
    """Generate features for all splits."""
    
    print("=" * 70)
    print("FEATURE GENERATION PIPELINE")
    print("=" * 70)
    
    # Setup paths
    data_dir = Path('data/processed')
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load catalog (same for all splits)
    print("\n📂 Loading item catalog...")
    catalog_df = pd.read_parquet('data/raw/item_catalog.parquet')
    print(f"   Items: {len(catalog_df):,}")
    
    # Generate features for each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Load events
        events_path = data_dir / f'{split}_events.parquet'
        print(f"\n📂 Loading {split} events from {events_path}...")
        events_df = pd.read_parquet(events_path)
        print(f"   Events: {len(events_df):,}")
        print(f"   Sessions: {events_df['session_id'].nunique():,}")
        
        # Generate features
        features_df = generate_features_for_split(
            events_df=events_df,
            catalog_df=catalog_df,
            split_name=split,
            output_dir=output_dir
        )
    
    print("\n" + "=" * 70)
    print("✅ FEATURE GENERATION COMPLETE!")
    print("=" * 70)
    
    print("\nGenerated files:")
    for split in splits:
        path = output_dir / f'{split}_features.parquet'
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"   {path}: {size_mb:.2f} MB")
    
    print("\nNext steps:")
    print("   1. Explore features in notebooks/")
    print("   2. Run: python src/models/trainer.py (Phase 4)")
    print()


if __name__ == "__main__":
    main()