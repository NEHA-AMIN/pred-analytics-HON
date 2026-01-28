"""
Main script to generate complete synthetic e-commerce dataset.

Usage:
    python scripts/generate_data.py
"""

import yaml
import pandas as pd
from pathlib import Path

from src.data.catalog import ItemCatalogGenerator
from src.data.users import UserBehaviorSimulator
from src.data.clickstream import ClickstreamGenerator


def main():
    """Generate complete synthetic dataset."""
    
    # Load configuration
    config_path = Path('configs/data_generation.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['dataset']
    random_seed = dataset_config['random_seed']
    
    print("=" * 70)
    print("🎲 E-COMMERCE SYNTHETIC DATA GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Users: {dataset_config['num_users']:,}")
    print(f"  Items: {dataset_config['num_items']:,}")
    print(f"  Time period: {dataset_config['start_date']} to "
          f"{dataset_config['num_days']} days")
    print(f"  Random seed: {random_seed}")
    print()
    
    # Create output directory
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate item catalog
    print("\n" + "=" * 70)
    print("STEP 1: Generating Item Catalog")
    print("=" * 70)
    
    catalog_generator = ItemCatalogGenerator(
        config=config['item_catalog'],
        random_seed=random_seed
    )
    
    catalog_df = catalog_generator.generate_catalog()
    catalog_stats = catalog_generator.get_catalog_stats(catalog_df)
    
    catalog_path = output_dir / 'item_catalog.parquet'
    catalog_df.to_parquet(catalog_path, compression='snappy')
    
    print(f"\n📊 Catalog Statistics:")
    print(f"  Total items: {catalog_stats['total_items']}")
    print(f"  Avg price: ${catalog_stats['avg_price']:.2f}")
    print(f"  Items with discount: {catalog_stats['items_with_discount']} "
          f"({catalog_stats['discount_rate']:.1%})")
    
    # Step 2: Generate users
    print("\n" + "=" * 70)
    print("STEP 2: Generating User Profiles")
    print("=" * 70)
    
    user_simulator = UserBehaviorSimulator(
        config=config['user_personas'],
        random_seed=random_seed
    )
    
    users_df = user_simulator.generate_users(
        num_users=dataset_config['num_users']
    )
    
    users_path = output_dir / 'users.parquet'
    users_df.to_parquet(users_path, compression='snappy')
    
    # Step 3: Generate clickstream
    print("\n" + "=" * 70)
    print("STEP 3: Generating Clickstream Events")
    print("=" * 70)
    print("⚠️  This may take 5-10 minutes for full dataset...")
    
    clickstream_generator = ClickstreamGenerator(
        users_df=users_df,
        catalog_df=catalog_df,
        config=config,
        random_seed=random_seed
    )
    
    events_df = clickstream_generator.generate_clickstream(
        start_date=dataset_config['start_date'],
        num_days=dataset_config['num_days'],
        output_path=output_dir / 'clickstream_events.parquet'
    )
    
    # Generate summary statistics
    print("\n" + "=" * 70)
    print("📊 FINAL DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Item Catalog:")
    print(f"   Path: {catalog_path}")
    print(f"   Rows: {len(catalog_df):,}")
    print(f"   Size: {catalog_path.stat().st_size / 1024:.2f} KB")
    
    print(f"\n✅ User Profiles:")
    print(f"   Path: {users_path}")
    print(f"   Rows: {len(users_df):,}")
    print(f"   Size: {users_path.stat().st_size / 1024:.2f} KB")
    
    events_path = output_dir / 'clickstream_events.parquet'
    print(f"\n✅ Clickstream Events:")
    print(f"   Path: {events_path}")
    print(f"   Rows: {len(events_df):,}")
    print(f"   Size: {events_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\n📈 Event Breakdown:")
    event_counts = events_df['event_type'].value_counts()
    for event_type, count in event_counts.items():
        pct = count / len(events_df) * 100
        print(f"   {event_type:15s}: {count:8,} ({pct:5.2f}%)")
    
    print(f"\n🎯 Key Metrics:")
    print(f"   Total sessions: {events_df['session_id'].nunique():,}")
    print(f"   Avg events/session: {len(events_df) / events_df['session_id'].nunique():.1f}")
    print(f"   Purchase rate: {(events_df['event_type'] == 'purchase').mean():.2%}")
    
    # Sessions with purchases
    purchase_sessions = events_df[events_df['event_type'] == 'purchase']['session_id'].unique()
    total_sessions = events_df['session_id'].nunique()
    session_conversion = len(purchase_sessions) / total_sessions
    print(f"   Session conversion: {session_conversion:.2%}")
    
    print("\n" + "=" * 70)
    print("✅ DATA GENERATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run: python src/data/sessionization.py (Phase 2)")
    print("  2. Explore data in notebooks/")
    print()


if __name__ == "__main__":
    main()