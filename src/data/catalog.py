"""
Item Catalog Generator

Generates realistic e-commerce item catalog with categories, pricing, and discounts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Category:
    """Represents an item category with pricing rules."""
    
    name: str
    num_items: int
    price_range: List[float]
    discount_probability: float
    discount_range: List[int]


class ItemCatalogGenerator:
    """
    Generates synthetic item catalog for e-commerce simulation.
    
    Design Principles:
    - Realistic price distributions (log-normal)
    - Category-specific pricing strategies
    - Random but reproducible (seeded)
    """
    
    def __init__(self, config: Dict, random_seed: int = 42):
        """
        Initialize catalog generator.
        
        Args:
            config: Category configuration from YAML
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
    def generate_catalog(self) -> pd.DataFrame:
        """
        Generate complete item catalog.
        
        Returns:
            DataFrame with columns:
            - item_id: Unique identifier
            - category: Category name
            - base_price: Original price
            - discount_pct: Discount percentage (0-100)
            - current_price: Price after discount
            - created_at: When item was added to catalog
        """
        all_items = []
        item_id_counter = 0
        
        for category_config in self.config['categories']:
            category = Category(**category_config)
            
            # Generate items for this category
            category_items = self._generate_category_items(
                category=category,
                start_item_id=item_id_counter
            )
            
            all_items.append(category_items)
            item_id_counter += category.num_items
            
        # Combine all categories
        catalog_df = pd.concat(all_items, ignore_index=True)
        
        print(f"✅ Generated catalog: {len(catalog_df)} items across "
              f"{len(self.config['categories'])} categories")
        
        return catalog_df
    
    def _generate_category_items(
        self, 
        category: Category, 
        start_item_id: int
    ) -> pd.DataFrame:
        """
        Generate items for a single category.
        
        Args:
            category: Category configuration
            start_item_id: Starting ID for items in this category
            
        Returns:
            DataFrame of items in this category
        """
        num_items = category.num_items
        
        # Generate item IDs
        item_ids = [f"item_{start_item_id + i:05d}" for i in range(num_items)]
        
        # Generate base prices (log-normal distribution for realism)
        min_price, max_price = category.price_range
        
        # Log-normal parameters
        log_min = np.log(min_price)
        log_max = np.log(max_price)
        log_mean = (log_min + log_max) / 2
        log_std = (log_max - log_min) / 4  # Most items within range
        
        base_prices = self.rng.lognormal(log_mean, log_std, num_items)
        base_prices = np.clip(base_prices, min_price, max_price)
        base_prices = np.round(base_prices, 2)
        
        # Generate discounts
        has_discount = self.rng.random(num_items) < category.discount_probability
        discount_pcts = np.zeros(num_items)
        
        if has_discount.any():
            min_disc, max_disc = category.discount_range
            discount_pcts[has_discount] = self.rng.uniform(
                min_disc, max_disc, has_discount.sum()
            )
            discount_pcts = np.round(discount_pcts, 0)
        
        # Calculate current prices
        current_prices = base_prices * (1 - discount_pcts / 100)
        current_prices = np.round(current_prices, 2)
        
        # Generate creation dates (items added over time)
        # Older items are less popular (will use this later for recommendations)
        days_ago = self.rng.exponential(scale=180, size=num_items)  # ~6 months avg
        days_ago = np.clip(days_ago, 0, 365 * 2)  # Max 2 years old
        
        created_dates = pd.Timestamp.now() - pd.to_timedelta(days_ago, unit='D')
        
        # Build dataframe
        items_df = pd.DataFrame({
            'item_id': item_ids,
            'category': category.name,
            'base_price': base_prices,
            'discount_pct': discount_pcts,
            'current_price': current_prices,
            'created_at': created_dates
        })
        
        return items_df
    
    def get_catalog_stats(self, catalog_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the catalog.
        
        Args:
            catalog_df: Generated catalog
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_items': len(catalog_df),
            'categories': catalog_df['category'].nunique(),
            'avg_price': catalog_df['current_price'].mean(),
            'median_price': catalog_df['current_price'].median(),
            'price_range': [
                catalog_df['current_price'].min(),
                catalog_df['current_price'].max()
            ],
            'items_with_discount': (catalog_df['discount_pct'] > 0).sum(),
            'discount_rate': (catalog_df['discount_pct'] > 0).mean(),
            'category_distribution': catalog_df['category'].value_counts().to_dict()
        }
        
        return stats


def main():
    """Test the catalog generator."""
    import yaml
    
    # Load config
    with open('configs/data_generation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate catalog
    generator = ItemCatalogGenerator(
        config=config['item_catalog'],
        random_seed=config['dataset']['random_seed']
    )
    
    catalog_df = generator.generate_catalog()
    
    # Show stats
    stats = generator.get_catalog_stats(catalog_df)
    print("\n📊 Catalog Statistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Avg price: ${stats['avg_price']:.2f}")
    print(f"  Median price: ${stats['median_price']:.2f}")
    print(f"  Price range: ${stats['price_range'][0]:.2f} - ${stats['price_range'][1]:.2f}")
    print(f"  Items with discount: {stats['items_with_discount']} "
          f"({stats['discount_rate']:.1%})")
    print(f"\n  Category distribution:")
    for cat, count in stats['category_distribution'].items():
        print(f"    {cat}: {count} items")
    
    # Show sample
    print(f"\n📦 Sample items:")
    print(catalog_df.head(10).to_string(index=False))
    
    # Save to file
    output_path = 'data/raw/item_catalog.parquet'
    catalog_df.to_parquet(output_path, compression='snappy')
    print(f"\n💾 Saved catalog to: {output_path}")


if __name__ == "__main__":
    main()