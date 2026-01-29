"""
Feature Engineering Pipeline

Extracts features with point-in-time correctness - NO DATA LEAKAGE.

Critical Principle:
When predicting at time T, only use data from BEFORE time T.
"""

import pandas as pd
from typing import Dict
from datetime import timedelta
import yaml


class FeatureEngineer:
    """
    Main feature engineering class.

    Generates features for (user, item, session) prediction:
    "Will user purchase this item in the NEXT session?"
    """

    def __init__(self, config_path: str = 'configs/feature_engineering.yaml'):
        """
        Initialize feature engineer.

        Args:
            config_path: Path to feature configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.windows_days = self.config['features']['frequency']['windows']

    def create_training_examples(
        self,
        events_df: pd.DataFrame,
        catalog_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create training examples from sessionized events.
        
        Logic:
        For each user's session N:
          - Items viewed in session N = candidates
          - Label: was item purchased in ANY of sessions N+1, N+2, N+3?
          - Features: computed using data UP TO END of session N
          
        Args:
            events_df: Sessionized clickstream events
            catalog_df: Item catalog with prices
            
        Returns:
            DataFrame with one row per (user, item, session) with features and labels
        """
        print("🔧 Creating training examples...")
        print(f"   Total events: {len(events_df):,}")
        print(f"   Total sessions: {events_df['session_id'].nunique():,}")
        
        # Sort events
        events_df = events_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Create examples
        examples = []
        
        # Group by user
        for user_id, user_events in events_df.groupby('user_id'):
            user_sessions = user_events.groupby('session_id')
            session_ids = list(user_sessions.groups.keys())
            
            # Need at least 4 sessions (current + 3 lookahead)
            if len(session_ids) < 4:
                continue
                
            # For each session except the last 3
            for i in range(len(session_ids) - 3):
                current_session_id = session_ids[i]
                # Look ahead 3 sessions
                next_3_session_ids = session_ids[i + 1:i + 4]
                
                current_session = user_sessions.get_group(current_session_id)
                
                # Get items viewed in current session
                viewed_items = current_session[
                    current_session['event_type'] == 'view'
                ]['item_id'].unique()
                
                # Get items purchased in ANY of the next 3 sessions
                purchased_items = set()
                for next_session_id in next_3_session_ids:
                    next_session = user_sessions.get_group(next_session_id)
                    purchased_in_session = next_session[
                        next_session['event_type'] == 'purchase'
                    ]['item_id'].unique()
                    purchased_items.update(purchased_in_session)
                
                # Get session end time (for point-in-time feature computation)
                session_end_time = current_session['timestamp'].max()
                
                # Create example for each viewed item
                for item_id in viewed_items:
                    # Label: was this item purchased in any of next 3 sessions?
                    label = int(item_id in purchased_items)
                    
                    # Extract features at session end time
                    features = self._extract_features(
                        user_id=user_id,
                        item_id=item_id,
                        current_session=current_session,
                        all_user_events=user_events,
                        session_end_time=session_end_time,
                        catalog_df=catalog_df
                    )
                    
                    # Combine into example
                    example = {
                        'user_id': user_id,
                        'item_id': item_id,
                        'session_id': current_session_id,
                        'session_end_time': session_end_time,
                        'label': label,
                        **features
                    }
                    
                    examples.append(example)
        
        examples_df = pd.DataFrame(examples)
        
        print(f"\n✅ Created {len(examples_df):,} training examples")
        print(f"   Positive examples (purchase): {examples_df['label'].sum():,} ({examples_df['label'].mean():.2%})")
        print(f"   Negative examples: {(examples_df['label'] == 0).sum():,}")
        
        return examples_df

    def _get_session_info(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Get session start/end times and metadata."""
        session_info = events_df.groupby('session_id').agg({
            'timestamp': ['min', 'max'],
            'user_id': 'first'
        })
        session_info.columns = ['start_time', 'end_time', 'user_id']
        return session_info.reset_index()

    def _extract_features(
        self,
        user_id: str,
        item_id: str,
        current_session: pd.DataFrame,
        all_user_events: pd.DataFrame,
        session_end_time: pd.Timestamp,
        catalog_df: pd.DataFrame
    ) -> Dict:
        """
        Extract all features for (user, item, session) at given point in time.

        CRITICAL: Only use events BEFORE session_end_time.
        """
        # Filter to events before current time
        historical_events = all_user_events[
            all_user_events["timestamp"] < session_end_time
        ]

        features = {}

        # 1. Recency features
        features.update(self._recency_features(
            user_id, item_id, historical_events, session_end_time
        ))

        # 2. Frequency features
        features.update(self._frequency_features(
            user_id, item_id, historical_events, session_end_time
        ))

        # 3. Intent strength features
        features.update(self._intent_features(
            user_id, historical_events, session_end_time
        ))

        # 4. Session context features
        features.update(self._session_context_features(
            current_session, catalog_df
        ))

        # 5. User-item affinity features
        features.update(self._user_item_affinity_features(
            user_id, item_id, historical_events, catalog_df
        ))

        # 6. Item features
        features.update(self._item_features(
            item_id, catalog_df
        ))

        return features

    def _recency_features(
        self,
        user_id: str,
        item_id: str,
        historical_events: pd.DataFrame,
        reference_time: pd.Timestamp
    ) -> Dict:
        """Calculate recency features (time since last event)."""
        features = {}

        event_types = ['view', 'scroll', 'add_to_cart', 'purchase']

        for event_type in event_types:
            events = historical_events[historical_events['event_type'] == event_type]

            if len(events) > 0:
                last_event_time = events['timestamp'].max()
                hours_since = (reference_time - last_event_time).total_seconds() / 3600
            else:
                hours_since = 9999  # Large value indicating "never"

            features[f'hours_since_last_{event_type}'] = hours_since

        # Days since user registration (first event)
        if len(historical_events) > 0:
            registration_time = historical_events['timestamp'].min()
            days_since_reg = (reference_time - registration_time).days
        else:
            days_since_reg = 0

        features['days_since_registration'] = days_since_reg

        return features

    def _frequency_features(
        self,
        user_id: str,
        item_id: str,
        historical_events: pd.DataFrame,
        reference_time: pd.Timestamp
    ) -> Dict:
        """Calculate frequency features across multiple time windows."""
        features = {}

        for window_days in self.windows_days:
            window_start = reference_time - timedelta(days=window_days)
            window_events = historical_events[
                historical_events['timestamp'] >= window_start
            ]

            # Event counts
            features[f'views_count_{window_days}d'] = (
                window_events['event_type'] == 'view'
            ).sum()

            features[f'scrolls_count_{window_days}d'] = (
                window_events['event_type'] == 'scroll'
            ).sum()

            features[f'cart_adds_count_{window_days}d'] = (
                window_events['event_type'] == 'add_to_cart'
            ).sum()

            features[f'purchases_count_{window_days}d'] = (
                window_events['event_type'] == 'purchase'
            ).sum()

            # Unique counts
            features[f'unique_items_viewed_{window_days}d'] = (
                window_events[window_events['event_type'] == 'view']['item_id'].nunique()
            )

        return features

    def _intent_features(
        self,
        user_id: str,
        historical_events: pd.DataFrame,
        reference_time: pd.Timestamp
    ) -> Dict:
        """Calculate user intent strength features."""
        features = {}

        # 7-day window for intent
        window_7d = reference_time - timedelta(days=7)
        recent_events = historical_events[historical_events['timestamp'] >= window_7d]

        view_count = (recent_events['event_type'] == 'view').sum()
        scroll_count = (recent_events['event_type'] == 'scroll').sum()
        cart_count = (recent_events['event_type'] == 'add_to_cart').sum()

        # Conversion ratios
        features['view_to_scroll_ratio_7d'] = (
            scroll_count / view_count if view_count > 0 else 0
        )

        features['scroll_to_cart_ratio_7d'] = (
            cart_count / scroll_count if scroll_count > 0 else 0
        )

        # 30-day cart to purchase
        window_30d = reference_time - timedelta(days=30)
        events_30d = historical_events[historical_events['timestamp'] >= window_30d]

        cart_30d = (events_30d['event_type'] == 'add_to_cart').sum()
        purchase_30d = (events_30d['event_type'] == 'purchase').sum()

        features['cart_to_purchase_ratio_30d'] = (
            purchase_30d / cart_30d if cart_30d > 0 else 0
        )

        return features

    def _session_context_features(
        self,
        current_session: pd.DataFrame,
        catalog_df: pd.DataFrame
    ) -> Dict:
        """Features about the current session context."""
        features = {}

        # Session duration
        duration = (
            current_session['timestamp'].max() -
            current_session['timestamp'].min()
        ).total_seconds() / 60
        features['session_length_minutes'] = duration

        # Event counts in session
        features['items_viewed_session'] = (
            current_session['event_type'] == 'view'
        ).sum()

        features['items_scrolled_session'] = (
            current_session['event_type'] == 'scroll'
        ).sum()

        features['items_carted_session'] = (
            current_session['event_type'] == 'add_to_cart'
        ).sum()

        # Unique items/categories
        features['unique_items_session'] = current_session['item_id'].nunique()

        return features

    def _user_item_affinity_features(
        self,
        user_id: str,
        item_id: str,
        historical_events: pd.DataFrame,
        catalog_df: pd.DataFrame
    ) -> Dict:
        """User's historical interaction with this specific item."""
        features = {}

        item_events = historical_events[historical_events['item_id'] == item_id]

        features['times_viewed_this_item'] = (
            item_events['event_type'] == 'view'
        ).sum()

        features['times_scrolled_this_item'] = (
            item_events['event_type'] == 'scroll'
        ).sum()

        features['times_carted_this_item'] = (
            item_events['event_type'] == 'add_to_cart'
        ).sum()

        # Category affinity
        item_category = catalog_df[catalog_df['item_id'] == item_id]['category'].iloc[0]
        category_events = historical_events.merge(
            catalog_df[['item_id', 'category']], on='item_id', how='left'
        )

        features['times_viewed_this_category'] = (
            (category_events['category'] == item_category) &
            (category_events['event_type'] == 'view')
        ).sum()

        return features

    def _item_features(
        self,
        item_id: str,
        catalog_df: pd.DataFrame
    ) -> Dict:
        """Static item features from catalog."""
        features = {}

        item_row = catalog_df[catalog_df['item_id'] == item_id].iloc[0]

        features['item_current_price'] = item_row['current_price']
        features['item_base_price'] = item_row['base_price']
        features['item_discount_pct'] = item_row['discount_pct']
        features['item_has_discount'] = int(item_row['discount_pct'] > 0)

        # Category as numeric (will one-hot encode later)
        features['item_category'] = item_row['category']

        return features


def main():
    """Test feature engineering on a small sample."""

    print("=" * 70)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 70)

    # Load data
    print("\n📂 Loading data...")
    events_df = pd.read_parquet('data/processed/train_events.parquet')
    catalog_df = pd.read_parquet('data/raw/item_catalog.parquet')

    # Test on small sample
    print("\n🧪 Testing on sample (100 users)...")
    sample_users = events_df['user_id'].unique()[:100]
    events_sample = events_df[events_df['user_id'].isin(sample_users)]

    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Create features
    examples_df = engineer.create_training_examples(
        events_df=events_sample,
        catalog_df=catalog_df
    )

    # Show sample
    print("\n📊 Sample features:")
    print(examples_df.head(10))

    print("\n📊 Feature summary:")
    print(f"   Total features: {len(examples_df.columns) - 5}")  # Minus metadata columns
    print(f"   Null values:")
    null_counts = examples_df.isnull().sum()
    null_features = null_counts[null_counts > 0]
    if len(null_features) > 0:
        for col, count in null_features.items():
            print(f"     {col}: {count} ({count/len(examples_df):.2%})")
    else:
        print("     None - all features complete! ✅")

    # Save sample
    output_path = 'data/processed/features_sample.parquet'
    examples_df.to_parquet(output_path, compression='snappy')
    print(f"\n💾 Saved sample to: {output_path}")


if __name__ == "__main__":
    main()
