"""
Clickstream Event Generator

Generates realistic e-commerce clickstream events with temporal patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm

from src.data.catalog import ItemCatalogGenerator
from src.data.users import UserBehaviorSimulator, UserPersona


class ClickstreamGenerator:
    """
    Generates realistic e-commerce clickstream events.
    
    Design Principles:
    - Temporal patterns (hourly/daily variations)
    - User persona-driven behavior
    - Realistic event sequences (view → scroll → cart → purchase)
    - No data leakage (events respect temporal order)
    """
    
    def __init__(
        self, 
        users_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
        config: Dict,
        random_seed: int = 42
    ):
        """
        Initialize clickstream generator.
        
        Args:
            users_df: User profiles with personas
            catalog_df: Item catalog
            config: Full configuration from YAML
            random_seed: Random seed for reproducibility
        """
        self.users_df = users_df
        self.catalog_df = catalog_df
        self.config = config
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize user behavior simulator
        self.behavior_simulator = UserBehaviorSimulator(
            config['user_personas'],
            random_seed=random_seed
        )
        
        # Parse temporal patterns
        self.hourly_multipliers = np.array(config['temporal']['hourly_traffic'])
        self.daily_multipliers = np.array(config['temporal']['daily_traffic'])
        
        # Parse event durations
        self.event_durations = config['events']['duration']
        
    def generate_clickstream(
        self, 
        start_date: str,
        num_days: int,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Generate complete clickstream for all users over specified time period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            num_days: Number of days to simulate
            output_path: Optional path to save results
            
        Returns:
            DataFrame with clickstream events
        """
        start_dt = pd.Timestamp(start_date)
        end_dt = start_dt + pd.Timedelta(days=num_days)
        
        print(f"🎬 Generating clickstream from {start_date} for {num_days} days...")
        print(f"   Users: {len(self.users_df)}")
        print(f"   Items: {len(self.catalog_df)}")
        
        all_events = []
        
        # Generate sessions for each user
        for _, user in tqdm(self.users_df.iterrows(), total=len(self.users_df), desc="Generating sessions"):
            user_events = self._generate_user_sessions(
                user=user,
                start_date=start_dt,
                end_date=end_dt
            )
            
            if user_events:
                all_events.extend(user_events)
        
        # Convert to DataFrame
        events_df = pd.DataFrame(all_events)
        
        # Sort by timestamp
        events_df = events_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\n✅ Generated {len(events_df):,} events")
        print(f"   Sessions: {events_df['session_id'].nunique():,}")
        print(f"   Purchases: {(events_df['event_type'] == 'purchase').sum():,}")
        print(f"   Purchase rate: {(events_df['event_type'] == 'purchase').mean():.2%}")
        
        # Save if path provided
        if output_path:
            events_df.to_parquet(output_path, compression='snappy')
            print(f"💾 Saved clickstream to: {output_path}")
        
        return events_df
    
    def _generate_user_sessions(
        self,
        user: pd.Series,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[Dict]:
        """
        Generate all sessions for a single user.
        
        Args:
            user: User profile row
            start_date: Simulation start date
            end_date: Simulation end date
            
        Returns:
            List of event dictionaries
        """
        # User can only have sessions after registration
        user_start = max(user['registration_date'], start_date)
        
        if user_start >= end_date:
            return []
        
        # Calculate number of sessions for this user
        num_weeks = (end_date - user_start).days / 7
        expected_sessions = int(user['sessions_per_week'] * num_weeks)
        
        if expected_sessions == 0:
            return []
        
        # Generate session timestamps with temporal patterns
        session_timestamps = self._generate_session_timestamps(
            start_date=user_start,
            end_date=end_date,
            num_sessions=expected_sessions
        )
        
        # Generate events for each session
        all_events = []
        persona = UserPersona(user['persona'])
        
        for session_idx, session_start in enumerate(session_timestamps):
            session_id = f"{user['user_id']}_session_{session_idx:04d}"
            
            session_events = self._generate_session_events(
                user_id=user['user_id'],
                session_id=session_id,
                session_start=session_start,
                persona=persona
            )
            
            all_events.extend(session_events)
        
        return all_events
    
    def _generate_session_timestamps(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        num_sessions: int
    ) -> List[pd.Timestamp]:
        """
        Generate session start timestamps with realistic temporal patterns.
        
        Args:
            start_date: Earliest possible session
            end_date: Latest possible session
            num_sessions: Number of sessions to generate
            
        Returns:
            List of session start timestamps
        """
        timestamps = []
        
        for _ in range(num_sessions):
            # Random day within period
            days_offset = self.rng.randint(0, (end_date - start_date).days)
            session_date = start_date + pd.Timedelta(days=days_offset)
            
            # Apply day of week multiplier
            day_of_week = session_date.dayofweek
            day_multiplier = self.daily_multipliers[day_of_week]
            
            # Random hour weighted by hourly pattern
            hour_probs = self.hourly_multipliers / self.hourly_multipliers.sum()
            hour = self.rng.choice(24, p=hour_probs)
            
            # Random minute and second
            minute = self.rng.randint(0, 60)
            second = self.rng.randint(0, 60)
            
            timestamp = session_date.replace(hour=hour, minute=minute, second=second)
            timestamps.append(timestamp)
        
        return sorted(timestamps)
    
    def _generate_session_events(
        self,
        user_id: str,
        session_id: str,
        session_start: pd.Timestamp,
        persona: UserPersona
    ) -> List[Dict]:
        """
        Generate all events within a single session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            session_start: Session start timestamp
            persona: User's persona type
            
        Returns:
            List of event dictionaries
        """
        # Determine how many items user will interact with
        num_items = self.behavior_simulator.get_items_per_session(persona)
        
        # Randomly select items from catalog
        item_indices = self.rng.choice(len(self.catalog_df), size=num_items, replace=False)
        selected_items = self.catalog_df.iloc[item_indices]['item_id'].tolist()
        
        # Simulate behavior (which items get scrolled, carted, purchased)
        behavior = self.behavior_simulator.simulate_session_behavior(persona, num_items)
        
        # Generate events
        events = []
        current_time = session_start
        
        for item_idx, item_id in enumerate(selected_items):
            # View event (always happens)
            view_duration = self.rng.uniform(*self.event_durations['view'])
            events.append({
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': current_time,
                'event_type': 'view',
                'item_id': item_id,
                'duration_seconds': round(view_duration, 1)
            })
            current_time += pd.Timedelta(seconds=view_duration)
            
            # Scroll event (if applicable)
            if item_idx in behavior['scrolled']:
                scroll_duration = self.rng.uniform(*self.event_durations['scroll'])
                events.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': current_time,
                    'event_type': 'scroll',
                    'item_id': item_id,
                    'duration_seconds': round(scroll_duration, 1)
                })
                current_time += pd.Timedelta(seconds=scroll_duration)
            
            # Add to cart event (if applicable)
            if item_idx in behavior['added_to_cart']:
                cart_duration = self.rng.uniform(*self.event_durations['add_to_cart'])
                events.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': current_time,
                    'event_type': 'add_to_cart',
                    'item_id': item_id,
                    'duration_seconds': round(cart_duration, 1)
                })
                current_time += pd.Timedelta(seconds=cart_duration)
            
            # Purchase event (if applicable)
            if item_idx in behavior['purchased']:
                purchase_duration = self.rng.uniform(*self.event_durations['purchase'])
                events.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': current_time,
                    'event_type': 'purchase',
                    'item_id': item_id,
                    'duration_seconds': round(purchase_duration, 1)
                })
                current_time += pd.Timedelta(seconds=purchase_duration)
        
        return events


def main():
    """Test the clickstream generator."""
    import yaml
    
    # Load config
    with open('configs/data_generation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load users and catalog
    users_df = pd.read_parquet('data/raw/users.parquet')
    catalog_df = pd.read_parquet('data/raw/item_catalog.parquet')
    
    # Generate clickstream (use smaller dataset for testing)
    generator = ClickstreamGenerator(
        users_df=users_df.head(100),  # Test with 100 users
        catalog_df=catalog_df,
        config=config,
        random_seed=config['dataset']['random_seed']
    )
    
    events_df = generator.generate_clickstream(
        start_date=config['dataset']['start_date'],
        num_days=7,  # Test with 1 week
        output_path='data/raw/clickstream_test.parquet'
    )
    
    # Show statistics
    print(f"\n📊 Event type distribution:")
    print(events_df['event_type'].value_counts().to_frame())
    
    print(f"\n📊 Sample events:")
    print(events_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()