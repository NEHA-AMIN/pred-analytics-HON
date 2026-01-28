"""
User Behavior Simulator

Simulates different user personas (browsers, researchers, buyers) with distinct behaviors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class UserPersona(Enum):
    """User behavior personas."""
    BROWSER = "browser"
    RESEARCHER = "researcher"
    BUYER = "buyer"


@dataclass
class PersonaConfig:
    """Configuration for a user persona."""
    probability: float
    sessions_per_week: int
    avg_items_per_session: int
    view_probability: float
    scroll_probability: float
    add_to_cart_probability: float
    purchase_probability: float


class UserBehaviorSimulator:
    """
    Simulates realistic user behavior patterns.
    
    Design Principles:
    - Three distinct personas with different conversion patterns
    - Stochastic behavior (probabilistic actions)
    - Realistic engagement patterns
    """
    
    def __init__(self, config: Dict, random_seed: int = 42):
        """
        Initialize user behavior simulator.
        
        Args:
            config: User persona configuration from YAML
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Parse persona configs
        self.personas = {
            UserPersona.BROWSER: PersonaConfig(**config['browser']),
            UserPersona.RESEARCHER: PersonaConfig(**config['researcher']),
            UserPersona.BUYER: PersonaConfig(**config['buyer'])
        }
        
    def generate_users(self, num_users: int) -> pd.DataFrame:
        """
        Generate user profiles with assigned personas.
        
        Args:
            num_users: Number of users to generate
            
        Returns:
            DataFrame with columns:
            - user_id: Unique identifier
            - persona: Assigned persona type
            - sessions_per_week: Expected session frequency
            - registration_date: When user joined
        """
        # Generate user IDs
        user_ids = [f"user_{i:06d}" for i in range(num_users)]
        
        # Assign personas based on probabilities
        persona_probs = [
            self.personas[UserPersona.BROWSER].probability,
            self.personas[UserPersona.RESEARCHER].probability,
            self.personas[UserPersona.BUYER].probability
        ]
        
        personas = self.rng.choice(
            [p.value for p in UserPersona],
            size=num_users,
            p=persona_probs
        )
        
        # Get sessions per week for each user
        sessions_per_week = []
        for persona_name in personas:
            persona = UserPersona(persona_name)
            base_sessions = self.personas[persona].sessions_per_week
            # Add some variance (±30%)
            actual_sessions = self.rng.normal(base_sessions, base_sessions * 0.3)
            actual_sessions = max(1, int(actual_sessions))  # At least 1 session/week
            sessions_per_week.append(actual_sessions)
        
        # Generate registration dates (users joined over past 2 years)
        days_ago = self.rng.exponential(scale=180, size=num_users)  # ~6 months avg
        days_ago = np.clip(days_ago, 0, 365 * 2)  # Max 2 years
        registration_dates = pd.Timestamp.now() - pd.to_timedelta(days_ago, unit='D')
        
        # Build dataframe
        users_df = pd.DataFrame({
            'user_id': user_ids,
            'persona': personas,
            'sessions_per_week': sessions_per_week,
            'registration_date': registration_dates
        })
        
        print(f"✅ Generated {num_users} users")
        print(f"   Browsers: {(users_df['persona'] == 'browser').sum()} "
              f"({(users_df['persona'] == 'browser').mean():.1%})")
        print(f"   Researchers: {(users_df['persona'] == 'researcher').sum()} "
              f"({(users_df['persona'] == 'researcher').mean():.1%})")
        print(f"   Buyers: {(users_df['persona'] == 'buyer').sum()} "
              f"({(users_df['persona'] == 'buyer').mean():.1%})")
        
        return users_df
    
    def get_persona_config(self, persona: UserPersona) -> PersonaConfig:
        """Get configuration for a specific persona."""
        return self.personas[persona]
    
    def simulate_session_behavior(
        self, 
        persona: UserPersona,
        num_items_viewed: int
    ) -> Dict[str, List[int]]:
        """
        Simulate user actions within a session.
        
        Args:
            persona: User's persona type
            num_items_viewed: Number of items user interacts with
            
        Returns:
            Dictionary with lists of item indices for each action type:
            - viewed: All items viewed
            - scrolled: Items where user scrolled
            - added_to_cart: Items added to cart
            - purchased: Items purchased
        """
        config = self.personas[persona]
        
        # All items are viewed
        viewed_items = list(range(num_items_viewed))
        
        # Scroll behavior (subset of viewed items)
        scroll_mask = self.rng.random(num_items_viewed) < config.scroll_probability
        scrolled_items = [i for i, scrolled in enumerate(scroll_mask) if scrolled]
        
        # Add to cart (subset of scrolled items - users need to engage first)
        cart_candidates = scrolled_items if scrolled_items else viewed_items
        cart_mask = self.rng.random(len(cart_candidates)) < config.add_to_cart_probability
        cart_indices = [cart_candidates[i] for i, added in enumerate(cart_mask) if added]
        
        # Purchase (subset of cart items)
        purchased_indices = []
        if cart_indices:
            purchase_mask = self.rng.random(len(cart_indices)) < config.purchase_probability
            purchased_indices = [cart_indices[i] for i, purchased in enumerate(purchase_mask) if purchased]
        
        return {
            'viewed': viewed_items,
            'scrolled': scrolled_items,
            'added_to_cart': cart_indices,
            'purchased': purchased_indices
        }
    
    def get_items_per_session(self, persona: UserPersona) -> int:
        """
        Get number of items a user will interact with in a session.
        
        Args:
            persona: User's persona type
            
        Returns:
            Number of items (with variance)
        """
        config = self.personas[persona]
        base_items = config.avg_items_per_session
        
        # Add variance (Poisson distribution for count data)
        actual_items = self.rng.poisson(base_items)
        actual_items = max(1, actual_items)  # At least 1 item
        
        return actual_items


def main():
    """Test the user behavior simulator."""
    import yaml
    
    # Load config
    with open('configs/data_generation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate users
    simulator = UserBehaviorSimulator(
        config=config['user_personas'],
        random_seed=config['dataset']['random_seed']
    )
    
    users_df = simulator.generate_users(
        num_users=config['dataset']['num_users']
    )
    
    # Show sample
    print(f"\n📊 Sample users:")
    print(users_df.head(10).to_string(index=False))
    
    # Test session behavior simulation
    print(f"\n🎭 Simulating session behaviors:")
    for persona in UserPersona:
        num_items = simulator.get_items_per_session(persona)
        behavior = simulator.simulate_session_behavior(persona, num_items)
        
        print(f"\n  {persona.value.upper()}:")
        print(f"    Items viewed: {len(behavior['viewed'])}")
        print(f"    Items scrolled: {len(behavior['scrolled'])}")
        print(f"    Items added to cart: {len(behavior['added_to_cart'])}")
        print(f"    Items purchased: {len(behavior['purchased'])}")
    
    # Save users
    output_path = 'data/raw/users.parquet'
    users_df.to_parquet(output_path, compression='snappy')
    print(f"\n💾 Saved users to: {output_path}")


if __name__ == "__main__":
    main()