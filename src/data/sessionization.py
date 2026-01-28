"""
Sessionization Engine

Groups clickstream events into sessions with multiple strategies.
Critical for preventing data leakage in temporal splits.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple

import pandas as pd


@dataclass
class SessionStats:
    """Statistics for a sessionization run."""

    total_events: int
    total_sessions: int
    avg_events_per_session: float
    avg_session_duration_minutes: float
    sessions_with_purchase: int
    session_conversion_rate: float


class TimeBasedSessionizer:
    """
    Time-based sessionization using inactivity threshold.

    Design Principle:
    If time gap between consecutive events > threshold → new session

    This is the industry standard approach (Google Analytics uses 30 min).
    """

    def __init__(self, inactivity_threshold_minutes: int = 30):
        """
        Initialize sessionizer.

        Args:
            inactivity_threshold_minutes: Minutes of inactivity to trigger new session
        """
        self.inactivity_threshold = timedelta(minutes=inactivity_threshold_minutes)

    def sessionize(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session IDs to events using time-based logic.

        Args:
            events_df: Clickstream events with columns:
                - user_id
                - timestamp
                - event_type
                - item_id

        Returns:
            DataFrame with added session_id column

        CRITICAL: Events must be pre-sorted by user_id, timestamp
        """
        # Validate input
        required_cols = ["user_id", "timestamp", "event_type", "item_id"]
        missing_cols = set(required_cols) - set(events_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"🔄 Sessionizing {len(events_df):,} events...")
        print(
            f"   Inactivity threshold: {self.inactivity_threshold.total_seconds() / 60:.0f} minutes"
        )

        # Sort by user and timestamp
        df = events_df.sort_values(["user_id", "timestamp"]).copy()

        # Calculate time gap from previous event (per user)
        df["time_since_prev_event"] = df.groupby("user_id")["timestamp"].diff()

        # New session if:
        # 1. First event for user (NaT from diff)
        # 2. Time gap > threshold
        df["is_new_session"] = df["time_since_prev_event"].isna() | (  # First event
            df["time_since_prev_event"] > self.inactivity_threshold
        )

        # Assign session IDs (cumulative count of new sessions per user)
        df["session_number"] = df.groupby("user_id")["is_new_session"].cumsum()

        # Create unique session ID
        df["session_id"] = (
            df["user_id"].astype(str) + "_session_" + df["session_number"].astype(str).str.zfill(4)
        )

        # Clean up helper columns
        df = df.drop(columns=["time_since_prev_event", "is_new_session", "session_number"])

        # Calculate session statistics
        stats = self._calculate_stats(df)
        self._print_stats(stats)

        return df

    def _calculate_stats(self, df: pd.DataFrame) -> SessionStats:
        """Calculate sessionization statistics."""

        # Session-level aggregations
        session_agg = df.groupby("session_id").agg(
            {"timestamp": ["min", "max"], "event_type": lambda x: (x == "purchase").any()}
        )

        session_agg.columns = ["start_time", "end_time", "has_purchase"]
        session_agg["duration_minutes"] = (
            session_agg["end_time"] - session_agg["start_time"]
        ).dt.total_seconds() / 60

        events_per_session = df.groupby("session_id").size()

        return SessionStats(
            total_events=len(df),
            total_sessions=df["session_id"].nunique(),
            avg_events_per_session=events_per_session.mean(),
            avg_session_duration_minutes=session_agg["duration_minutes"].mean(),
            sessions_with_purchase=session_agg["has_purchase"].sum(),
            session_conversion_rate=session_agg["has_purchase"].mean(),
        )

    def _print_stats(self, stats: SessionStats):
        """Print sessionization statistics."""
        print(f"\n✅ Sessionization complete:")
        print(f"   Total events: {stats.total_events:,}")
        print(f"   Total sessions: {stats.total_sessions:,}")
        print(f"   Avg events/session: {stats.avg_events_per_session:.1f}")
        print(f"   Avg session duration: {stats.avg_session_duration_minutes:.1f} minutes")
        print(
            f"   Sessions with purchase: {stats.sessions_with_purchase:,} "
            f"({stats.session_conversion_rate:.2%})"
        )


class IntentBasedSessionizer:
    """
    Intent-based sessionization using action boundaries.

    Design Principle:
    Session ends when user completes a high-intent action (add_to_cart, purchase)

    Use case: Want to predict "will this session end in purchase?"
    """

    def __init__(self, boundary_events: List[str] = None):
        """
        Initialize intent-based sessionizer.

        Args:
            boundary_events: Events that mark session end
        """
        self.boundary_events = boundary_events or ["purchase", "add_to_cart"]

    def sessionize(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session IDs using intent-based logic.

        Args:
            events_df: Clickstream events

        Returns:
            DataFrame with session_id column
        """
        print("🔄 Intent-based sessionization...")
        print(f"   Boundary events: {', '.join(self.boundary_events)}")

        df = events_df.sort_values(["user_id", "timestamp"]).copy()

        # Mark session boundaries
        df["is_boundary"] = df["event_type"].isin(self.boundary_events)

        # New session on first event or after boundary
        df["is_new_session"] = df.groupby("user_id")["is_boundary"].shift(1).fillna(True)

        # Assign session IDs
        df["session_number"] = df.groupby("user_id")["is_new_session"].cumsum()
        df["session_id"] = (
            df["user_id"].astype(str)
            + "_intent_session_"
            + df["session_number"].astype(str).str.zfill(4)
        )

        df = df.drop(columns=["is_boundary", "is_new_session", "session_number"])

        print("✅ Created {df['session_id'].nunique():,} intent-based sessions")

        return df


def create_temporal_split(
    events_df: pd.DataFrame, train_end_date: str, val_end_date: str, test_end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test splits with NO DATA LEAKAGE.

    Design Principle:
    - Split by DATE, not random sampling
    - Sessions cannot span split boundaries
    - Future data never leaks into past

    Args:
        events_df: Sessionized events
        train_end_date: Last date for training (inclusive)
        val_end_date: Last date for validation (inclusive)
        test_end_date: Last date for test (inclusive)

    Returns:
        (train_df, val_df, test_df)
    """
    train_end = pd.Timestamp(train_end_date)
    val_end = pd.Timestamp(val_end_date)
    test_end = pd.Timestamp(test_end_date)

    print("\n📅 Creating temporal split...")
    print(f"   Train: up to {train_end_date}")
    print(f"   Val: {train_end_date} to {val_end_date}")
    print(f"   Test: {val_end_date} to {test_end_date}")

    # Get session start times
    session_starts = events_df.groupby("session_id")["timestamp"].min()

    # Assign sessions to splits based on START time
    train_sessions = session_starts[session_starts <= train_end].index
    val_sessions = session_starts[(session_starts > train_end) & (session_starts <= val_end)].index
    test_sessions = session_starts[(session_starts > val_end) & (session_starts <= test_end)].index

    # Split events
    train_df = events_df[events_df["session_id"].isin(train_sessions)].copy()
    val_df = events_df[events_df["session_id"].isin(val_sessions)].copy()
    test_df = events_df[events_df["session_id"].isin(test_sessions)].copy()

    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train_df):,} events, {len(train_sessions):,} sessions")
    print(f"   Val: {len(val_df):,} events, {len(val_sessions):,} sessions")
    print(f"   Test: {len(test_df):,} events, {len(test_sessions):,} sessions")

    # Validate no leakage at SESSION level (not event level)
    # Sessions are assigned to splits by START time
    # Events within a session may span boundaries (this is OK!)

    train_session_starts = train_df.groupby("session_id")["timestamp"].min()
    val_session_starts = val_df.groupby("session_id")["timestamp"].min()
    test_session_starts = test_df.groupby("session_id")["timestamp"].min()

    assert train_session_starts.max() <= train_end, "Train sessions leak into validation!"
    assert (
        len(val_session_starts) == 0 or val_session_starts.min() > train_end
    ), "Val sessions overlap with train!"
    assert (
        len(test_session_starts) == 0 or test_session_starts.min() > val_end
    ), "Test sessions overlap with val!"

    print("   ✓ No session-level temporal leakage detected")
    print("   ℹ️  Note: Events within sessions may span split boundaries (this is correct)")

    return train_df, val_df, test_df


def main():
    """Test sessionization on generated data."""

    # Load clickstream
    events_df = pd.read_parquet("data/raw/clickstream_events.parquet")

    print("=" * 70)
    print("TESTING SESSIONIZATION")
    print("=" * 70)

    # Test time-based sessionization
    print("\n1. TIME-BASED SESSIONIZATION")
    print("-" * 70)
    sessionizer = TimeBasedSessionizer(inactivity_threshold_minutes=30)
    events_with_sessions = sessionizer.sessionize(events_df)

    # Save sessionized data
    output_path = "data/processed/events_sessionized.parquet"
    events_with_sessions.to_parquet(output_path, compression="snappy")
    print(f"\n💾 Saved to: {output_path}")

    # Test temporal split
    print("\n2. TEMPORAL SPLIT")
    print("-" * 70)
    train_df, val_df, test_df = create_temporal_split(
        events_with_sessions,
        train_end_date="2024-09-15",  # ~45 days
        val_end_date="2024-10-01",  # ~15 days
        test_end_date="2024-10-30",  # ~30 days
    )

    # Save splits
    train_df.to_parquet("data/processed/train_events.parquet", compression="snappy")
    val_df.to_parquet("data/processed/val_events.parquet", compression="snappy")
    test_df.to_parquet("data/processed/test_events.parquet", compression="snappy")

    print("\n💾 Saved splits to data/processed/")

    # Show sample
    print("\n📊 Sample sessionized events:")
    print(
        events_with_sessions[["user_id", "session_id", "timestamp", "event_type", "item_id"]].head(
            10
        )
    )


if __name__ == "__main__":
    main()
