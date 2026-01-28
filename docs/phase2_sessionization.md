# Phase 2: Sessionization - Implementation Summary

## Overview
Built production-grade session logic that groups events into meaningful units while preventing data leakage.

## Critical Design Principle
**Sessions must be complete** - we cannot split a session across train/val/test boundaries. This would leak future information into the past.

## Implementation

### Time-Based Sessionization
**Algorithm:**
```
For each user:
  Sort events by timestamp
  For each event:
    If time_gap_from_previous > 30 minutes:
      Start new session
    Else:
      Continue current session
```

**Why 30 minutes?**
- Industry standard (Google Analytics default)
- Balances user intent persistence vs natural breaks
- Research shows 30-min captures distinct shopping trips

**Statistics from our data:**
- Total events: 313,036
- Total sessions: 15,858
- Avg events/session: 19.7
- Avg session duration: 7.5 minutes
- Session conversion rate: 5.07%

### Temporal Splitting Strategy

**Split Logic:**
```
Sessions assigned to splits by START time:
- Train: sessions starting before 2024-09-15 (45 days)
- Val: sessions starting 2024-09-15 to 2024-10-01 (16 days)
- Test: sessions starting 2024-10-01 to 2024-10-30 (29 days)
```

**Result:**
- Train: 138,059 events (7,048 sessions)
- Val: 56,585 events (2,909 sessions)
- Test: 118,392 events (5,901 sessions)

**Why this split?**
- **50/18/32** split (train/val/test by events)
- Train gets more data (model needs it)
- Val is smaller (just for hyperparameter tuning)
- Test is substantial (reliable performance estimate)

### Leakage Prevention

**Session-level validation:**
```python
assert train_session_starts.max() <= train_end
assert val_session_starts.min() > train_end
assert test_session_starts.min() > val_end
```

**Key insight:** Individual events within a session can span date boundaries (user starts session at 11:55pm, continues to 12:05am). This is CORRECT - we want complete sessions.

**What we prevent:**
- ❌ Using same-session events to predict same-session outcomes
- ❌ Training on future data
- ❌ Feature computation using "future" user actions

## Alternative: Intent-Based Sessionization

Also implemented but not used as primary:

**Logic:** Session ends when user completes high-intent action (add_to_cart, purchase)

**Use case:** Predicting "will THIS action sequence end in purchase?"

**Trade-off:**
- Pro: Better aligned with prediction target
- Con: Creates very short sessions (2-5 events)
- Con: Less natural for feature engineering

We stick with time-based for main pipeline.

## Data Quality Checks

### Session validity
- ✅ All sessions have at least 1 event
- ✅ Events within session are chronologically ordered
- ✅ Session IDs are unique across all users
- ✅ No sessions span multiple days beyond natural user behavior

### Temporal correctness
- ✅ No session starts in future
- ✅ Train sessions end before val sessions start
- ✅ Val sessions end before test sessions start

## File Outputs
```
data/processed/
├── events_sessionized.parquet     # All events with session_id
├── train_events.parquet            # Training set events
├── val_events.parquet              # Validation set events
└── test_events.parquet             # Test set events
```

## Next Steps (Phase 3)
Now that we have clean, leak-free sessions, we can:
1. Engineer features at session level
2. Create point-in-time feature lookbacks
3. Build prediction targets safely

## Key Learnings

### Why not random split?
Random splitting would:
- Put same user's sessions in train AND test
- Allow model to memorize user patterns
- Overestimate real-world performance

### Why not split by user?
Splitting by user would:
- Put some users entirely in test
- Model never sees their behavior patterns
- Underestimate performance (too pessimistic)

**Temporal split is the goldilocks:** Tests model on future behavior of known users (realistic deployment scenario).

---

**Decision Rationale:**
- Time-based over intent-based: More events per session, better for features
- 30-min threshold: Industry standard, validated by research
- 50/18/32 split: Balanced between training data and test reliability
- Session-start-time assignment: Prevents session fragmentation
