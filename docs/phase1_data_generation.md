# Phase 1: Data Generation - Implementation Summary

## Overview
Built a production-grade synthetic data generator for e-commerce clickstream simulation.

## Components

### 1. Item Catalog Generator (`src/data/catalog.py`)
**Purpose:** Generate realistic product catalog with pricing strategies.

**Key Features:**
- **Log-normal price distribution** (most items mid-range, few expensive)
- **5 product categories** with distinct pricing rules
- **Dynamic discounts** (category-specific probability and ranges)
- **Temporal metadata** (item creation dates for popularity modeling)

**Output:** `data/raw/item_catalog.parquet`
- 1,000 items
- Columns: `item_id`, `category`, `base_price`, `discount_pct`, `current_price`, `created_at`

**Why This Matters:**
- Price sensitivity features need realistic distributions
- Category affinity scoring requires diverse catalog
- Discount patterns affect purchase behavior

---

### 2. User Behavior Simulator (`src/data/users.py`)
**Purpose:** Create users with distinct behavioral personas.

**Key Features:**
- **3 Personas** with different conversion patterns:
  - **Browsers (70%):** High views, low purchases (~1% conversion)
  - **Researchers (25%):** Medium engagement, comparison shopping (~5% conversion)
  - **Buyers (5%):** High intent, quick decisions (~25% conversion)
- **Probabilistic actions** (not deterministic)
- **Session frequency** varies by persona
- **Registration dates** (users joined over time)

**Output:** `data/raw/users.parquet`
- 10,000 users
- Columns: `user_id`, `persona`, `sessions_per_week`, `registration_date`

**Why This Matters:**
- Creates realistic 95:5 class imbalance (only 5% are buyers)
- User history features need temporal depth
- Persona diversity tests model generalization

---

### 3. Clickstream Event Generator (`src/data/clickstream.py`)
**Purpose:** Generate temporal event sequences with realistic patterns.

**Key Features:**
- **Temporal realism:**
  - Hourly traffic patterns (peak evenings: 6pm-9pm)
  - Daily patterns (weekends > weekdays)
  - Seasonal variations possible
- **Event sequences:** view → scroll → add_to_cart → purchase
- **User-driven:** Each session respects user's persona and registration date
- **No leakage:** Events strictly chronological, no future data

**Output:** `data/raw/clickstream_events.parquet`
- ~500k-1M events over 90 days
- Columns: `user_id`, `session_id`, `timestamp`, `event_type`, `item_id`, `duration_seconds`

**Event Distribution:**
- Views: ~70-80%
- Scrolls: ~40-50%
- Add-to-cart: ~5-10%
- Purchases: ~1-3%

**Why This Matters:**
- Temporal patterns test feature engineering correctness
- Realistic imbalance (5% purchase rate at event level)
- Session boundaries critical for label definition

---

## Data Quality Guarantees

### 1. Reproducibility
- Fixed random seed (42)
- Deterministic generation
- Same config → same data

### 2. Temporal Correctness
- Users can't have events before registration
- Events within session are chronologically ordered
- Session timestamps follow realistic patterns

### 3. Statistical Realism
- Price distributions follow market patterns (log-normal)
- User behavior matches e-commerce research (conversion funnels)
- Traffic patterns match Google Analytics benchmarks

---

## Dataset Statistics
```
Item Catalog:
  Total items: 1,000
  Categories: 5
  Price range: $10 - $2,000
  Items with discounts: ~30%

Users:
  Total users: 10,000
  Browsers: 7,000 (70%)
  Researchers: 2,500 (25%)
  Buyers: 500 (5%)

Clickstream:
  Total events: ~500k-1M
  Date range: 2024-08-01 to 2024-10-30 (90 days)
  Total sessions: ~50k-100k
  Purchase rate: 1-3% (events), 3-7% (sessions)
```

---

## File Sizes
- `item_catalog.parquet`: ~100 KB
- `users.parquet`: ~300 KB
- `clickstream_events.parquet`: ~15-30 MB (compressed)

---

## Design Decisions

### Why Synthetic Data?
1. **Control:** Can tune imbalance ratio, seasonality, user behaviors
2. **Privacy:** No real user data, no GDPR concerns
3. **Reproducibility:** Same seed → same experiments
4. **Scale:** Generate millions of events easily
5. **Ground truth:** Know true user intent (persona labels)

### Why Parquet?
1. **Compression:** 50-90% smaller than CSV
2. **Columnar:** Fast for analytical queries
3. **Schema:** Built-in type information
4. **Arrow-compatible:** Zero-copy with pandas, DuckDB

### Why 3 Personas?
1. **Captures spectrum:** From window shoppers to ready buyers
2. **Creates imbalance:** Mirrors real e-commerce (most users browse)
3. **Testable:** Can measure model performance per persona

---

## Next Steps (Phase 2)
1. **Sessionization:** Group events into meaningful units
2. **Temporal splits:** Train/val/test with no leakage
3. **Data quality checks:** Great Expectations suite

---

## Code Metrics
- Lines of code: ~800
- Test coverage: Not yet implemented
- Documentation: Inline docstrings + this guide
