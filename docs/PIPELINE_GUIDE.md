# Pipeline Orchestration Guide

## Overview

The `run_pipeline.py` script orchestrates the complete end-to-end machine learning pipeline, connecting all the isolated scripts in the `src/` directory in the correct order.

## Pipeline Stages

The pipeline consists of 5 main stages:

### Stage 1: Data Generation
- **Purpose**: Generate synthetic e-commerce data
- **Components**:
  - User profiles with personas (browsers, researchers, buyers)
  - Item catalog with categories and pricing
  - Clickstream events with temporal patterns
- **Output**: `data/raw/users.parquet`, `data/raw/item_catalog.parquet`, `data/raw/clickstream_events.parquet`

### Stage 2: Sessionization
- **Purpose**: Group clickstream events into sessions
- **Method**: Time-based sessionization (30-minute inactivity threshold)
- **Output**: `data/processed/sessionized_events.parquet`

### Stage 3: Temporal Split
- **Purpose**: Create train/val/test splits with NO data leakage
- **Split**: 70% train / 15% validation / 15% test (by date)
- **Output**: `data/processed/train_events.parquet`, `data/processed/val_events.parquet`, `data/processed/test_events.parquet`

### Stage 4: Feature Engineering
- **Purpose**: Extract features with point-in-time correctness
- **Features**:
  - Recency features (time since last event)
  - Frequency features (event counts in time windows)
  - Intent features (user engagement signals)
  - Session context features
  - User-item affinity features
  - Item features (price, category, discount)
- **Output**: `data/features/train_features.parquet`, `data/features/val_features.parquet`, `data/features/test_features.parquet`

### Stage 5: Model Training
- **Purpose**: Train models with extreme class imbalance handling
- **Models**:
  - Baseline: Logistic Regression
  - Main: XGBoost with scale_pos_weight
- **Output**: `models/xgboost_YYYYMMDD_HHMMSS.pkl`

## Usage

### Prerequisites

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies** (if not already done):
   ```bash
   pip install -e ".[dev]"
   ```

### Running the Pipeline

#### Option 1: Using Python directly

**Run full pipeline**:
```bash
python run_pipeline.py --full
```

**Run specific stages**:
```bash
# Data generation only
python run_pipeline.py --data-gen

# Sessionization only
python run_pipeline.py --sessionize

# Feature engineering and training
python run_pipeline.py --features --train

# Multiple stages
python run_pipeline.py --data-gen --sessionize --split
```

**Skip data generation** (use existing data):
```bash
python run_pipeline.py --skip-data-gen --full
```

**Use custom config**:
```bash
python run_pipeline.py --config configs/custom.yaml --full
```

#### Option 2: Using shell script wrapper

**Run full pipeline**:
```bash
./scripts/run_pipeline.sh
```

**Run with arguments**:
```bash
./scripts/run_pipeline.sh --full
./scripts/run_pipeline.sh --features --train
./scripts/run_pipeline.sh --skip-data-gen --full
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--full` | Run complete end-to-end pipeline |
| `--data-gen` | Run data generation stage only |
| `--sessionize` | Run sessionization stage only |
| `--split` | Run temporal split stage only |
| `--features` | Run feature engineering stage only |
| `--train` | Run model training stage only |
| `--skip-data-gen` | Skip data generation (use existing data) |
| `--config PATH` | Use custom configuration file |

### Examples

**Quick start** (full pipeline):
```bash
python run_pipeline.py --full
```

**Iterative development** (skip data generation):
```bash
# First run: generate data
python run_pipeline.py --data-gen --sessionize --split

# Subsequent runs: iterate on features/models
python run_pipeline.py --features --train
```

**Custom configuration**:
```bash
# Create custom config
cp configs/data_generation.yaml configs/experiment_1.yaml
# Edit configs/experiment_1.yaml

# Run with custom config
python run_pipeline.py --config configs/experiment_1.yaml --full
```

## Output Structure

After running the full pipeline, your directory structure will look like:

```
pred-analytics-HON/
├── data/
│   ├── raw/                          # Stage 1 output
│   │   ├── users.parquet
│   │   ├── item_catalog.parquet
│   │   └── clickstream_events.parquet
│   ├── processed/                    # Stages 2-3 output
│   │   ├── sessionized_events.parquet
│   │   ├── train_events.parquet
│   │   ├── val_events.parquet
│   │   └── test_events.parquet
│   └── features/                     # Stage 4 output
│       ├── train_features.parquet
│       ├── val_features.parquet
│       └── test_features.parquet
├── models/                           # Stage 5 output
│   └── xgboost_20240101_120000.pkl
└── pipeline.log                      # Execution log
```

## Logging

The pipeline logs all activities to:
- **Console**: Real-time progress with colored output
- **File**: `pipeline.log` for detailed execution history

Log format:
```
2024-01-01 12:00:00 - __main__ - INFO - 🚀 E-Commerce Purchase Prediction Pipeline
2024-01-01 12:00:01 - __main__ - INFO - ✅ Loaded configuration from configs/data_generation.yaml
...
```

## Configuration

The pipeline uses YAML configuration files in the `configs/` directory:

- **`data_generation.yaml`**: Main configuration for data generation, user personas, item catalog, temporal patterns
- **`feature_engineering.yaml`**: Feature extraction parameters and time windows
- **`model_training.yaml`**: Model hyperparameters and training settings

### Key Configuration Parameters

**Data Generation** (`configs/data_generation.yaml`):
```yaml
dataset:
  num_users: 10000        # Number of users to generate
  num_items: 1000         # Number of items in catalog
  num_days: 90            # Days of historical data
  start_date: "2024-08-01"
  random_seed: 42
```

**Session Parameters**:
```yaml
session:
  inactivity_threshold_minutes: 30  # Session timeout
```

## Error Handling

The pipeline includes comprehensive error handling:

1. **Graceful failures**: Each stage logs errors and exits cleanly
2. **Intermediate saves**: All stages save outputs to disk
3. **Idempotent stages**: Stages can be re-run safely
4. **Detailed logging**: Full stack traces in `pipeline.log`

If a stage fails:
1. Check `pipeline.log` for detailed error messages
2. Verify input files exist in expected locations
3. Check configuration parameters
4. Re-run failed stage individually

## Performance Tips

**For faster iteration**:
```bash
# Generate data once
python run_pipeline.py --data-gen --sessionize --split

# Iterate on features/models
python run_pipeline.py --features --train
```

**For smaller datasets** (faster testing):
```yaml
# Edit configs/data_generation.yaml
dataset:
  num_users: 1000    # Reduce from 10000
  num_days: 30       # Reduce from 90
```

**For production runs**:
```yaml
dataset:
  num_users: 100000  # Scale up
  num_days: 365      # Full year
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Solution**: Make sure you're in the project root and have installed the package:
  ```bash
  pip install -e .
  ```

**Issue**: `FileNotFoundError: configs/data_generation.yaml`
- **Solution**: Run from project root directory:
  ```bash
  cd /path/to/pred-analytics-HON
  python run_pipeline.py --full
  ```

**Issue**: Pipeline runs out of memory
- **Solution**: Reduce dataset size in configuration or process in chunks

**Issue**: XGBoost training is slow
- **Solution**: Reduce `max_depth` or `n_estimators` in `configs/model_training.yaml`

## Next Steps

After running the pipeline:

1. **Evaluate results**: Check `pipeline.log` for metrics
2. **Analyze features**: Use notebooks to explore feature importance
3. **Tune models**: Adjust hyperparameters in `configs/model_training.yaml`
4. **Deploy model**: Use the saved model in `models/` for serving
5. **Monitor performance**: Set up MLflow tracking (if available)

## Integration with Existing Scripts

The pipeline orchestrator imports and uses the existing `main()` functions from:

- `src/data/users.py` → `UserBehaviorSimulator`
- `src/data/catalog.py` → `ItemCatalogGenerator`
- `src/data/clickstream.py` → `ClickstreamGenerator`
- `src/data/sessionization.py` → `TimeBasedSessionizer`, `create_temporal_split`
- `src/features/engineering.py` → `FeatureEngineer`
- `src/models/trainer.py` → `ModelTrainer`

You can still run individual scripts directly:
```bash
python -m src.data.users
python -m src.data.catalog
python -m src.data.clickstream
python -m src.data.sessionization
python -m src.features.engineering
python -m src.models.trainer
```

The orchestrator simply connects them in the correct order with proper data flow.
