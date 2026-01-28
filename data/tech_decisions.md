# Technical Decisions Log

> This document explains the "why" behind every major technical decision in this project.

## Data Processing

### Decision: DuckDB over Pandas
**Date:** 2025-01-29

**Context:** Need to process large clickstream datasets efficiently with SQL-like queries.

**Options Considered:**
1. Pandas - Industry standard, familiar
2. Polars - Fast, modern, Rust-based
3. DuckDB - OLAP database, SQL interface

**Decision:** DuckDB

**Rationale:**
- **SQL Interface**: Natural for sessionization queries with window functions
- **Out-of-core Processing**: Handles datasets larger than RAM via disk spilling
- **Zero-copy Arrow**: Efficient integration with other tools (pandas, pyarrow)
- **OLAP-optimized**: Columnar storage perfect for analytical queries
- **Temporal Queries**: Built-in time-travel for point-in-time correctness
- **Performance**: 10-100x faster than pandas for aggregations

**Trade-offs:**
- Less mature ecosystem than pandas
- Smaller community for troubleshooting
- SQL learning curve for pure Python developers

**References:**
- [DuckDB Benchmarks](https://duckdb.org/docs/guides/performance/benchmarks)

---

## Feature Storage

### Decision: DuckDB + Parquet over Redis/PostgreSQL
**Date:** 2025-01-29

**Context:** Need point-in-time feature retrieval with historical lookback.

**Options Considered:**
1. Redis - Fast, in-memory
2. PostgreSQL - Mature, ACID compliant
3. Feast - Dedicated feature store
4. DuckDB + Parquet - Embedded DB + columnar format

**Decision:** DuckDB + Parquet

**Rationale:**
- **Simplicity**: No separate server to manage (embedded database)
- **Cost**: Zero infrastructure cost for local development
- **Time-travel**: Native temporal queries for point-in-time features
- **Columnar Format**: Parquet is ideal for analytical workloads
- **Portability**: Files can be version-controlled, shared easily
- **Query Performance**: DuckDB handles complex aggregations efficiently

**Trade-offs:**
- Not suitable for low-latency online serving (< 10ms)
- Limited multi-user concurrency (fine for batch workloads)
- No built-in feature versioning (manual tracking needed)

**For Production Scale:** Would migrate to Feast + Redis/DynamoDB for online serving.

---

## ML Framework

### Decision: XGBoost + LightGBM over Deep Learning
**Date:** 2025-01-29

**Context:** Predict binary classification on tabular user-item features.

**Options Considered:**
1. Logistic Regression - Simple baseline
2. Random Forest - Interpretable, robust
3. XGBoost/LightGBM - Gradient boosting
4. Neural Networks - Deep learning approach

**Decision:** XGBoost as primary, LightGBM as alternative

**Rationale:**
- **Tabular Data Excellence**: GBMs are state-of-art for structured data
- **Imbalance Handling**: Built-in class weighting (`scale_pos_weight`)
- **Feature Importance**: Native support for SHAP values
- **Training Speed**: Fast convergence on moderate datasets
- **Production Proven**: Used at Airbnb, Uber, LinkedIn for similar tasks
- **Hyperparameter Control**: Extensive tuning options (depth, learning rate, etc.)

**Trade-offs:**
- Doesn't capture complex sequential patterns (would need LSTM/Transformer)
- Less effective for very high-cardinality features vs embeddings
- Requires careful feature engineering

**Baseline:** Start with Logistic Regression for sanity check.

---

## Experiment Tracking

### Decision: MLflow over Weights & Biases
**Date:** 2025-01-29

**Context:** Need to track experiments, hyperparameters, and model versions.

**Options Considered:**
1. MLflow - Open-source, self-hosted
2. Weights & Biases - SaaS, feature-rich
3. Neptune.ai - Paid, enterprise features
4. Manual tracking (JSON logs)

**Decision:** MLflow

**Rationale:**
- **Open Source**: No vendor lock-in, full control
- **Self-hosted**: Free for unlimited experiments
- **Model Registry**: Built-in versioning and staging (dev/prod)
- **Framework Agnostic**: Works with sklearn, XGBoost, PyTorch, etc.
- **Integration**: Easy export to AWS S3, Azure, GCS
- **UI**: Clean interface for comparing runs

**Trade-offs:**
- UI less polished than W&B
- Manual setup for remote tracking server
- Limited collaboration features

**For Team Scale:** Consider W&B for better collaboration and visualization.

---

## API Framework

### Decision: FastAPI over Flask
**Date:** 2025-01-29

**Context:** Build REST API for model serving with <100ms latency.

**Options Considered:**
1. Flask - Mature, widely adopted
2. FastAPI - Modern, async
3. Django REST - Full-featured
4. gRPC - High performance

**Decision:** FastAPI

**Rationale:**
- **Async Support**: Native async/await for I/O-bound operations
- **Type Safety**: Pydantic validation catches errors at development time
- **Auto-documentation**: OpenAPI/Swagger UI generated automatically
- **Performance**: 3x faster than Flask in async scenarios
- **Modern Python**: Leverages Python 3.10+ type hints
- **Dependency Injection**: Clean architecture for testing

**Trade-offs:**
- Smaller ecosystem than Flask
- Learning curve for async patterns
- Overkill for simple sync workloads

---

## Dashboard

### Decision: Streamlit over Dash/Gradio
**Date:** 2025-01-29

**Context:** Build monitoring dashboard for ML metrics and data quality.

**Options Considered:**
1. Streamlit - Pure Python, simple
2. Plotly Dash - More control, Flask-based
3. Gradio - ML-focused, minimal code
4. Custom React - Maximum flexibility

**Decision:** Streamlit

**Rationale:**
- **Rapid Development**: Build dashboard in <100 lines of code
- **Python-native**: No HTML/CSS/JavaScript required
- **Auto-refresh**: Built-in data reactivity
- **ML-friendly**: Great for plots, dataframes, metrics
- **Deployment**: Easy to containerize and share

**Trade-offs:**
- Limited customization vs custom frontend
- Reloads entire app on interaction (can be slow)
- Not suitable for complex multi-page apps with auth

**For Production Scale:** Would build custom React dashboard with better UX.

---

## Imbalance Handling

### Decision: Class Weights + Threshold Tuning over SMOTE
**Date:** 2025-01-29

**Context:** Handle 95:5 class imbalance (only 5% purchases).

**Options Considered:**
1. SMOTE - Oversampling minority class
2. Random Undersampling - Reduce majority class
3. Class Weights - Penalize misclassification differently
4. Threshold Tuning - Adjust decision boundary

**Decision:** Class weights + threshold tuning

**Rationale:**
- **Preserves Data**: Doesn't modify original training set
- **Computational Efficiency**: No synthetic data generation overhead
- **XGBoost Native**: `scale_pos_weight` parameter built-in
- **Business Alignment**: Threshold tuning optimizes for real metric (revenue)
- **Less Overfitting Risk**: SMOTE can create unrealistic samples

**Trade-offs:**
- May not work as well as SMOTE for extreme imbalance (99:1)
- Requires careful threshold selection

**Baseline:** Test all 4 approaches and compare PR-AUC.

---

## Testing Framework

### Decision: pytest over unittest
**Date:** 2025-01-29

**Context:** Need comprehensive testing with fixtures and parametrization.

**Options Considered:**
1. unittest - Standard library
2. pytest - Third-party, feature-rich
3. nose2 - unittest extensions

**Decision:** pytest

**Rationale:**
- **Better Assertions**: `assert x == y` vs `self.assertEqual(x, y)`
- **Fixtures**: Powerful setup/teardown with dependency injection
- **Parametrization**: Easy to test multiple inputs
- **Plugin Ecosystem**: coverage, asyncio, mock, benchmark
- **Output**: Clearer failure messages

**Trade-offs:**
- Extra dependency (but universally adopted)

---

## CI/CD

### Decision: GitHub Actions over Jenkins
**Date:** 2025-01-29

**Context:** Automate testing, linting, and Docker builds.

**Options Considered:**
1. GitHub Actions - Native, free for public repos
2. Jenkins - Self-hosted, flexible
3. GitLab CI - If using GitLab
4. CircleCI - Cloud-based

**Decision:** GitHub Actions

**Rationale:**
- **Native Integration**: Built into GitHub
- **Free**: 2000 minutes/month for private repos
- **Matrix Builds**: Test on multiple Python versions easily
- **Secrets Management**: Secure environment variables
- **Community Actions**: Reusable workflows (e.g., docker/build-push-action)

**Trade-offs:**
- Less flexible than Jenkins for complex pipelines
- Vendor lock-in to GitHub

---

## Deployment

### Decision: Docker Compose over Kubernetes (for now)
**Date:** 2025-01-29

**Context:** Deploy locally on MacBook, easily portable to cloud.

**Options Considered:**
1. Docker Compose - Simple orchestration
2. Kubernetes - Production-grade orchestration
3. Bare metal - No containers

**Decision:** Docker Compose

**Rationale:**
- **Simplicity**: Single `docker-compose.yml` file
- **Local Development**: Easy to run on laptop
- **Multi-service**: Orchestrate API, dashboard, MLflow together
- **Portable**: Same config works on AWS ECS, Azure Container Instances
- **Learning Curve**: Much simpler than Kubernetes

**Trade-offs:**
- No auto-scaling or health checks like Kubernetes
- Not suitable for large-scale production (100+ services)

**For Production Scale:** Migrate to Kubernetes with Helm charts.

---

## Summary Table

| Decision Area | Choice | Key Reason |
|---------------|--------|------------|
| Data Processing | DuckDB | OLAP-optimized, SQL interface |
| Feature Store | DuckDB + Parquet | Time-travel, simplicity |
| ML Framework | XGBoost | Best for tabular data |
| Experiment Tracking | MLflow | Open-source, self-hosted |
| API | FastAPI | Async, type-safe |
| Dashboard | Streamlit | Rapid prototyping |
| Imbalance | Class weights | Preserves data, efficient |
| Testing | pytest | Better DX, fixtures |
| CI/CD | GitHub Actions | Native integration |
| Deployment | Docker Compose | Simple, portable |

---

**Last Updated:** 2025-01-29  
**Author:** Neha Amin