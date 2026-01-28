

# 🛒 E-Commerce Purchase Prediction System

> Production-grade ML system for predicting user purchase intent in e-commerce applications

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project demonstrates **production-ready machine learning engineering** for predicting whether a user will purchase an item in their next session, based on:
- Clickstream events (view, scroll, add-to-cart, remove)
- Item metadata (price, category, discount)
- User behavioral history
- Highly imbalanced data (~5% purchase rate)

**Key Features:**
- ✅ Zero data leakage with point-in-time feature engineering
- ✅ Production-grade sessionization logic
- ✅ MLflow experiment tracking
- ✅ FastAPI serving layer
- ✅ Streamlit monitoring dashboard
- ✅ Docker-based deployment
- ✅ Comprehensive testing (>80% coverage)

## 🏗️ Architecture
```
┌─────────────────┐
│  Clickstream    │
│  Data Generator │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sessionization  │
│   Engine        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Feature Store   │◄─────┤   DuckDB     │
│ (Point-in-Time) │      └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Model Training │◄─────┤   MLflow     │
│  (XGBoost/LGB)  │      └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  FastAPI        │◄─────┤  Monitoring  │
│  Serving        │      │  Dashboard   │
└─────────────────┘      └──────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- macOS/Linux (Windows with WSL2)
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ecommerce-purchase-prediction
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e ".[dev]"
```

4. **Setup pre-commit hooks**
```bash
pre-commit install
```

## 📊 Project Structure
```
ecommerce-purchase-prediction/
├── src/
│   ├── data/           # Data generation & sessionization
│   ├── features/       # Feature engineering pipeline
│   ├── models/         # Model training & evaluation
│   └── evaluation/     # Metrics & validation
├── deployment/
│   ├── api/            # FastAPI serving
│   └── dashboard/      # Streamlit dashboard
├── tests/              # Unit, integration, e2e tests
├── notebooks/          # Analysis & exploration
├── docs/               # Documentation
├── configs/            # Configuration files
└── data/               # Data storage
```

## 🎯 Development Roadmap

- [x] Phase 0: Project setup & architecture
- [ ] Phase 1: Synthetic data generation
- [ ] Phase 2: Sessionization engine
- [ ] Phase 3: Feature engineering pipeline
- [ ] Phase 4: Model training
- [ ] Phase 5: Evaluation framework
- [ ] Phase 6: Model serving API
- [ ] Phase 7: Monitoring dashboard
- [ ] Phase 8: Testing & quality
- [ ] Phase 9: Deployment
- [ ] Phase 10: Documentation

## 📚 Documentation

Detailed documentation available in `/docs`:
- [Architecture Design](docs/architecture.md) *(coming soon)*
- [Data Generation Strategy](docs/data_generation.md) *(coming soon)*
- [Feature Engineering Guide](docs/feature_engineering.md) *(coming soon)*
- [Model Selection Rationale](docs/model_selection.md) *(coming soon)*

## 🧪 Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/
```

## 🔧 Tech Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Data Processing | DuckDB | OLAP-optimized, SQL interface, out-of-core |
| Feature Store | DuckDB + Parquet | Time-travel queries, columnar format |
| ML Framework | XGBoost, LightGBM | Industry standard for tabular data |
| Experiment Tracking | MLflow | Model versioning, metric tracking |
| API | FastAPI | Async, type-safe, auto-documentation |
| Dashboard | Streamlit | Rapid prototyping, interactive |
| Testing | pytest | Comprehensive plugin ecosystem |
| CI/CD | GitHub Actions | Native integration |

## 📈 Key Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| PR-AUC | > 0.35 | *TBD* |
| Precision@5% | > 0.30 | *TBD* |
| API Latency (p95) | < 100ms | *TBD* |
| Test Coverage | > 80% | *TBD* |

## 🤝 Contributing

This is a portfolio project demonstrating ML engineering best practices. Feedback and suggestions are welcome!

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 👤 Author

**Harit**
- M.S. Computer Science @ Northeastern University
- Research: Chess AI, Transformers, HPC Systems
- [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

Built with ❤️ for production ML excellence