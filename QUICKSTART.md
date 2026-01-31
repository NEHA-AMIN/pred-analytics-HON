# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Fix Dependencies (Important!)

There's a NumPy version compatibility issue. Fix it first:

```bash
# Activate virtual environment
source venv/bin/activate

# Downgrade NumPy to compatible version
pip install "numpy<2.0"
pip install --upgrade scipy scikit-learn
```

### Step 2: Run the Pipeline

Generate data and train the model:

```bash
# Full pipeline (takes ~3-5 minutes)
python run_pipeline.py --full
```

This will:
- ✅ Generate 10,000 users and 1,000 items
- ✅ Simulate 90 days of clickstream data
- ✅ Create sessions and temporal splits
- ✅ Engineer features
- ✅ Train XGBoost model
- ✅ Save model to `models/` directory

### Step 3: Start the API

```bash
# Start FastAPI server
cd deployment/api
python main.py
```

The API will be available at:
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### Step 4: Test the API

In a new terminal:

```bash
# Test the API
python deployment/api/test_client.py
```

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [{
      "user_id": "user_test",
      "item_id": "item_test",
      "item_price": 89.99,
      "item_base_price": 129.99,
      "item_discount_pct": 30.0,
      "item_category": "Electronics"
    }],
    "return_probabilities": true
  }'
```

---

## 📚 Common Commands

### Pipeline Commands

```bash
# Full pipeline
python run_pipeline.py --full

# Individual stages
python run_pipeline.py --data-gen
python run_pipeline.py --sessionize
python run_pipeline.py --features
python run_pipeline.py --train

# Skip data generation (use existing data)
python run_pipeline.py --skip-data-gen --full

# Get help
python run_pipeline.py --help
```

### API Commands

```bash
# Start API (development)
python deployment/api/main.py

# Start with uvicorn (more control)
uvicorn deployment.api.main:app --reload --port 8000

# Run tests
pytest deployment/api/test_api.py -v

# Test client
python deployment/api/test_client.py
```

---

## 📁 Where to Find Things

| What | Where |
|------|-------|
| Pipeline orchestrator | `run_pipeline.py` |
| API application | `deployment/api/main.py` |
| Generated data | `data/raw/`, `data/processed/`, `data/features/` |
| Trained models | `models/` |
| Pipeline logs | `pipeline.log` |
| API documentation | http://localhost:8000/docs |
| Pipeline guide | `docs/PIPELINE_GUIDE.md` |
| API guide | `deployment/api/README.md` |
| Implementation summary | `IMPLEMENTATION_SUMMARY.md` |

---

## 🔧 Troubleshooting

### NumPy Error
```
AttributeError: _ARRAY_API not found
```
**Fix**: `pip install "numpy<2.0"`

### Model Not Loading
```
503 Service Unavailable: Model not loaded
```
**Fix**: Run pipeline first: `python run_pipeline.py --full`

### Port Already in Use
```
Error: Address already in use
```
**Fix**: Use different port: `uvicorn deployment.api.main:app --port 8001`

---

## 🎯 Next Steps

1. ✅ Fix NumPy dependency
2. ✅ Run pipeline to generate model
3. ✅ Start API and test predictions
4. ⬜ Explore API documentation at http://localhost:8000/docs
5. ⬜ Read detailed guides in `docs/` directory
6. ⬜ Customize configuration in `configs/` directory

---

## 💡 Tips

- **Faster iteration**: Use `--skip-data-gen` to skip data generation
- **Smaller dataset**: Edit `configs/data_generation.yaml` to reduce `num_users` and `num_days`
- **API testing**: Use the interactive docs at http://localhost:8000/docs
- **Logs**: Check `pipeline.log` for detailed execution logs

---

## 📖 Documentation

- **Pipeline Guide**: `docs/PIPELINE_GUIDE.md` - Comprehensive pipeline documentation
- **API Guide**: `deployment/api/README.md` - API usage and examples
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md` - What was built and why

---

**Need Help?** Check the documentation or review the logs!
