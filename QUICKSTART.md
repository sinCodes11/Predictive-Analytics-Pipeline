# Quick Start Guide

Get the churn prediction pipeline running in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Installation

```bash
# 1. Navigate to project directory
cd ~/Desktop/predictive-analytics-pipeline

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Sample Data
```bash
python scripts/generate_sample_data.py
```

**Output:**
```
Generating 10,000 customer records...
Dataset Statistics:
  Total customers: 10,000
  Churned customers: 2,347 (23.5%)
  Retained customers: 7,653 (76.5%)

✅ Data generation complete!
```

### Step 2: Train the Model
```bash
python src/train_pipeline.py
```

**Output (takes 2-3 minutes):**
```
CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE

Training Logistic Regression...
  Accuracy: 0.841
  AUC-ROC: 0.897

Training Random Forest...
  Accuracy: 0.872
  AUC-ROC: 0.921

Training XGBoost...
  Accuracy: 0.868
  AUC-ROC: 0.918

🏆 Best baseline model: Random Forest (AUC: 0.921)

HYPERPARAMETER TUNING (Random Forest)
Running grid search...

✅ TRAINING COMPLETE!
```

### Step 3: Evaluate Model
```bash
python src/evaluate_model.py
```

**Output:**
```
MODEL EVALUATION & BUSINESS ANALYSIS

Technical Metrics:
  Accuracy: 87%
  AUC-ROC: 0.92

Business Impact:
  Revenue protected: $336,960.00
  Intervention costs: $85,500.00
  Net value: $251,460.00
  ROI: 9,234%

✅ EVALUATION COMPLETE
```

### Step 4: Start the API
```bash
python src/api.py
```

**Output:**
```
🚀 Starting Customer Churn Prediction API

Endpoints:
  GET  /            - API information
  GET  /health      - Health check
  POST /predict     - Single prediction
  POST /predict/batch - Batch predictions

API running at: http://localhost:5000
```

### Step 5: Test the API

**In a new terminal:**
```bash
# Test with sample customer
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @data/sample_customer.json
```

**Response:**
```json
{
  "customer_id": "CUST-000123",
  "prediction": {
    "will_churn": true,
    "churn_probability": 0.847,
    "risk_score": 85,
    "risk_category": "HIGH"
  },
  "recommendations": [
    "Immediate intervention required"
  ],
  "suggested_actions": [
    "Offer annual contract discount (20% savings)",
    "Assign dedicated account manager",
    "Review support ticket history"
  ],
  "expected_value": {
    "customer_lifetime_value": 1200,
    "intervention_cost": 150,
    "expected_return": 461.76
  }
}
```

## What's Next?

### For Local Development
- Explore Jupyter notebooks: `jupyter notebook notebooks/`
- Modify features in `src/feature_engineering.py`
- Tune hyperparameters in `src/train_pipeline.py`

### For Cloud Deployment
- See `docs/OCI_SETUP.md` for OCI deployment
- Run `scripts/deploy_oci.sh` to upload models to cloud

### For Your ePortfolio
- Read `docs/EPORTFOLIO_GUIDE.md` for presentation tips
- Take screenshots of API responses
- Document your deployment process

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"
```bash
# Install dependencies
pip install -r requirements.txt
```

### "FileNotFoundError: customer_churn_data.csv"
```bash
# Generate data first
python scripts/generate_sample_data.py
```

### "Model not found"
```bash
# Train model first
python src/train_pipeline.py
```

### API returns 500 error
```bash
# Check if model is trained
ls models/
# Should see: churn_model.pkl, feature_engineer.pkl, model_metadata.json

# If missing, train the model
python src/train_pipeline.py
```

## Project Structure Explained

```
predictive-analytics-pipeline/
├── data/                    # Dataset and sample files
│   ├── customer_churn_data.csv
│   └── sample_customer.json
├── models/                  # Trained models (created after training)
│   ├── churn_model.pkl
│   ├── feature_engineer.pkl
│   └── model_metadata.json
├── src/                     # Source code
│   ├── feature_engineering.py  # Feature creation
│   ├── train_pipeline.py       # Model training
│   ├── evaluate_model.py       # Model evaluation
│   └── api.py                  # Flask API
├── scripts/                 # Utility scripts
│   ├── generate_sample_data.py
│   └── deploy_oci.sh
├── docs/                    # Documentation
│   ├── OCI_SETUP.md
│   └── EPORTFOLIO_GUIDE.md
└── requirements.txt         # Python dependencies
```

## Common Commands

```bash
# Generate fresh dataset
python scripts/generate_sample_data.py

# Train new model
python src/train_pipeline.py

# Evaluate model performance
python src/evaluate_model.py

# Start API server
python src/api.py

# Test API health
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/model/info

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "TEST-001",
    "tenure_months": 6,
    "monthly_charges": 89.99,
    "total_charges": 539.94,
    "contract_type": "Month-to-Month",
    "support_calls_30d": 4,
    "data_usage_gb": 15.5,
    "payment_method": "Electronic check",
    "streaming_services": 2,
    "internet_service": "Fiber optic",
    "has_dependents": 0,
    "is_senior": 0,
    "paperless_billing": 1,
    "has_phone_service": 1,
    "has_multiple_lines": 0
  }'
```

## Resources

- **Full Documentation**: See `README.md`
- **OCI Deployment**: See `docs/OCI_SETUP.md`
- **ePortfolio Guide**: See `docs/EPORTFOLIO_GUIDE.md`
- **GitHub**: [Your repository URL]

## Support

Having issues? Check:
1. Python version: `python --version` (need 3.9+)
2. Dependencies installed: `pip list | grep scikit-learn`
3. Data generated: `ls data/customer_churn_data.csv`
4. Model trained: `ls models/churn_model.pkl`

---

**Estimated Time to Complete:**
- Data generation: 30 seconds
- Model training: 2-3 minutes
- Evaluation: 10 seconds
- API startup: 5 seconds

**Total: ~5 minutes** to go from zero to working ML API! 🚀
