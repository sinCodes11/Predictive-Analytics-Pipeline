# Predictive Analytics Pipeline — Customer Churn Prediction

<div align="center">
  <img src="assets/banner.svg" width="100%" />
</div>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-ML-orange?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/OCI-Free%20Tier-red?style=flat-square" alt="OCI">
  <img src="https://img.shields.io/badge/Flask-API-green?style=flat-square" alt="Flask">
</p>

End-to-end ML pipeline for predicting customer churn, deployed on Oracle Cloud Infrastructure using **100% free tier resources**. Predicts which customers churn in the next 30 days so retention effort goes where it's actually needed.

## Problem

Acquiring a new customer costs 5–25x more than retaining an existing one. Most companies find out a customer churned after the fact. This pipeline catches them before they leave.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OCI Free Tier Components                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Compute    │    │  Object      │    │   Autonomous │  │
│  │   Instance   │───▶│  Storage     │───▶│   Database   │  │
│  │  (E2.1.Micro)│    │  (50GB free) │    │   (20GB free)│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Data Ingestion & Validation                      │  │
│  │  2. Feature Engineering (15+ features)              │  │
│  │  3. Model Training (RF, LogReg, XGBoost comparison) │  │
│  │  4. Evaluation & Monitoring                         │  │
│  │  5. Prediction API (Flask, <100ms)                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87% |
| AUC-ROC | 0.92 |
| Precision | 84% |
| Recall | 79% |
| F1-Score | 0.81 |

**Top feature drivers:** customer tenure (23%), monthly charges (19%), support call volume (15%), contract type (12%), recent usage patterns (11%).

Optimized for recall — missing a churner costs more than a false positive retention offer.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py

# Train
python src/train_pipeline.py

# Evaluate
python src/evaluate_model.py

# Start API
python src/api.py
# http://localhost:5000
```

## API Example

```bash
POST /predict
Content-Type: application/json

{
  "customer_id": "CUST-12345",
  "tenure_months": 24,
  "monthly_charges": 89.99,
  "contract_type": "Month-to-Month",
  "support_calls_30d": 5,
  "data_usage_gb": 12.5
}
```

```json
{
  "customer_id": "CUST-12345",
  "churn_probability": 0.73,
  "risk_score": 73,
  "risk_category": "HIGH",
  "recommendation": "Immediate intervention required",
  "suggested_actions": [
    "Offer annual contract discount (20% savings)",
    "Assign dedicated account manager",
    "Review support ticket history"
  ]
}
```

## Project Structure

```
predictive-analytics-pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_customer.json
├── models/
│   ├── churn_model.pkl
│   ├── feature_scaler.pkl
│   └── model_metadata.json
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train_pipeline.py
│   ├── evaluate_model.py
│   └── api.py
├── scripts/
│   ├── generate_sample_data.py
│   ├── deploy_oci.sh
│   └── monitor_model.py
└── docs/
    ├── OCI_SETUP.md
    └── API_DOCUMENTATION.md
```

## OCI Free Tier Resources Used

| Resource | Spec |
|----------|------|
| Compute | VM.Standard.E2.1.Micro (2 instances) |
| Block Storage | 100 GB |
| Object Storage | 10 GB |
| Autonomous DB | 2x 20 GB |
| Outbound Transfer | 10 TB/month |

## Cost

- Development: **$0** (all free tier)
- Production at 10K predictions/day: ~$50/month

## Roadmap

- [ ] A/B testing framework for model variants
- [ ] Drift detection and auto-retraining
- [ ] Streamlit dashboard for business users
- [ ] SHAP explainability for individual predictions
- [ ] CRM integration (Salesforce API)

## License

MIT

## Author

**Daniel Gregg Jr**
- GitHub: [@sinCodes11](https://github.com/sinCodes11)
- Portfolio: [daniel-eportfolio.web.app](https://daniel-eportfolio.web.app)
- LinkedIn: [linkedin.com/in/daniel-sin-1881ske89](https://linkedin.com/in/daniel-sin-1881ske89)
