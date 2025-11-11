#!/usr/bin/env python3
"""
Flask API for customer churn prediction.
Serves real-time predictions via REST API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

from feature_engineering import ChurnFeatureEngineer

app = Flask(__name__)
CORS(app)  # Enable CORS for web applications

# Global variables for model and feature engineer
model = None
feature_engineer = None
metadata = None

def load_model_artifacts():
    """Load model and feature engineer on startup."""
    global model, feature_engineer, metadata

    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'

    print("Loading model artifacts...")

    # Load model
    model_path = models_dir / 'churn_model.pkl'
    model = joblib.load(model_path)
    print(f"✅ Model loaded: {model_path}")

    # Load feature engineer
    fe_path = models_dir / 'feature_engineer.pkl'
    feature_engineer = ChurnFeatureEngineer()
    feature_engineer.load(fe_path)

    # Load metadata
    metadata_path = models_dir / 'model_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"✅ API ready!")
    print(f"   Model: {metadata['model_type']}")
    print(f"   Training date: {metadata['training_date']}")
    print(f"   AUC-ROC: {metadata['metrics']['auc_roc']:.3f}")

def get_risk_category(probability):
    """Convert probability to risk category."""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def get_recommendations(customer_data, probability):
    """Generate actionable recommendations based on churn probability and features."""
    recommendations = []
    actions = []

    # High churn risk
    if probability >= 0.7:
        recommendations.append("Immediate intervention required")

        if customer_data.get('contract_type') == 'Month-to-Month':
            actions.append("Offer annual contract discount (20% savings)")
        if customer_data.get('support_calls_30d', 0) > 3:
            actions.append("Assign dedicated account manager")
            actions.append("Review support ticket history")
        if customer_data.get('monthly_charges', 0) > 80:
            actions.append("Consider loyalty discount or service bundle")
        if customer_data.get('tenure_months', 0) < 12:
            actions.append("Onboarding success team outreach")

    # Medium churn risk
    elif probability >= 0.4:
        recommendations.append("Proactive engagement recommended")

        if customer_data.get('contract_type') == 'Month-to-Month':
            actions.append("Highlight benefits of longer-term contracts")
        if customer_data.get('support_calls_30d', 0) > 2:
            actions.append("Proactive support follow-up")
        actions.append("Send customer satisfaction survey")
        actions.append("Monitor usage patterns closely")

    # Low churn risk
    else:
        recommendations.append("Customer appears stable")
        actions.append("Continue standard retention activities")
        actions.append("Consider upsell opportunities")

    return recommendations, actions

def calculate_expected_value(probability):
    """Calculate expected value of intervention."""
    AVG_CUSTOMER_VALUE = 1200
    INTERVENTION_COST = 150
    SUCCESS_RATE = 0.60

    # Expected value if we intervene
    expected_save = probability * SUCCESS_RATE * AVG_CUSTOMER_VALUE
    expected_cost = INTERVENTION_COST
    expected_return = expected_save - expected_cost

    return {
        'customer_lifetime_value': AVG_CUSTOMER_VALUE,
        'intervention_cost': INTERVENTION_COST,
        'expected_return': round(expected_return, 2),
        'recommendation': 'INTERVENE' if expected_return > 0 else 'MONITOR'
    }

@app.route('/', methods=['GET'])
def home():
    """API home page with information."""
    return jsonify({
        'service': 'Customer Churn Prediction API',
        'version': '1.0.0',
        'model': metadata['model_type'] if metadata else 'Unknown',
        'status': 'ready',
        'endpoints': {
            '/predict': 'POST - Predict churn for single customer',
            '/predict/batch': 'POST - Predict churn for multiple customers',
            '/health': 'GET - Health check',
            '/model/info': 'GET - Model information and metrics'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Return model information and metrics."""
    if metadata is None:
        return jsonify({'error': 'Model metadata not available'}), 500

    return jsonify({
        'model_type': metadata['model_type'],
        'training_date': metadata['training_date'],
        'version': metadata['version'],
        'metrics': {
            'accuracy': metadata['metrics']['accuracy'],
            'auc_roc': metadata['metrics']['auc_roc']
        },
        'top_features': metadata['metrics'].get('feature_importance', [])[:10]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn for a single customer.

    Expected JSON input:
    {
        "customer_id": "CUST-12345",
        "tenure_months": 24,
        "monthly_charges": 89.99,
        "total_charges": 2159.76,
        "contract_type": "Month-to-Month",
        "support_calls_30d": 5,
        "data_usage_gb": 12.5,
        "payment_method": "Electronic check",
        "streaming_services": 2,
        "internet_service": "Fiber optic",
        "has_dependents": 0,
        "is_senior": 0,
        "paperless_billing": 1,
        "has_phone_service": 1,
        "has_multiple_lines": 0
    }
    """
    try:
        # Get customer data
        customer_data = request.get_json()

        if not customer_data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract customer ID
        customer_id = customer_data.get('customer_id', 'UNKNOWN')

        # Convert to DataFrame
        df = pd.DataFrame([customer_data])

        # Remove customer_id from features if present
        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)

        # Prepare features
        X, _ = feature_engineer.prepare_features(df, fit=False)

        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]

        # Risk score (0-100)
        risk_score = int(probability * 100)

        # Risk category
        risk_category = get_risk_category(probability)

        # Recommendations
        recommendations, actions = get_recommendations(customer_data, probability)

        # Expected value
        expected_value = calculate_expected_value(probability)

        # Response
        response = {
            'customer_id': customer_id,
            'prediction': {
                'will_churn': bool(prediction),
                'churn_probability': round(float(probability), 3),
                'risk_score': risk_score,
                'risk_category': risk_category
            },
            'recommendations': recommendations,
            'suggested_actions': actions,
            'expected_value': expected_value,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate prediction'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict churn for multiple customers.

    Expected JSON input:
    {
        "customers": [
            {...customer_data...},
            {...customer_data...}
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'customers' not in data:
            return jsonify({'error': 'No customers data provided'}), 400

        customers = data['customers']

        if not isinstance(customers, list):
            return jsonify({'error': 'Customers must be a list'}), 400

        # Extract customer IDs
        customer_ids = [c.get('customer_id', f'UNKNOWN-{i}') for i, c in enumerate(customers)]

        # Convert to DataFrame
        df = pd.DataFrame(customers)

        # Remove customer_id from features
        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)

        # Prepare features
        X, _ = feature_engineer.prepare_features(df, fit=False)

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Build response
        results = []
        for i, (cid, pred, prob) in enumerate(zip(customer_ids, predictions, probabilities)):
            results.append({
                'customer_id': cid,
                'will_churn': bool(pred),
                'churn_probability': round(float(prob), 3),
                'risk_score': int(prob * 100),
                'risk_category': get_risk_category(prob)
            })

        response = {
            'total_customers': len(results),
            'predictions': results,
            'summary': {
                'high_risk': sum(1 for r in results if r['risk_category'] == 'HIGH'),
                'medium_risk': sum(1 for r in results if r['risk_category'] == 'MEDIUM'),
                'low_risk': sum(1 for r in results if r['risk_category'] == 'LOW')
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate batch predictions'
        }), 500

if __name__ == '__main__':
    # Load model artifacts
    load_model_artifacts()

    # Start Flask server
    print("\n" + "="*60)
    print("🚀 Starting Customer Churn Prediction API")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /            - API information")
    print("  GET  /health      - Health check")
    print("  GET  /model/info  - Model details")
    print("  POST /predict     - Single prediction")
    print("  POST /predict/batch - Batch predictions")
    print("\n" + "="*60)
    print("API running at: http://localhost:5000")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
