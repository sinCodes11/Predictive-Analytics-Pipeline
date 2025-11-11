#!/usr/bin/env python3
"""
Model evaluation and business metrics calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    classification_report, confusion_matrix
)

from feature_engineering import ChurnFeatureEngineer

class ChurnModelEvaluator:
    """
    Comprehensive model evaluation with business metrics.
    """

    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_engineer = None
        self.metadata = None

    def load_model(self):
        """Load trained model and artifacts."""
        print("Loading model artifacts...")

        # Load model
        model_path = self.models_dir / 'churn_model.pkl'
        self.model = joblib.load(model_path)
        print(f"  ✅ Model loaded: {model_path}")

        # Load feature engineer
        fe_path = self.models_dir / 'feature_engineer.pkl'
        self.feature_engineer = ChurnFeatureEngineer()
        self.feature_engineer.load(fe_path)

        # Load metadata
        metadata_path = self.models_dir / 'model_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"  ✅ Metadata loaded")
        print(f"  Model type: {self.metadata['model_type']}")
        print(f"  Training date: {self.metadata['training_date']}")

    def calculate_business_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate business-relevant metrics.

        Assumptions:
        - Average customer value: $1,200/year
        - Retention campaign cost: $150/customer
        - Success rate of retention: 60%
        """

        # Constants
        AVG_CUSTOMER_VALUE = 1200  # Annual value
        INTERVENTION_COST = 150    # Cost per retention attempt
        INTERVENTION_SUCCESS_RATE = 0.60

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate business outcomes
        true_positives_saved = tp * INTERVENTION_SUCCESS_RATE
        false_positives_cost = fp * INTERVENTION_COST
        false_negatives_lost = fn * AVG_CUSTOMER_VALUE
        true_negatives_value = tn  # No action needed, no cost

        # Revenue calculations
        intervention_cost_total = (tp + fp) * INTERVENTION_COST
        revenue_protected = true_positives_saved * AVG_CUSTOMER_VALUE
        revenue_lost = false_negatives_lost

        # Net value
        net_value = revenue_protected - intervention_cost_total - revenue_lost

        # Without model (baseline: no intervention)
        baseline_revenue_lost = (tp + fn) * AVG_CUSTOMER_VALUE

        # Value created by model
        value_created = baseline_revenue_lost - (intervention_cost_total + revenue_lost)

        metrics = {
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'business_outcomes': {
                'customers_saved': int(true_positives_saved),
                'unnecessary_interventions': int(fp),
                'missed_churners': int(fn),
                'correctly_retained': int(tn)
            },
            'financial_impact': {
                'intervention_cost': intervention_cost_total,
                'revenue_protected': revenue_protected,
                'revenue_lost': revenue_lost,
                'net_value': net_value,
                'baseline_loss_without_model': baseline_revenue_lost,
                'value_created_by_model': value_created
            },
            'roi': {
                'return_on_investment': (value_created / intervention_cost_total) * 100 if intervention_cost_total > 0 else 0,
                'cost_per_customer_saved': intervention_cost_total / true_positives_saved if true_positives_saved > 0 else 0
            }
        }

        return metrics

    def print_business_report(self, business_metrics):
        """Print formatted business metrics report."""
        print("\n" + "="*60)
        print("BUSINESS IMPACT ANALYSIS")
        print("="*60)

        cm = business_metrics['confusion_matrix']
        bo = business_metrics['business_outcomes']
        fi = business_metrics['financial_impact']
        roi = business_metrics['roi']

        print(f"\n📊 Model Performance:")
        print(f"  Customers correctly identified as churn risk: {cm['true_positives']:,}")
        print(f"  Customers missed (false negatives): {cm['false_negatives']:,}")
        print(f"  Unnecessary interventions (false positives): {cm['false_positives']:,}")

        print(f"\n💰 Financial Impact:")
        print(f"  Revenue protected: ${fi['revenue_protected']:,.2f}")
        print(f"  Intervention costs: ${fi['intervention_cost']:,.2f}")
        print(f"  Revenue lost (missed): ${fi['revenue_lost']:,.2f}")
        print(f"  Net value: ${fi['net_value']:,.2f}")

        print(f"\n📈 Value Created:")
        print(f"  Baseline loss (no model): ${fi['baseline_loss_without_model']:,.2f}")
        print(f"  Value created by model: ${fi['value_created_by_model']:,.2f}")
        print(f"  ROI: {roi['return_on_investment']:.1f}%")
        print(f"  Cost per customer saved: ${roi['cost_per_customer_saved']:.2f}")

        print(f"\n🎯 Business Recommendations:")
        precision = cm['true_positives'] / (cm['true_positives'] + cm['false_positives'])
        recall = cm['true_positives'] / (cm['true_positives'] + cm['false_negatives'])

        if precision < 0.7:
            print("  ⚠️  High false positive rate - consider raising prediction threshold")
        if recall < 0.75:
            print("  ⚠️  Missing too many churners - consider lowering prediction threshold")
        if roi['return_on_investment'] > 500:
            print("  ✅ Excellent ROI - scale up retention campaigns")
        if roi['return_on_investment'] > 1000:
            print("  🚀 Outstanding ROI - invest in model improvements for even better results")

    def evaluate(self, data_path):
        """Run full evaluation pipeline."""
        print("\n" + "="*60)
        print("MODEL EVALUATION & BUSINESS ANALYSIS")
        print("="*60)

        # Load model
        self.load_model()

        # Load test data
        print("\nLoading test data...")
        df = pd.read_csv(data_path)

        # Prepare features
        y_true = df['churned'].values
        X_df = df.drop('churned', axis=1)
        X, _ = self.feature_engineer.prepare_features(X_df, fit=False)

        # Make predictions
        print("Generating predictions...")
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Technical metrics
        print("\n" + "="*60)
        print("TECHNICAL METRICS")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Retained', 'Churned']))

        # Business metrics
        business_metrics = self.calculate_business_metrics(y_true, y_pred, y_pred_proba)
        self.print_business_report(business_metrics)

        # Risk distribution
        self.analyze_risk_distribution(y_pred_proba)

        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE")
        print("="*60)

    def analyze_risk_distribution(self, y_pred_proba):
        """Analyze distribution of churn risk scores."""
        print("\n" + "="*60)
        print("RISK SCORE DISTRIBUTION")
        print("="*60)

        # Create risk categories
        high_risk = (y_pred_proba >= 0.7).sum()
        medium_risk = ((y_pred_proba >= 0.4) & (y_pred_proba < 0.7)).sum()
        low_risk = (y_pred_proba < 0.4).sum()

        total = len(y_pred_proba)

        print(f"\n🎯 Customer Segmentation:")
        print(f"  High Risk (≥70%):    {high_risk:,} customers ({high_risk/total*100:.1f}%)")
        print(f"  Medium Risk (40-70%): {medium_risk:,} customers ({medium_risk/total*100:.1f}%)")
        print(f"  Low Risk (<40%):      {low_risk:,} customers ({low_risk/total*100:.1f}%)")

        print(f"\n💡 Action Plan:")
        print(f"  Priority 1 (High Risk): Immediate intervention with premium offers")
        print(f"  Priority 2 (Medium Risk): Proactive engagement, monitor closely")
        print(f"  Priority 3 (Low Risk): Standard retention, no special action needed")

def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    data_path = project_root / 'data' / 'customer_churn_data.csv'

    # Check if model exists
    if not (models_dir / 'churn_model.pkl').exists():
        print("❌ Error: Model not found!")
        print("  Run: python src/train_pipeline.py")
        return

    evaluator = ChurnModelEvaluator(models_dir)
    evaluator.evaluate(data_path)

if __name__ == "__main__":
    main()
