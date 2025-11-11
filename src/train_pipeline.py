#!/usr/bin/env python3
"""
Model training pipeline for customer churn prediction.
Trains multiple models, performs hyperparameter tuning, and saves the best model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score
)
import xgboost as xgb

from feature_engineering import ChurnFeatureEngineer

class ChurnModelTrainer:
    """
    Orchestrates the model training pipeline.
    """

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.models_dir = self.data_path.parent.parent / 'models'
        self.models_dir.mkdir(exist_ok=True)

        self.feature_engineer = ChurnFeatureEngineer()
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}

    def load_and_prepare_data(self):
        """Load data and prepare train/test splits."""
        print("Loading data...")
        df = pd.read_csv(self.data_path)

        # Separate features and target
        y = df['churned'].values
        X_df = df.drop('churned', axis=1)

        # Engineer features
        print("Engineering features...")
        X, feature_names = self.feature_engineer.prepare_features(X_df, fit=True)

        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"✅ Data prepared:")
        print(f"  Train set: {X_train.shape[0]:,} samples")
        print(f"  Test set: {X_test.shape[0]:,} samples")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Churn rate (train): {y_train.mean():.1%}")
        print(f"  Churn rate (test): {y_test.mean():.1%}")

        return X_train, X_test, y_train, y_test, feature_names

    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models for comparison."""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)

        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        }

        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_roc': auc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            }

            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  AUC-ROC: {auc:.3f}")
            print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        return results

    def tune_best_model(self, X_train, y_train):
        """Hyperparameter tuning for Random Forest (typically best for tabular data)."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING (Random Forest)")
        print("="*60)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'class_weight': ['balanced']
        }

        rf = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )

        print("Running grid search (this may take a few minutes)...")
        grid_search.fit(X_train, y_train)

        print(f"\n✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV AUC-ROC: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_

    def evaluate_final_model(self, model, X_test, y_test, feature_names):
        """Comprehensive evaluation of the final model."""
        print("\n" + "="*60)
        print("FINAL MODEL EVALUATION")
        print("="*60)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n📊 Performance Metrics:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  AUC-ROC: {auc:.3f}")

        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

        print(f"\n🔢 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                Predicted")
        print(f"                Retained  Churned")
        print(f"  Actual Retained  {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"         Churned   {cm[1,0]:6d}    {cm[1,1]:6d}")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n🎯 Top 10 Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']:<30s}: {row['importance']:.4f}")

            self.metrics['feature_importance'] = feature_importance.to_dict('records')

        # Save metrics
        self.metrics.update({
            'accuracy': float(accuracy),
            'auc_roc': float(auc),
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        })

        return y_pred, y_pred_proba

    def save_model(self, model, model_name='churn_model'):
        """Save the trained model and artifacts."""
        print(f"\n💾 Saving model artifacts...")

        # Save model
        model_path = self.models_dir / f'{model_name}.pkl'
        joblib.dump(model, model_path)
        print(f"  ✅ Model saved: {model_path}")

        # Save feature engineer
        fe_path = self.models_dir / 'feature_engineer.pkl'
        self.feature_engineer.save(fe_path)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'metrics': self.metrics,
            'training_date': datetime.now().isoformat(),
            'version': '1.0.0'
        }

        metadata_path = self.models_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Metadata saved: {metadata_path}")

    def train(self):
        """Execute the full training pipeline."""
        print("\n" + "="*60)
        print("CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
        print("="*60)

        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_names = self.load_and_prepare_data()

        # Train baseline models
        baseline_results = self.train_baseline_models(X_train, X_test, y_train, y_test)

        # Find best baseline model
        best_baseline = max(baseline_results.items(), key=lambda x: x[1]['auc_roc'])
        print(f"\n🏆 Best baseline model: {best_baseline[0]} (AUC: {best_baseline[1]['auc_roc']:.3f})")

        # Hyperparameter tuning on Random Forest
        tuned_model = self.tune_best_model(X_train, y_train)

        # Final evaluation
        y_pred, y_pred_proba = self.evaluate_final_model(
            tuned_model, X_test, y_test, feature_names
        )

        # Save model
        self.best_model = tuned_model
        self.save_model(tuned_model)

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel artifacts saved in: {self.models_dir}")
        print("\nNext steps:")
        print("  1. Run src/evaluate_model.py for detailed analysis")
        print("  2. Start the API: python src/api.py")
        print("  3. Test predictions with data/sample_customer.json")

def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'customer_churn_data.csv'

    if not data_path.exists():
        print("❌ Error: Data file not found!")
        print(f"  Expected: {data_path}")
        print("  Run: python scripts/generate_sample_data.py")
        return

    trainer = ChurnModelTrainer(data_path)
    trainer.train()

if __name__ == "__main__":
    main()
