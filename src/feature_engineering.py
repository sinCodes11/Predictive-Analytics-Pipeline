"""
Feature engineering module for customer churn prediction.
Creates derived features and handles preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class ChurnFeatureEngineer:
    """
    Feature engineering pipeline for churn prediction.
    Handles feature creation, encoding, and scaling.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def create_features(self, df):
        """
        Create engineered features from raw data.

        Args:
            df: DataFrame with raw customer data

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # 1. Monetary features
        df['avg_monthly_charge'] = df['total_charges'] / (df['tenure_months'] + 1)
        df['charge_to_tenure_ratio'] = df['monthly_charges'] / (df['tenure_months'] + 1)

        # 2. Engagement features
        df['services_per_month'] = df['streaming_services'] / (df['tenure_months'] + 1)
        df['total_services'] = (
            df['streaming_services'] +
            df['has_phone_service'] +
            df['has_multiple_lines'] +
            (df['internet_service'] != 'No').astype(int)
        )

        # 3. Support & satisfaction proxy
        df['support_intensity'] = df['support_calls_30d'] / (df['tenure_months'] + 1)
        df['high_support_calls'] = (df['support_calls_30d'] > 3).astype(int)

        # 4. Contract stability
        df['is_month_to_month'] = (df['contract_type'] == 'Month-to-Month').astype(int)
        df['is_long_contract'] = (df['contract_type'] == 'Two Year').astype(int)

        # 5. Customer lifecycle stage
        df['customer_stage'] = pd.cut(
            df['tenure_months'],
            bins=[0, 6, 24, 72],
            labels=['New', 'Growing', 'Mature']
        )

        # 6. Payment reliability
        df['unreliable_payment'] = (df['payment_method'] == 'Electronic check').astype(int)

        # 7. Usage patterns
        df['high_data_user'] = (df['data_usage_gb'] > df['data_usage_gb'].median()).astype(int)
        df['data_per_dollar'] = df['data_usage_gb'] / (df['monthly_charges'] + 1)

        # 8. Customer value estimate
        df['estimated_clv'] = df['monthly_charges'] * (72 - df['tenure_months'])

        # 9. Risk flags (multiple indicators)
        df['risk_flags'] = (
            df['is_month_to_month'] +
            (df['tenure_months'] < 6).astype(int) +
            df['high_support_calls'] +
            df['unreliable_payment'] +
            (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
        )

        return df

    def prepare_features(self, df, fit=True):
        """
        Prepare features for model training/prediction.

        Args:
            df: DataFrame with engineered features
            fit: If True, fit the encoders and scaler. If False, use existing.

        Returns:
            Feature matrix (numpy array), feature names
        """
        df = self.create_features(df)

        # Categorical columns to encode
        categorical_cols = [
            'contract_type', 'payment_method', 'internet_service', 'customer_stage'
        ]

        # Numerical columns
        numerical_cols = [
            'tenure_months', 'monthly_charges', 'total_charges',
            'support_calls_30d', 'data_usage_gb', 'streaming_services',
            'avg_monthly_charge', 'charge_to_tenure_ratio',
            'services_per_month', 'total_services', 'support_intensity',
            'data_per_dollar', 'estimated_clv', 'risk_flags'
        ]

        # Binary columns (already 0/1)
        binary_cols = [
            'has_dependents', 'is_senior', 'paperless_billing',
            'has_phone_service', 'has_multiple_lines',
            'high_support_calls', 'is_month_to_month', 'is_long_contract',
            'unreliable_payment', 'high_data_user'
        ]

        # Encode categorical variables
        encoded_cols = []
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
            encoded_cols.append(f'{col}_encoded')

        # Combine all feature columns
        feature_cols = numerical_cols + binary_cols + encoded_cols
        X = df[feature_cols].values

        # Scale numerical features
        if fit:
            X = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            X = self.scaler.transform(X)

        return X, feature_cols

    def save(self, path):
        """Save feature engineer (scaler and encoders)."""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
        print(f"Feature engineer saved to {path}")

    def load(self, path):
        """Load feature engineer (scaler and encoders)."""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        print(f"Feature engineer loaded from {path}")

if __name__ == "__main__":
    # Test feature engineering
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data' / 'customer_churn_data.csv'

    if data_file.exists():
        print("Testing feature engineering...")
        df = pd.read_csv(data_file)
        fe = ChurnFeatureEngineer()
        X, feature_names = fe.prepare_features(df, fit=True)

        print(f"\n✅ Feature engineering successful!")
        print(f"  Original features: {len(df.columns)}")
        print(f"  Engineered features: {X.shape[1]}")
        print(f"  Sample count: {X.shape[0]:,}")
        print(f"\nTop 10 features:")
        for i, name in enumerate(feature_names[:10]):
            print(f"  {i+1}. {name}")
    else:
        print("❌ Data file not found. Run generate_sample_data.py first.")
