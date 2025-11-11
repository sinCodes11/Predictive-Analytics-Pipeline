#!/usr/bin/env python3
"""
Generate synthetic customer churn dataset for training and testing.
Creates realistic customer behavior patterns with known churn indicators.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_customer_data(n_customers=10000):
    """
    Generate synthetic customer data with realistic churn patterns.

    Churn Indicators (weighted by importance):
    1. Short tenure (< 6 months) - 23%
    2. High monthly charges - 19%
    3. Many support calls - 15%
    4. Month-to-month contract - 12%
    5. Low engagement - 11%
    """

    print(f"Generating {n_customers} customer records...")

    # Customer ID
    customer_ids = [f"CUST-{str(i).zfill(6)}" for i in range(1, n_customers + 1)]

    # Tenure (months) - bimodal distribution (new vs loyal customers)
    tenure_new = np.random.exponential(scale=4, size=int(n_customers * 0.3))
    tenure_loyal = np.random.normal(loc=36, scale=15, size=int(n_customers * 0.7))
    tenure = np.concatenate([tenure_new, tenure_loyal])
    tenure = np.clip(tenure, 0, 72).astype(int)
    np.random.shuffle(tenure)

    # Contract type (correlated with tenure)
    contract_type = []
    for t in tenure:
        if t < 6:
            contract_type.append(np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                                  p=[0.8, 0.15, 0.05]))
        elif t < 24:
            contract_type.append(np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                                  p=[0.4, 0.4, 0.2]))
        else:
            contract_type.append(np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                                  p=[0.2, 0.3, 0.5]))

    # Monthly charges (higher for newer customers)
    base_charge = np.random.uniform(20, 120, n_customers)
    tenure_discount = (tenure / 72) * 20  # Loyal customers get discounts
    monthly_charges = np.clip(base_charge - tenure_discount, 20, 120)

    # Total charges (tenure * average monthly charge)
    total_charges = tenure * (monthly_charges * np.random.uniform(0.9, 1.1, n_customers))

    # Support calls (proxy for dissatisfaction)
    support_calls_30d = np.random.poisson(lam=1.5, size=n_customers)

    # Data usage (GB per month)
    data_usage_gb = np.random.lognormal(mean=2, sigma=1, size=n_customers)
    data_usage_gb = np.clip(data_usage_gb, 0.5, 100)

    # Payment method
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        size=n_customers,
        p=[0.35, 0.15, 0.25, 0.25]
    )

    # Number of streaming services (0-3)
    streaming_services = np.random.choice([0, 1, 2, 3], size=n_customers,
                                         p=[0.3, 0.3, 0.25, 0.15])

    # Internet service type
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'],
                                       size=n_customers, p=[0.4, 0.5, 0.1])

    # Has dependents (affects churn - more stable)
    has_dependents = np.random.choice([0, 1], size=n_customers, p=[0.6, 0.4])

    # Senior citizen
    is_senior = np.random.choice([0, 1], size=n_customers, p=[0.85, 0.15])

    # Paperless billing
    paperless_billing = np.random.choice([0, 1], size=n_customers, p=[0.4, 0.6])

    # Phone service
    has_phone_service = np.random.choice([0, 1], size=n_customers, p=[0.1, 0.9])

    # Multiple lines (if has phone service)
    has_multiple_lines = np.where(
        has_phone_service == 1,
        np.random.choice([0, 1], size=n_customers, p=[0.6, 0.4]),
        0
    )

    # Calculate churn probability based on features
    churn_probability = np.zeros(n_customers)

    # Tenure effect (short tenure = higher churn)
    churn_probability += np.where(tenure < 6, 0.4,
                                 np.where(tenure < 24, 0.2, 0.05))

    # Monthly charges effect (high charges = higher churn)
    churn_probability += (monthly_charges - 20) / 100 * 0.3

    # Support calls effect (more calls = higher churn)
    churn_probability += np.clip(support_calls_30d / 10, 0, 0.25)

    # Contract type effect
    contract_effect = {'Month-to-Month': 0.25, 'One Year': 0.1, 'Two Year': 0.05}
    churn_probability += np.array([contract_effect[ct] for ct in contract_type])

    # Payment method effect (electronic check less reliable)
    payment_effect = {'Electronic check': 0.15, 'Mailed check': 0.05,
                     'Bank transfer': 0.02, 'Credit card': 0.02}
    churn_probability += np.array([payment_effect[pm] for pm in payment_method])

    # Dependents effect (stabilizing)
    churn_probability -= has_dependents * 0.1

    # Senior effect (slightly higher churn)
    churn_probability += is_senior * 0.05

    # Streaming services effect (more services = lower churn)
    churn_probability -= streaming_services * 0.03

    # Add random noise
    churn_probability += np.random.normal(0, 0.1, n_customers)
    churn_probability = np.clip(churn_probability, 0, 1)

    # Generate actual churn (0 or 1) based on probability
    churned = (np.random.random(n_customers) < churn_probability).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure_months': tenure,
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'contract_type': contract_type,
        'support_calls_30d': support_calls_30d,
        'data_usage_gb': np.round(data_usage_gb, 2),
        'payment_method': payment_method,
        'streaming_services': streaming_services,
        'internet_service': internet_service,
        'has_dependents': has_dependents,
        'is_senior': is_senior,
        'paperless_billing': paperless_billing,
        'has_phone_service': has_phone_service,
        'has_multiple_lines': has_multiple_lines,
        'churned': churned
    })

    return df

def create_sample_json(df, output_path):
    """Create a sample JSON file for API testing."""
    sample_customer = df[df['churned'] == 1].iloc[0].to_dict()
    # Remove churned field for API input
    sample_customer.pop('churned')

    import json
    with open(output_path, 'w') as f:
        json.dump(sample_customer, f, indent=2)

    print(f"\nSample customer JSON created at: {output_path}")

def main():
    """Generate and save customer churn dataset."""

    # Create output directories
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)

    # Generate data
    df = generate_customer_data(n_customers=10000)

    # Print statistics
    churn_rate = df['churned'].mean() * 100
    print(f"\nDataset Statistics:")
    print(f"  Total customers: {len(df):,}")
    print(f"  Churned customers: {df['churned'].sum():,} ({churn_rate:.1f}%)")
    print(f"  Retained customers: {(~df['churned'].astype(bool)).sum():,} ({100-churn_rate:.1f}%)")

    print(f"\nFeature Statistics:")
    print(f"  Avg tenure: {df['tenure_months'].mean():.1f} months")
    print(f"  Avg monthly charge: ${df['monthly_charges'].mean():.2f}")
    print(f"  Avg support calls: {df['support_calls_30d'].mean():.1f}")

    print(f"\nContract Distribution:")
    print(df['contract_type'].value_counts())

    # Save full dataset
    output_file = data_dir / 'customer_churn_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")

    # Create sample JSON for API testing
    sample_json_path = data_dir / 'sample_customer.json'
    create_sample_json(df, sample_json_path)

    print("\n✅ Data generation complete!")

if __name__ == "__main__":
    main()
