"""Generates a synthetic 'customer churn / risk' enterprise dataset.

Used to seed the CSV and SQL demo sources and to train/evaluate ML models
without requiring proprietary customer data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_customer_dataset(n_rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic customer risk/churn dataset with realistic structure.

    Includes numeric, categorical, and missing values so the feature
    engineering pipeline has real work to do.
    """
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 72, n_rows)
    monthly_spend = rng.normal(120, 45, n_rows).clip(10, None)
    support_tickets = rng.poisson(1.5, n_rows)
    region = rng.choice(["AMER", "EMEA", "APAC", "LATAM"], n_rows, p=[0.4, 0.3, 0.2, 0.1])
    plan_type = rng.choice(["basic", "pro", "enterprise"], n_rows, p=[0.5, 0.35, 0.15])
    satisfaction_score = rng.normal(7, 1.8, n_rows).clip(0, 10)
    contract_length = rng.choice([1, 12, 24], n_rows, p=[0.3, 0.5, 0.2])

    # Latent churn probability driven by a mix of features (ground truth signal)
    risk_logit = (
        -0.06 * tenure_months
        + 0.35 * support_tickets
        - 0.55 * satisfaction_score
        - 0.04 * contract_length
        + 3.3
        + rng.normal(0, 1.0, n_rows)
    )
    churn_prob = 1 / (1 + np.exp(-risk_logit))
    churned = (rng.uniform(0, 1, n_rows) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n_rows)],
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend.round(2),
            "support_tickets": support_tickets,
            "region": region,
            "plan_type": plan_type,
            "satisfaction_score": satisfaction_score.round(2),
            "contract_length_months": contract_length,
            "churned": churned,
        }
    )

    # Inject realistic missingness
    missing_idx = rng.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    df.loc[missing_idx, "satisfaction_score"] = np.nan
    missing_idx2 = rng.choice(n_rows, size=int(n_rows * 0.03), replace=False)
    df.loc[missing_idx2, "monthly_spend"] = np.nan

    return df


if __name__ == "__main__":
    from src.utils.config import get_settings

    settings = get_settings()
    df = generate_customer_dataset()
    out_path = settings.data_raw_dir / "customers.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")