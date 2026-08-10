"""Data drift detection utilities.

Implements two complementary drift signals used across the industry:
- Kolmogorov-Smirnov test for numeric feature distribution shift
- Population Stability Index (PSI) for binned distribution shift

Both are cheap to compute and don't require a live model, so they can run
as a pre-training data-quality gate or a scheduled monitoring job.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils.config import get_settings


@dataclass
class DriftResult:
    feature: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    drifted: bool
    details: dict = field(default_factory=dict)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI between reference and current distributions of a numeric feature."""
    ref = reference.dropna().to_numpy()
    cur = current.dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(ref, quantiles))
    if len(breakpoints) < 3:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def detect_numeric_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, numeric_columns: list[str]
) -> list[DriftResult]:
    """Run KS test + PSI for each numeric column, flagging drift against configured thresholds."""
    settings = get_settings()
    results: list[DriftResult] = []
    for col in numeric_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_series = reference_df[col].dropna()
        cur_series = current_df[col].dropna()
        if len(ref_series) < 5 or len(cur_series) < 5:
            continue
        ks_stat, ks_p = ks_2samp(ref_series, cur_series)
        psi = population_stability_index(ref_series, cur_series)
        drifted = bool(ks_p < settings.drift_ks_pvalue_threshold or psi > settings.drift_psi_threshold)
        results.append(
            DriftResult(
                feature=col,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p),
                psi=psi,
                drifted=drifted,
            )
        )
    return results


def drift_report_to_dict(results: list[DriftResult]) -> dict:
    return {
        "n_features_checked": len(results),
        "n_drifted": sum(r.drifted for r in results),
        "features": [
            {
                "feature": r.feature,
                "ks_statistic": round(r.ks_statistic, 4),
                "ks_pvalue": round(r.ks_pvalue, 4),
                "psi": round(r.psi, 4),
                "drifted": r.drifted,
            }
            for r in results
        ],
    }