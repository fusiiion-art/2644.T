import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects data drift using Population Stability Index (PSI) and statistical checks.
    """
    def __init__(self, psi_threshold: float = 0.25):
        self.psi_threshold = psi_threshold

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for a single feature.
        """
        def scale_range(input, min, max):
            input += (1e-6)  # Avoid zero division
            return (input - min) / (max - min)

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        try:
            # Handle potential NaNs
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]
            
            if len(expected) == 0 or len(actual) == 0:
                return 0.0

            # Define buckets based on expected distribution
            # percentiles = np.percentile(expected, breakpoints)
            # Use fixed range based on expected to ensure consistent buckets
            min_val = min(expected.min(), actual.min())
            max_val = max(expected.max(), actual.max())
            
             # If values are constant
            if min_val == max_val:
                return 0.0

            # Histogram
            expected_percents = np.histogram(expected, bins=buckets, range=(min_val, max_val))[0] / len(expected)
            actual_percents = np.histogram(actual, bins=buckets, range=(min_val, max_val))[0] / len(actual)

            # Avoid zero for division
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
            return psi_value
            
        except Exception as e:
            logger.warning(f"Failed to calculate PSI: {e}")
            return 0.0

    def detect_drift(self, 
                     reference_df: pd.DataFrame, 
                     current_df: pd.DataFrame, 
                     features: List[str]) -> Dict[str, Any]:
        """
        Check for drift across specified features.
        Returns a dictionary of drift reports.
        """
        drift_report = {
            "drift_detected": False,
            "details": {}
        }
        
        warnings = []

        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
            
            psi = self.calculate_psi(reference_df[feature].values, current_df[feature].values)
            
            # Simple stats check (Mean shift)
            ref_mean = reference_df[feature].mean()
            curr_mean = current_df[feature].mean()
            mean_shift = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-9)

            status = "OK"
            if psi > self.psi_threshold:
                status = "DRIFT"
                drift_report["drift_detected"] = True
                warnings.append(f"{feature} PSI={psi:.4f} (High Drift)")
            elif psi > 0.1:
                status = "WARNING"
                warnings.append(f"{feature} PSI={psi:.4f} (Moderate Drift)")

            drift_report["details"][feature] = {
                "psi": psi,
                "mean_shift": mean_shift,
                "status": status
            }
            
        drift_report["summary"] = warnings
        return drift_report
