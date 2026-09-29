import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import warnings
import sys
import re
import joblib
from typing import Dict, List, Set, Union, Optional, Any

# --- Project Path Setup ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except (NameError, IndexError):
    PROJECT_ROOT = Path('.').resolve()

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Utility Functions (consolidated in common_utils.data_fetcher) ---
from common_utils.data_fetcher import (
    retry, write_atomic, strict_validate, apply_column_mapping,
    get_yfinance_data, get_fred_data, get_tiingo_data, get_stooq_data,
)

# --- Target Functions (consolidated in common_utils.target_utils) ---
from common_utils.target_utils import (
    create_targets, create_open_targets, create_binary_targets,
    infer_open_column,
)

# --- Feature Engineering Class (Alias to FeatureEngineBase) ---
from common_utils.feature_base import FeatureEngineBase
FeatureEngine = FeatureEngineBase