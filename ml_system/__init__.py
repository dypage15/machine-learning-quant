"""
PCA MNQ ML Intelligence System
Lorentzian KNN + LSTM neural network for futures signal classification.

Usage:
    python run_nightly.py              # fetch data, retrain, generate predictions
    streamlit run ml_system/dashboard.py   # launch live dashboard
"""
__version__ = "1.0.0"
__author__  = "PCA MNQ ML System"

from .pipeline import run_nightly, ingest_from_dict, generate_predictions
from .db       import init_db, get_stats
from .config   import PRED_FILE, DB_PATH

__all__ = [
    "run_nightly",
    "ingest_from_dict",
    "generate_predictions",
    "init_db",
    "get_stats",
    "PRED_FILE",
    "DB_PATH",
]
