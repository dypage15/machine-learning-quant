"""
PCA MNQ ML System — Configuration
All tuneable constants in one place.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────
# Data lives in the user's home directory (~/.pca_mnq_ml/) so it works
# regardless of where the code is stored (Desktop, OneDrive, network share).
# SQLite requires proper OS-level file locking — cloud-synced or mounted
# folders (OneDrive, Dropbox, Google Drive) can cause "disk I/O error".
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(os.path.expanduser("~"), ".pca_mnq_ml")
DB_PATH     = os.path.join(DATA_DIR, "pca_ml.db")
MODEL_DIR   = os.path.join(DATA_DIR, "models")
LOG_DIR     = os.path.join(DATA_DIR, "logs")
PRED_FILE   = os.path.join(DATA_DIR, "next_session_predictions.json")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ── TradingView symbols ───────────────────────────────────────────────────
SYMBOLS = {
    "MNQ": "CME_MINI:MNQ1!",
    "ES":  "CME_MINI:ES1!",
    "YM":  "CBOT_MINI:YM1!",
    "RTY": "CME_MINI:RTY1!",
    "ZN":  "CBOT:ZN1!",
    "GC":  "COMEX:GC1!",
}
TIMEFRAME   = "60"          # 60-minute bars
BARS_FETCH  = 500           # bars to pull each night

# ── PCA signal params (best from optimisation) ────────────────────────────
PCA_H       = 10            # rolling beta window
PCA_RL      = 20            # sigma_D lookback
PCA_SM      = 0.75          # signal multiplier
PCA_CB      = 2             # streak confirmation bars
SIGNAL_DIR  = -1            # -1 = Momentum (flip from MIT mean-rev)

# ── Backtest / execution params ───────────────────────────────────────────
MH          = 2             # max hold bars
SL_MULT     = 1.0           # ATR stop-loss multiplier
RR          = 1.5           # reward/risk ratio
TICK        = 0.25          # MNQ tick size
MULT        = 2.0           # $ per point
COMM        = 0.35          # commission per contract one-way
SLIP        = TICK          # 1-tick slippage

# ── Lorentzian KNN ────────────────────────────────────────────────────────
KNN_K           = 8         # neighbours
KNN_MAX_BACK    = 2000      # max historical bars to search
KNN_SKIP        = 4         # use every 4th bar (diversity)
KNN_THRESHOLD   = 0.55      # min vote fraction to signal

# ── LSTM ──────────────────────────────────────────────────────────────────
LSTM_SEQ_LEN    = 20        # input sequence length (bars)
LSTM_HIDDEN     = 64
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.2
LSTM_EPOCHS     = 50        # nightly retrain epochs
LSTM_LR         = 1e-3
LSTM_BATCH      = 32
LSTM_MIN_ROWS   = 100       # minimum rows before training

# ── Candle grading ────────────────────────────────────────────────────────
GRADE_WEIGHTS = {
    "pca_zscore":   0.30,   # signal strength
    "vol_ratio":    0.15,   # volume vs 20-bar avg
    "body_ratio":   0.15,   # body / range (candle conviction)
    "trend_align":  0.20,   # price vs EMA20
    "atr_ratio":    0.10,   # range / ATR14 (volatility normal)
    "wick_score":   0.10,   # lower wick (for longs) or upper (shorts)
}
GRADE_THRESHOLDS = [0.80, 0.65, 0.50, 0.35, 0.20]  # A B C D E → else F

# ── Feature list (order matters for model inputs) ─────────────────────────
FEATURE_COLS = [
    # ── PCA signal ───────────────────────────────────────────────────────
    "pca_zscore",       # PCA dislocation z-score (vs rolling sigma)
    "pca_cs",           # PCA confirmed signal: -1 / 0 / +1  ← core MIT signal
    # ── Factor log returns (all 5 factors) ──────────────────────────────
    "log_ret_mnq",      # MNQ 1-bar log return
    "log_ret_es",       # ES  1-bar log return
    "log_ret_ym",       # YM  1-bar log return
    "log_ret_zn",       # ZN  1-bar log return (bonds — risk-off indicator)
    "log_ret_rty",      # RTY 1-bar log return (small-cap risk proxy)
    "log_ret_gc",       # GC  1-bar log return (gold — safe haven)
    # ── Rolling beta & trend ─────────────────────────────────────────────
    "beta_es",          # rolling 10-bar beta to ES
    "trend_align",      # sign(close - EMA20): -1 / 0 / +1
    # ── Candle structure ─────────────────────────────────────────────────
    "atr_ratio",        # candle range / ATR14 (volatility normalised)
    "body_ratio",       # abs(close-open) / range (conviction)
    "upper_wick",       # upper wick fraction
    "lower_wick",       # lower wick fraction
    # ── Momentum / mean-reversion ────────────────────────────────────────
    "rsi14",            # RSI-14 (0–1 normalised)
    "bb_pos",           # Bollinger Band position (0=lower band, 1=upper)
    # ── Volume ───────────────────────────────────────────────────────────
    "vol_ratio",        # volume / 20-bar avg volume
    # ── Candle quality composite ─────────────────────────────────────────
    "grade_score",      # composite grade 0–1 (A=~0.9 … F=~0.1)
    # ── Session timing ───────────────────────────────────────────────────
    "hour_sin",         # hour of day sin
    "hour_cos",         # hour of day cos
    "dow_sin",          # day of week sin
    "dow_cos",          # day of week cos
]
N_FEATURES = len(FEATURE_COLS)   # 22
