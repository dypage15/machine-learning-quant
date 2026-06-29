#!/usr/bin/env python3
"""
PCA MNQ · Nightly Pipeline Runner
===================================
Run this script each evening after market close (e.g. 5:30 PM ET).

What it does:
  1. Fetches 500 bars of OHLCV data for MNQ + all 5 factors from TradingView
  2. Computes PCA dislocation features, RSI, Bollinger Bands, candle metrics
  3. Grades each candle A–F
  4. Retrains the Lorentzian KNN classifier
  5. Retrains the LSTM neural network (if 100+ bars available)
  6. Generates next-session signal prediction (LONG / SHORT / NEUTRAL)
  7. Saves prediction to next_session_predictions.json  ← dashboard reads this

Usage:
    python run_nightly.py                   # auto-fetch from TradingView
    python run_nightly.py --skip-fetch      # use data already in DB
    python run_nightly.py --dashboard       # launch Streamlit after pipeline
    python run_nightly.py --no-train        # skip model retraining
"""
import argparse, sys, os, json, logging
from datetime import datetime

# ── Add project root to path ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Configure logging to console + file ───────────────────────────────────
# Use home dir for logs (same as DB) to avoid permission issues on cloud-synced paths
LOG_DIR = os.path.join(os.path.expanduser("~"), ".pca_mnq_ml", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"nightly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Use UTF-8 for stdout on Windows to avoid cp1252 UnicodeEncodeError on box chars
stdout_handler = logging.StreamHandler(sys.stdout)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        stdout_handler,
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)
log = logging.getLogger("run_nightly")


# ── TradingView fetch ──────────────────────────────────────────────────────
def fetch_from_tradingview() -> tuple[dict, dict]:
    """
    Pulls OHLCV bars from TradingView via MCP.
    Returns (mnq_data, factor_data).

    If TradingView MCP is not connected, raises RuntimeError with instructions.
    """
    try:
        # Import TradingView MCP wrapper — only available when MCP is connected
        from mcp_tradingview import data_get_ohlcv     # noqa (MCP runtime injection)
    except ImportError:
        raise RuntimeError(
            "TradingView MCP not connected.\n"
            "  • Open TradingView in Claude Desktop\n"
            "  • Or run with --skip-fetch to use data already in the database\n"
            "  • Or import data manually with: python run_nightly.py --import-json <file>"
        )

    from ml_system.config import SYMBOLS, TIMEFRAME, BARS_FETCH

    log.info("Fetching bars from TradingView ...")

    mnq_sym = SYMBOLS["MNQ"]
    raw_mnq = data_get_ohlcv(symbol=mnq_sym, timeframe=TIMEFRAME, bars=BARS_FETCH)

    # raw_mnq is a list of {time, open, high, low, close, volume}
    mnq_data = {}
    for bar in raw_mnq:
        ts = int(bar["time"])
        mnq_data[ts] = {
            "open":   float(bar["open"]),
            "high":   float(bar["high"]),
            "low":    float(bar["low"]),
            "close":  float(bar["close"]),
            "volume": float(bar.get("volume", 0)),
        }

    factor_data = {}
    for name, sym in SYMBOLS.items():
        if name == "MNQ":
            continue
        raw = data_get_ohlcv(symbol=sym, timeframe=TIMEFRAME, bars=BARS_FETCH)
        factor_data[name] = {int(b["time"]): float(b["close"]) for b in raw}
        log.info(f"  {name}: {len(raw)} bars")

    log.info(f"MNQ: {len(mnq_data)} bars")
    return mnq_data, factor_data


# ── Import from JSON (manual data feed) ───────────────────────────────────
def import_from_json(path: str) -> tuple[dict, dict]:
    """
    Load pre-collected data from a JSON file.
    Expected format:
    {
      "mnq":  { "<unix_ts>": {open,high,low,close,volume}, ... },
      "factors": {
        "ES":  { "<unix_ts>": <close>, ... },
        "YM":  { ... },
        ...
      }
    }
    """
    with open(path) as f:
        d = json.load(f)

    mnq_data    = {int(k): v for k, v in d["mnq"].items()}
    factor_data = {sym: {int(k): float(v) for k, v in bars.items()}
                   for sym, bars in d["factors"].items()}
    log.info(f"Imported {len(mnq_data)} MNQ bars from {path}")
    return mnq_data, factor_data


# ── Main entry point ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="PCA MNQ Nightly ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-fetch",  action="store_true",
                        help="Skip data fetch; use whatever is already in the DB")
    parser.add_argument("--no-train",    action="store_true",
                        help="Skip model retraining (just re-run predictions)")
    parser.add_argument("--dashboard",   action="store_true",
                        help="Launch Streamlit dashboard after pipeline completes")
    parser.add_argument("--import-json", metavar="FILE",
                        help="Import data from a JSON file instead of TradingView")
    parser.add_argument("--export-json", metavar="FILE",
                        help="Export fetched bars to JSON before running pipeline")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("PCA MNQ · NIGHTLY PIPELINE")
    log.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    from ml_system.db import init_db
    from ml_system.pipeline import (ingest_from_dict, run_feature_engineering,
                                     train_lorentzian, train_lstm,
                                     generate_predictions)
    from ml_system.config import PRED_FILE

    init_db()

    # ── Step 1: Data ─────────────────────────────────────────────────────
    mnq_data = factor_data = None

    if args.import_json:
        mnq_data, factor_data = import_from_json(args.import_json)
    elif not args.skip_fetch:
        try:
            mnq_data, factor_data = fetch_from_tradingview()
        except RuntimeError as e:
            log.error(str(e))
            sys.exit(1)

    if mnq_data and factor_data:
        # Optional: save raw bars for later replay / debugging
        if args.export_json:
            payload = {
                "mnq":     {str(k): v for k, v in mnq_data.items()},
                "factors": {sym: {str(k): v for k, v in d.items()}
                            for sym, d in factor_data.items()},
            }
            with open(args.export_json, "w") as f:
                json.dump(payload, f, indent=2)
            log.info(f"Exported bars → {args.export_json}")

        ingest_from_dict(mnq_data, factor_data)
    else:
        log.info("Using existing DB data (--skip-fetch)")

    # ── Step 2: Features + grades ─────────────────────────────────────────
    log.info("-" * 40)
    log.info("Building features ...")
    feat_rows = run_feature_engineering()

    if not feat_rows:
        log.error("No features generated. Aborting.")
        sys.exit(1)

    log.info(f"  {len(feat_rows)} feature rows ready")

    # ── Step 3: Train models ──────────────────────────────────────────────
    if args.no_train:
        log.info("-" * 40)
        log.info("Skipping model training (--no-train)")
        from ml_system.lorentzian import LorentzianClassifier
        from ml_system.lstm_model  import LSTMPredictor
        clf  = LorentzianClassifier(); clf.load()
        lstm = LSTMPredictor();        lstm.load()
    else:
        log.info("-" * 40)
        log.info("Training Lorentzian KNN ...")
        clf = train_lorentzian(feat_rows)

        log.info("-" * 40)
        log.info("Training LSTM ...")
        lstm = train_lstm(feat_rows)

    # ── Step 4: Predict ───────────────────────────────────────────────────
    log.info("-" * 40)
    log.info("Generating next-session prediction ...")
    pred = generate_predictions(feat_rows, clf, lstm)

    if pred:
        log.info("=" * 60)
        log.info(f"  SIGNAL      : {pred['signal_text']}")
        log.info(f"  Ensemble    : {pred['ensemble']:+.3f}")
        log.info(f"  Lorentzian  : {pred['lorentzian']:+.3f}")
        log.info(f"  LSTM conf   : {pred['lstm_conf']:.1%}")
        log.info(f"  Last grade  : {pred['grade']}")
        log.info(f"  Saved to    : {PRED_FILE}")
        log.info("=" * 60)
    else:
        log.warning("Prediction could not be generated (not enough bars?)")

    log.info(f"Log written → {log_file}")


    # -- Step 5: Launch dashboard (optional) ---------------------------------
    if args.dashboard:
        log.info("Launching Streamlit dashboard ...")
        os.system(
            f"streamlit run {os.path.join(ROOT, 'ml_system', 'dashboard.py')}"
        )


if __name__ == "__main__":
    main()
