"""
Nightly pipeline — orchestrates data fetch → features → grade → train → predict.
Run once after market close each day.
"""
import logging, json, os
from datetime import datetime

from .config import PRED_FILE, LSTM_MIN_ROWS, FEATURE_COLS, PCA_H, PCA_RL
from .db     import (init_db, upsert_candles, upsert_features, get_candles,
                     get_features, insert_prediction, log_model_run, get_stats)
from .features import build_features, normalise_features, _log_ret, _atr14
from .grader   import grade_all
from .lorentzian import LorentzianClassifier, build_labels
from .lstm_model  import LSTMPredictor

log = logging.getLogger(__name__)


# ── Step 1: ingest candles already collected in this session ───────────────
def ingest_from_dict(mnq_data: dict, factor_data: dict):
    """
    mnq_data   : {ts: {open,high,low,close,volume}}
    factor_data: {symbol: {ts: close}}
    Call this after fetching from TradingView MCP.
    """
    import numpy as np

    def _row(symbol, ts, bar, atr_val=0.0, ret=0.0):
        return dict(
            symbol=symbol, ts=ts,
            dt=datetime.utcfromtimestamp(ts).isoformat(),
            open=bar.get("open", bar.get("close")),
            high=bar.get("high", bar.get("close")),
            low =bar.get("low",  bar.get("close")),
            close=bar["close"],
            volume=bar.get("volume", 0),
            atr14=atr_val,
            log_ret=ret
        )

    # MNQ candles with ATR + log_ret
    ts_sorted = sorted(mnq_data.keys())
    closes = [mnq_data[t]["close"] for t in ts_sorted]
    highs  = [mnq_data[t]["high"]  for t in ts_sorted]
    lows   = [mnq_data[t]["low"]   for t in ts_sorted]
    atrs   = list(_atr14(highs, lows, closes))
    rets   = list(_log_ret(closes))

    mnq_rows = [_row("MNQ", ts, mnq_data[ts], atrs[i], rets[i])
                for i, ts in enumerate(ts_sorted)]
    upsert_candles(mnq_rows)

    # Factor close-only candles
    for sym, ts_close in factor_data.items():
        ts_s  = sorted(ts_close.keys())
        f_cls = [ts_close[t] for t in ts_s]
        f_ret = list(_log_ret(f_cls))
        rows  = [_row(sym, ts, {"close": ts_close[ts]}, ret=f_ret[i])
                 for i, ts in enumerate(ts_s)]
        upsert_candles(rows)

    log.info(f"Ingested {len(mnq_rows)} MNQ bars + {len(factor_data)} factor symbols")


# ── Step 2: compute + store features ──────────────────────────────────────
def run_feature_engineering():
    from .config import SYMBOLS

    # Pull raw candles from DB
    mnq_rows   = get_candles("MNQ", limit=2000)
    factor_syms = [s for s in SYMBOLS if s != "MNQ"]

    if not mnq_rows:
        log.warning("No MNQ candles in DB — skipping feature engineering")
        return []

    mnq_candle_data   = {r["ts"]: r for r in mnq_rows}
    factor_close_data = {}
    for sym in factor_syms:
        rows = get_candles(sym, limit=2000)
        if rows:
            factor_close_data[sym] = {r["ts"]: r["close"] for r in rows}

    if len(factor_close_data) < 3:
        log.warning("Insufficient factor data for PCA features")
        return []

    feat_rows = build_features(mnq_candle_data, factor_close_data)
    feat_rows = grade_all(feat_rows)
    upsert_features(feat_rows)
    log.info(f"Computed {len(feat_rows)} feature rows")
    return feat_rows


# ── Step 3: train Lorentzian ───────────────────────────────────────────────
def train_lorentzian(feat_rows: list[dict]) -> LorentzianClassifier:
    import numpy as np

    if len(feat_rows) < 50:
        log.warning("Too few feature rows for Lorentzian training")
        clf = LorentzianClassifier()
        clf.load()
        return clf

    X      = normalise_features(feat_rows)
    closes = [r["_close"] for r in feat_rows]
    y      = build_labels(closes, lookahead=1)

    # Train/val split (last 15% = val)
    cut    = int(len(X) * 0.85)
    clf    = LorentzianClassifier()
    clf.fit(X[:cut], y[:cut])

    score  = clf.score(X[cut:], y[cut:])
    log.info(f"Lorentzian score: {score}")
    clf.save()
    log_model_run("lorentzian", cut, score.get("accuracy"), str(score))
    return clf


# ── Step 4: train LSTM ─────────────────────────────────────────────────────
def train_lstm(feat_rows: list[dict]) -> LSTMPredictor:
    import numpy as np

    predictor = LSTMPredictor()
    predictor.load()      # warm-start from previous weights if available

    if len(feat_rows) < LSTM_MIN_ROWS:
        log.warning(f"Only {len(feat_rows)} rows — LSTM needs {LSTM_MIN_ROWS}")
        return predictor

    X      = normalise_features(feat_rows)
    closes = [r["_close"] for r in feat_rows]
    y      = build_labels(closes, lookahead=1)

    result = predictor.fit(X, y)
    log.info(f"LSTM result: {result}")
    predictor.save()
    log_model_run("lstm", len(X),
                  result.get("best_val_acc"), str(result))
    return predictor


# ── Step 5: generate next-session predictions ──────────────────────────────
def generate_predictions(feat_rows: list[dict],
                         clf: LorentzianClassifier,
                         lstm: LSTMPredictor) -> dict:
    """
    Use the last LSTM_SEQ_LEN bars to predict the next session's first signal.
    Saves predictions to JSON and DB.
    """
    import numpy as np
    from .config import LSTM_SEQ_LEN

    if len(feat_rows) < LSTM_SEQ_LEN + 5:
        log.warning("Not enough bars for prediction")
        return {}

    X        = normalise_features(feat_rows)
    last_x   = X[-1]                       # most recent bar
    recent_X = X[-LSTM_SEQ_LEN:]           # for LSTM

    loren_sig, loren_conf = clf.predict_one(last_x)
    lstm_sig,  lstm_conf  = lstm.predict_one(recent_X)

    # Ensemble: weighted vote (Lorentzian 40%, LSTM 60%)
    ensemble = 0.4 * loren_sig * loren_conf + 0.6 * lstm_sig * lstm_conf

    last_feat = feat_rows[-1]
    pred = dict(
        target_ts   = None,             # filled when next bar arrives
        lorentzian  = round(loren_conf * loren_sig, 4),
        lstm_dir    = lstm_sig,
        lstm_conf   = round(lstm_conf, 4),
        ensemble    = round(ensemble, 4),
        grade       = last_feat.get("candle_grade", "?"),
        signal_text = ("LONG" if ensemble > 0.2 else
                       "SHORT" if ensemble < -0.2 else "NEUTRAL"),
    )

    insert_prediction(pred)

    # Write JSON for dashboard + Pine Script integration
    with open(PRED_FILE, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "prediction":   pred,
            "last_grade":   last_feat.get("candle_grade"),
            "last_zscore":  last_feat.get("pca_zscore"),
            "stats":        get_stats(),
        }, f, indent=2)

    log.info(f"Prediction → {pred['signal_text']}  "
             f"ensemble={pred['ensemble']:.3f}  grade={pred['grade']}")
    return pred


# ── Master nightly run ────────────────────────────────────────────────────
def run_nightly(mnq_data: dict = None, factor_data: dict = None):
    """
    Full nightly pipeline.
    If mnq_data/factor_data provided, ingests them first.
    Otherwise works from whatever is already in the DB.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(os.path.dirname(__file__), "logs",
                             f"nightly_{datetime.now().strftime('%Y%m%d')}.log")
            )
        ]
    )
    log.info("═" * 60)
    log.info("NIGHTLY PIPELINE START")
    log.info("═" * 60)

    init_db()

    if mnq_data and factor_data:
        ingest_from_dict(mnq_data, factor_data)

    feat_rows = run_feature_engineering()

    if not feat_rows:
        log.error("No features generated — aborting")
        return None

    clf   = train_lorentzian(feat_rows)
    lstm  = train_lstm(feat_rows)
    pred  = generate_predictions(feat_rows, clf, lstm)

    log.info("NIGHTLY PIPELINE COMPLETE")
    stats = get_stats()
    log.info(f"DB stats: {stats}")
    return pred
