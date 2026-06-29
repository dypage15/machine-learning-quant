"""
Feature engineering — computes all ML input features from raw OHLCV + PCA signal.
"""
import numpy as np
import math
from .config import (PCA_H, PCA_RL, PCA_SM, PCA_CB, SIGNAL_DIR,
                     GRADE_WEIGHTS, GRADE_THRESHOLDS, FEATURE_COLS)

# ── Utility ────────────────────────────────────────────────────────────────
def _log_ret(arr):
    arr = np.array(arr, dtype=float)
    r   = np.zeros(len(arr))
    pos = arr[:-1] > 0
    r[1:][pos] = np.log(arr[1:][pos] / arr[:-1][pos])
    return r

def _atr14(h, l, c):
    h, l, c = np.array(h,float), np.array(l,float), np.array(c,float)
    n  = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = np.zeros(n)
    if n >= 14:
        atr[13] = tr[:14].mean()
        for i in range(14, n):
            atr[i] = (atr[i-1]*13 + tr[i]) / 14
    return atr

def _rsi(close, period=14):
    close = np.array(close, float)
    n     = len(close)
    rsi   = np.full(n, 0.5)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses= np.where(delta < 0, -delta, 0.0)
    avg_g = gains[:period].mean()
    avg_l = losses[:period].mean()
    for i in range(period, n):
        avg_g = (avg_g * (period-1) + gains[i])  / period
        avg_l = (avg_l * (period-1) + losses[i]) / period
        rs    = avg_g / avg_l if avg_l > 1e-12 else 1e6
        rsi[i]= 1 - 1/(1+rs)     # normalised 0–1
    return rsi

def _ema(close, period=20):
    close = np.array(close, float)
    ema   = np.zeros(len(close))
    k     = 2 / (period + 1)
    ema[0]= close[0]
    for i in range(1, len(close)):
        ema[i] = close[i]*k + ema[i-1]*(1-k)
    return ema

def _bollinger(close, period=20):
    close = np.array(close, float)
    n     = len(close)
    pos   = np.full(n, 0.5)
    for i in range(period, n):
        w   = close[i-period:i]
        mu  = w.mean()
        sd  = w.std(ddof=0)
        if sd > 1e-12:
            pos[i] = (close[i] - (mu - 2*sd)) / (4*sd)
            pos[i] = float(np.clip(pos[i], 0, 1))
    return pos

# ── PCA signal computation ─────────────────────────────────────────────────
def compute_pca_signal(r_mnq, factor_returns: dict,
                       H=PCA_H, RL=PCA_RL, SM=PCA_SM, CB=PCA_CB,
                       direction=SIGNAL_DIR):
    """
    Returns arrays: disloc, sigma_D, zscore, cs (confirmed signal)
    direction: -1 = Momentum (MNQ leads factors → keep going)
               +1 = MeanRev  (MNQ disloc → revert)
    """
    from numpy.lib.stride_tricks import sliding_window_view

    r_mnq   = np.array(r_mnq, float)
    factors  = [np.array(v, float) for v in factor_returns.values()]
    N, K     = len(r_mnq), len(factors)

    # Vectorised rolling OLS
    y_w  = sliding_window_view(r_mnq, H)
    F_w  = np.stack([sliding_window_view(f, H) for f in factors], axis=2)
    y_c  = y_w - y_w.mean(axis=1, keepdims=True)
    F_c  = F_w - F_w.mean(axis=1, keepdims=True)
    cov  = (F_c * y_c[:,:,None]).mean(axis=1)
    var  = (F_c**2).mean(axis=1)
    sty  = y_c.std(axis=1)
    valid= (var > 1e-12) & (sty[:,None] > 1e-12)
    betas= np.where(valid, cov/var, 0.0)
    F_all= np.column_stack(factors)
    pred = (betas[:N-H] * F_all[H:N]).sum(axis=1)
    disloc = np.zeros(N)
    disloc[H:N] = r_mnq[H:N] - pred

    # Rolling sigma_D
    sigma_D = np.zeros(N)
    if RL < N:
        wins = sliding_window_view(disloc, RL)
        sigma_D[RL:] = wins[:N-RL].std(axis=1, ddof=0)

    # Z-score
    zscore = np.zeros(N)
    nz = sigma_D > 1e-12
    zscore[nz] = disloc[nz] / sigma_D[nz]

    # Raw signal (momentum direction)
    start = max(H, RL)
    fs    = np.zeros(N, dtype=np.int8)
    v     = sigma_D[start:] > 1e-12
    d     = disloc[start:]; s = sigma_D[start:]
    # Momentum: disloc > +SM*sigma → long (MNQ leading up → follow)
    fs[start:][ v & (d >  SM*s) ] = direction * (-1)
    fs[start:][ v & (d < -SM*s) ] = direction * ( 1)

    # Streak confirmation
    stk = np.zeros(N, dtype=np.int8)
    cs  = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        if fs[i] != 0 and fs[i] == fs[i-1]:
            stk[i] = min(stk[i-1] + 1, 127)
        elif fs[i] != 0:
            stk[i] = 1
        if stk[i] >= CB:
            cs[i] = fs[i]

    return disloc, sigma_D, zscore, cs


# ── Main feature builder ───────────────────────────────────────────────────
def build_features(candle_data: dict, factor_closes: dict) -> list[dict]:
    """
    candle_data: {ts: {open,high,low,close,volume}, ...}  (MNQ)
    factor_closes: {symbol: {ts: close, ...}, ...}
    Returns list of feature dicts (one per common bar).
    """
    # Align timestamps
    ts_sets = [set(candle_data.keys())] + [set(v.keys()) for v in factor_closes.values()]
    common  = sorted(set.intersection(*ts_sets))
    if len(common) < max(PCA_RL, PCA_H) + 30:
        return []

    mnq_o = np.array([candle_data[t]["open"]   for t in common], float)
    mnq_h = np.array([candle_data[t]["high"]   for t in common], float)
    mnq_l = np.array([candle_data[t]["low"]    for t in common], float)
    mnq_c = np.array([candle_data[t]["close"]  for t in common], float)
    mnq_v = np.array([candle_data[t].get("volume", 0) for t in common], float)

    factor_c = {sym: np.array([factor_closes[sym][t] for t in common], float)
                for sym in factor_closes}

    # Core arrays
    r_mnq   = _log_ret(mnq_c)
    atr14   = _atr14(mnq_h, mnq_l, mnq_c)
    rsi14   = _rsi(mnq_c)
    bb_pos  = _bollinger(mnq_c)
    ema20   = _ema(mnq_c, 20)

    # Factor returns
    factor_ret = {sym: _log_ret(factor_c[sym]) for sym in factor_c}

    # Volume ratio (vol / 20-bar avg)
    vol_avg = np.zeros(len(common))
    for i in range(20, len(common)):
        vol_avg[i] = mnq_v[i-20:i].mean() if mnq_v[i-20:i].mean() > 0 else 1.0
    vol_ratio = np.where(vol_avg > 0, mnq_v / (vol_avg + 1e-9), 1.0)

    # Rolling beta to ES
    beta_es = np.zeros(len(common))
    es_ret  = factor_ret.get("ES", np.zeros(len(common)))
    for i in range(PCA_H, len(common)):
        y = r_mnq[i-PCA_H:i]; x = es_ret[i-PCA_H:i]
        xc = x - x.mean(); yc = y - y.mean()
        vx = (xc**2).mean()
        if vx > 1e-12:
            beta_es[i] = (xc*yc).mean() / vx

    # PCA signal
    disloc, sigma_D, zscore, cs = compute_pca_signal(
        r_mnq, factor_ret
    )

    # Session encoding
    import datetime as dt
    hours = np.array([dt.datetime.utcfromtimestamp(t).hour + 4  # CT offset approx
                      for t in common], float) % 24
    dows  = np.array([dt.datetime.utcfromtimestamp(t).weekday() for t in common], float)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    dow_sin  = np.sin(2 * np.pi * dows  / 5)
    dow_cos  = np.cos(2 * np.pi * dows  / 5)

    # Candle components
    ranges     = mnq_h - mnq_l
    bodies     = np.abs(mnq_c - mnq_o)
    upper_wick = mnq_h - np.maximum(mnq_c, mnq_o)
    lower_wick = np.minimum(mnq_c, mnq_o) - mnq_l
    body_ratio = np.where(ranges > 0, bodies / ranges, 0.5)
    upper_r    = np.where(ranges > 0, upper_wick / ranges, 0.0)
    lower_r    = np.where(ranges > 0, lower_wick / ranges, 0.0)
    atr_ratio  = np.where(atr14 > 0, ranges / atr14, 1.0)
    trend_align= np.sign(mnq_c - ema20).astype(float)

    rows = []
    for i, ts in enumerate(common):
        row = dict(
            ts          = ts,
            pca_zscore  = float(zscore[i]),
            atr_ratio   = float(np.clip(atr_ratio[i], 0, 5)),
            body_ratio  = float(body_ratio[i]),
            upper_wick  = float(upper_r[i]),
            lower_wick  = float(lower_r[i]),
            log_ret_mnq = float(r_mnq[i]),
            log_ret_es  = float(factor_ret.get("ES",  [0]*len(common))[i]),
            log_ret_ym  = float(factor_ret.get("YM",  [0]*len(common))[i]),
            log_ret_zn  = float(factor_ret.get("ZN",  [0]*len(common))[i]),
            rsi14       = float(rsi14[i]),
            bb_pos      = float(bb_pos[i]),
            vol_ratio   = float(np.clip(vol_ratio[i], 0, 5)),
            hour_sin    = float(hour_sin[i]),
            hour_cos    = float(hour_cos[i]),
            dow_sin     = float(dow_sin[i]),
            dow_cos     = float(dow_cos[i]),
            beta_es     = float(beta_es[i]),
            trend_align = float(trend_align[i]),
            # extras for grader
            _cs         = int(cs[i]),
            _atr14      = float(atr14[i]),
            _close      = float(mnq_c[i]),
        )
        rows.append(row)
    return rows


# ── Normalise for ML ───────────────────────────────────────────────────────
def normalise_features(rows: list[dict]) -> np.ndarray:
    """Returns (N, F) float32 array, robust z-score normalised."""
    from .config import FEATURE_COLS
    X = np.array([[r.get(c, 0.0) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    # Clip extremes then standardise
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-8
    X   = np.clip((X - med) / (1.4826 * mad), -4, 4)
    return X
