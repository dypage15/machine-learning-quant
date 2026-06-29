"""
Lorentzian KNN Classifier
Based on jdehorty's Lorentzian Distance concept — uses log(1+|x-y|) distance
which is more robust to fat tails and outliers than Euclidean distance.

Key properties vs Euclidean:
  - Compresses large differences logarithmically
  - Treats all features more equally regardless of scale
  - Works better on financial time series with non-normal distributions
"""
import numpy as np
import json, os
from .config import KNN_K, KNN_MAX_BACK, KNN_SKIP, KNN_THRESHOLD, MODEL_DIR


# ── Distance metric ────────────────────────────────────────────────────────
def lorentzian_distance(x: np.ndarray, y: np.ndarray) -> float:
    """d(x,y) = Σ log(1 + |xᵢ − yᵢ|)"""
    return float(np.sum(np.log1p(np.abs(x - y))))


def lorentzian_distances_batch(X: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Vectorised: compute Lorentzian distance from query to every row of X.
    X: (N, F),  query: (F,)  →  returns (N,)
    """
    return np.sum(np.log1p(np.abs(X - query)), axis=1)


# ── Classifier ─────────────────────────────────────────────────────────────
class LorentzianClassifier:
    """
    KNN with Lorentzian distance. Learns from accumulating bar data.

    label convention:  +1 = next bar up (trade long)
                       -1 = next bar down (trade short)
                        0 = no signal
    """

    def __init__(self, k=KNN_K, max_back=KNN_MAX_BACK,
                 skip=KNN_SKIP, threshold=KNN_THRESHOLD):
        self.k         = k
        self.max_back  = max_back
        self.skip      = skip          # use every Nth bar for diversity
        self.threshold = threshold     # min |vote| fraction to generate signal
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    # ── Fit ──────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, F) normalised features
        y: (N,)   labels  +1 / -1 / 0
        Store every skip-th bar, capped at max_back.
        """
        idx = np.arange(0, len(X), self.skip)
        if len(idx) > self.max_back:
            idx = idx[-self.max_back:]
        self.X_train = X[idx].astype(np.float32)
        self.y_train = y[idx].astype(np.int8)
        return self

    # ── Predict single bar ────────────────────────────────────────────────
    def predict_one(self, x: np.ndarray) -> tuple[int, float]:
        """
        Returns (signal, confidence).
        signal: +1 / -1 / 0
        confidence: 0–1 (fraction of k neighbours in majority direction)
        """
        if self.X_train is None or len(self.X_train) < self.k:
            return 0, 0.0

        dists = lorentzian_distances_batch(self.X_train, x.astype(np.float32))
        knn   = np.argsort(dists)[:self.k]
        votes = self.y_train[knn]

        n_long  = int((votes ==  1).sum())
        n_short = int((votes == -1).sum())
        n_total = self.k

        vote_frac = (n_long - n_short) / n_total   # -1 to +1

        if vote_frac >= self.threshold:
            return 1, abs(vote_frac)
        elif vote_frac <= -self.threshold:
            return -1, abs(vote_frac)
        else:
            return 0, abs(vote_frac)

    # ── Predict array ─────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns signals and confidences arrays."""
        signals = np.zeros(len(X), dtype=int)
        confs   = np.zeros(len(X))
        for i in range(len(X)):
            signals[i], confs[i] = self.predict_one(X[i])
        return signals, confs

    # ── Accuracy on labelled data ─────────────────────────────────────────
    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate on held-out set. Returns accuracy + directional stats."""
        sigs, confs = self.predict(X)
        active  = y != 0
        if active.sum() == 0:
            return dict(accuracy=0.0, n_active=0, n_signals=0)
        correct = ((sigs == y) & active).sum()
        n_sigs  = (sigs != 0).sum()
        return dict(
            accuracy   = float(correct / active.sum()),
            n_active   = int(active.sum()),
            n_signals  = int(n_sigs),
            long_rate  = float((sigs == 1).sum() / len(sigs)),
            short_rate = float((sigs == -1).sum() / len(sigs)),
        )

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str = None):
        path = path or os.path.join(MODEL_DIR, "lorentzian.npz")
        np.savez_compressed(
            path,
            X_train=self.X_train if self.X_train is not None else np.array([]),
            y_train=self.y_train if self.y_train is not None else np.array([]),
            params=np.array([self.k, self.max_back, self.skip, self.threshold])
        )

    def load(self, path: str = None) -> bool:
        path = path or os.path.join(MODEL_DIR, "lorentzian.npz")
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=True)
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        if len(data["params"]) == 4:
            self.k, self.max_back, self.skip, self.threshold = data["params"]
        return True


# ── Label builder ──────────────────────────────────────────────────────────
def build_labels(closes: np.ndarray, lookahead: int = 1,
                 threshold_pct: float = 0.0005) -> np.ndarray:
    """
    y[i] = +1 if close[i+lookahead] > close[i] * (1 + threshold_pct)
           -1 if close[i+lookahead] < close[i] * (1 - threshold_pct)
            0 otherwise (too small to care)
    """
    n = len(closes)
    y = np.zeros(n, dtype=np.int8)
    for i in range(n - lookahead):
        ret = (closes[i+lookahead] - closes[i]) / closes[i]
        if ret >  threshold_pct: y[i] =  1
        elif ret < -threshold_pct: y[i] = -1
    return y


# ── Rational quadratic kernel smoother (optional) ─────────────────────────
def rq_kernel_smooth(signals: np.ndarray, confs: np.ndarray,
                     h: float = 8.0, alpha: float = 1.0) -> np.ndarray:
    """
    Smooth raw vote signals with a rational quadratic kernel.
    This is the same smoother jdehorty uses in his TradingView indicator.

    K(x,y) = (1 + |x-y|² / (2·α·h²))^(−α)

    Returns smoothed signal array.
    """
    n = len(signals)
    smoothed = np.zeros(n)
    for i in range(n):
        weights = np.zeros(n)
        for j in range(n):
            dist = abs(i - j)
            weights[j] = (1 + dist**2 / (2 * alpha * h**2)) ** (-alpha)
        weights *= confs          # weight by confidence
        total = weights.sum()
        if total > 1e-8:
            smoothed[i] = (weights * signals).sum() / total
    return smoothed
