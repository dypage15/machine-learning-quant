"""
LSTM Neural Network for sequential pattern recognition.
Uses PyTorch. Falls back to a simple numpy perceptron if PyTorch unavailable.
"""
import numpy as np
import os, json, logging
from .config import (LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
                     LSTM_EPOCHS, LSTM_LR, LSTM_BATCH, LSTM_MIN_ROWS,
                     N_FEATURES, MODEL_DIR)

log = logging.getLogger(__name__)

# ── PyTorch availability ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    log.warning("PyTorch not installed — using fallback linear model")


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch LSTM model
# ═══════════════════════════════════════════════════════════════════════════
if TORCH_OK:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size=N_FEATURES, hidden=LSTM_HIDDEN,
                     layers=LSTM_LAYERS, dropout=LSTM_DROPOUT):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers,
                                batch_first=True, dropout=dropout if layers>1 else 0)
            self.bn   = nn.BatchNorm1d(hidden)
            self.drop = nn.Dropout(dropout)
            self.fc   = nn.Linear(hidden, 3)   # 3 classes: short(-1) neutral(0) long(+1)

        def forward(self, x):
            out, _ = self.lstm(x)          # (B, T, H)
            h      = out[:, -1, :]         # last timestep  (B, H)
            h      = self.bn(h)
            h      = self.drop(h)
            return self.fc(h)              # (B, 3)  logits


# ═══════════════════════════════════════════════════════════════════════════
# Fallback: simple online-learning linear model (scikit-learn SGD)
# ═══════════════════════════════════════════════════════════════════════════
class _LinearFallback:
    def __init__(self):
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True,
                                   random_state=42)
        self.fitted = False

    def partial_fit(self, X, y):
        classes = np.array([0, 1, 2])  # LABEL_MAP mapped values: 0=short, 1=neutral, 2=long
        self.model.partial_fit(X, y, classes=classes)
        self.fitted = True

    def predict_proba(self, X):
        if not self.fitted:
            return np.ones((len(X), 3)) / 3
        return self.model.predict_proba(X)


# ═══════════════════════════════════════════════════════════════════════════
# Public wrapper
# ═══════════════════════════════════════════════════════════════════════════
class LSTMPredictor:
    """
    Wraps either the PyTorch LSTM or the sklearn fallback.
    Interface is identical regardless of backend.
    """

    LABEL_MAP  = {-1: 0, 0: 1, 1: 2}    # class index for loss
    LABEL_IMAP = {0: -1, 1: 0, 2: 1}    # reverse

    def __init__(self):
        self.backend = "torch" if TORCH_OK else "linear"
        self.net     = None
        self.history = {"train_loss": [], "val_acc": []}
        self._build()

    def _build(self):
        if TORCH_OK:
            self.net    = _LSTMNet()
            self.optim  = torch.optim.Adam(self.net.parameters(), lr=LSTM_LR)
            self.crit   = nn.CrossEntropyLoss()
        else:
            self.net = _LinearFallback()

    # ── Sequence builder ─────────────────────────────────────────────────
    @staticmethod
    def make_sequences(X: np.ndarray, y: np.ndarray,
                       seq_len: int = LSTM_SEQ_LEN
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of seq_len over (X, y) to create 3-D input tensor.
        Returns X_seq (N, seq_len, F) and y_seq (N,).
        """
        N, F = X.shape
        if N < seq_len:
            return np.empty((0, seq_len, F)), np.empty(0, dtype=int)
        n_seq  = N - seq_len
        X_seq  = np.zeros((n_seq, seq_len, F), dtype=np.float32)
        y_seq  = np.zeros(n_seq, dtype=np.int64)
        for i in range(n_seq):
            X_seq[i] = X[i:i+seq_len]
            y_seq[i] = LSTMPredictor.LABEL_MAP.get(int(y[i+seq_len]), 1)
        return X_seq, y_seq

    # ── Train / retrain ───────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray,
            val_split: float = 0.15) -> dict:
        """
        Full (re)train on all available data.
        X: (N, F) normalised features
        y: (N,)   labels  -1 / 0 / +1
        """
        if len(X) < LSTM_MIN_ROWS:
            log.warning(f"Only {len(X)} rows — skipping LSTM train (need {LSTM_MIN_ROWS})")
            return {"skipped": True}

        X_seq, y_seq = self.make_sequences(X, y)
        if len(X_seq) == 0:
            return {"skipped": True}

        # Train/val split
        cut = max(1, int(len(X_seq) * (1 - val_split)))
        X_tr, y_tr = X_seq[:cut], y_seq[:cut]
        X_val, y_val = X_seq[cut:], y_seq[cut:]

        if self.backend == "torch":
            return self._train_torch(X_tr, y_tr, X_val, y_val)
        else:
            # Flatten for linear model
            X_flat = X_seq.reshape(len(X_seq), -1)
            self.net.partial_fit(X_flat[:cut], y_seq[:cut])
            return {"backend": "linear", "n_train": cut}

    def _train_torch(self, X_tr, y_tr, X_val, y_val) -> dict:
        self.net.train()
        ds     = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        loader = DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True)

        best_val_acc = 0.0
        for epoch in range(LSTM_EPOCHS):
            total_loss = 0.0
            for xb, yb in loader:
                self.optim.zero_grad()
                logits = self.net(xb)
                loss   = self.crit(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optim.step()
                total_loss += loss.item()

            # Validation
            if len(X_val) > 0:
                val_acc = self._eval_torch(X_val, y_val)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                self.history["val_acc"].append(val_acc)
            self.history["train_loss"].append(total_loss / max(len(loader), 1))

            if (epoch+1) % 10 == 0:
                log.info(f"  Epoch {epoch+1}/{LSTM_EPOCHS}  loss={total_loss:.4f}"
                         f"  val_acc={best_val_acc:.3f}")

        return {"backend": "torch", "epochs": LSTM_EPOCHS,
                "best_val_acc": best_val_acc, "n_train": len(X_tr)}

    def _eval_torch(self, X_val, y_val) -> float:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X_val))
            preds  = logits.argmax(dim=1).numpy()
        self.net.train()
        return float((preds == y_val).mean())

    # ── Predict ───────────────────────────────────────────────────────────
    def predict_one(self, X_recent: np.ndarray) -> tuple[int, float]:
        """
        X_recent: (seq_len, F) — the last seq_len bars of normalised features.
        Returns (direction, confidence).
        """
        if self.net is None:
            return 0, 0.0

        seq = X_recent[-LSTM_SEQ_LEN:] if len(X_recent) >= LSTM_SEQ_LEN else None
        if seq is None:
            return 0, 0.0

        if self.backend == "torch":
            self.net.eval()
            with torch.no_grad():
                x    = torch.tensor(seq[None].astype(np.float32))  # (1, T, F)
                logits = self.net(x)
                probs  = torch.softmax(logits, dim=1).numpy()[0]
            self.net.train()
            cls  = int(probs.argmax())
            conf = float(probs[cls])
            return self.LABEL_IMAP[cls], conf
        else:
            flat = seq.flatten().reshape(1, -1)
            if not self.net.fitted:
                return 0, 0.0
            probs = self.net.predict_proba(flat)[0]
            cls   = int(probs.argmax())
            conf  = float(probs[cls])
            return self.LABEL_IMAP.get(cls, 0), conf

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str = None):
        path = path or os.path.join(MODEL_DIR, "lstm.pt" if TORCH_OK else "lstm_linear.npz")
        if TORCH_OK and self.net is not None:
            torch.save({
                "state_dict": self.net.state_dict(),
                "history":    self.history,
            }, path)
        elif not TORCH_OK and hasattr(self.net, "model"):
            import pickle
            with open(path.replace(".pt", ".pkl"), "wb") as f:
                pickle.dump(self.net.model, f)
        log.info(f"LSTM saved → {path}")

    def load(self, path: str = None) -> bool:
        path = path or os.path.join(MODEL_DIR, "lstm.pt" if TORCH_OK else "lstm.pkl")
        if not os.path.exists(path):
            return False
        if TORCH_OK:
            try:
                ckpt = torch.load(path, map_location="cpu")
                self.net.load_state_dict(ckpt["state_dict"])
                self.history = ckpt.get("history", self.history)
                log.info(f"LSTM loaded ← {path}")
            except (RuntimeError, Exception) as e:
                # Shape mismatch (e.g. N_FEATURES changed) — start fresh
                log.warning(f"LSTM checkpoint incompatible ({e}), starting fresh")
                self._build()
                return False
        else:
            import pickle
            try:
                with open(path, "rb") as f:
                    self.net.model = pickle.load(f)
                self.net.fitted = True
                log.info(f"LSTM loaded ← {path}")
            except Exception as e:
                log.warning(f"LSTM linear checkpoint failed ({e}), starting fresh")
                return False
        return True
