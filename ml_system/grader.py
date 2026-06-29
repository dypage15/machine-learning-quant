"""
Candle grader — assigns A–F grade to each bar based on composite feature score.
"""
import numpy as np
from .config import GRADE_WEIGHTS, GRADE_THRESHOLDS


def grade_candle(feat: dict) -> tuple[str, float]:
    """
    Returns (letter_grade, score_0_to_1) for a single feature dict.

    Scoring logic:
      pca_zscore  → |z| mapped to 0–1  (higher = stronger signal)
      vol_ratio   → capped at 3, mapped 0–1
      body_ratio  → already 0–1  (1 = full body, no wicks)
      trend_align → 0 or 1  (1 = price above EMA20)
      atr_ratio   → ideal range 0.8–1.5 → score peaks at 1.0
      wick_score  → depends on cs direction
    """
    z   = abs(feat.get("pca_zscore", 0.0))
    vol = feat.get("vol_ratio", 1.0)
    body= feat.get("body_ratio", 0.5)
    ta  = (feat.get("trend_align", 0.0) + 1) / 2    # -1/0/1 → 0/0.5/1
    atr = feat.get("atr_ratio", 1.0)
    lw  = feat.get("lower_wick", 0.0)
    uw  = feat.get("upper_wick", 0.0)
    cs  = feat.get("_cs", 0)

    # Individual sub-scores (each 0–1)
    pca_score  = min(z / 2.0, 1.0)                  # saturates at 2σ
    vol_score  = min(vol / 2.0, 1.0)
    body_score = body
    trend_score= ta
    # ATR ratio: ideal ≈ 1.0 (one average-range bar), penalise tiny/huge
    atr_score  = max(0.0, 1.0 - abs(atr - 1.0) / 1.5)
    # Wick favours lower wicks for longs, upper wicks for shorts
    if cs >= 0:   wick_score = lw          # long: lower wick = bounce
    else:         wick_score = uw          # short: upper wick = rejection

    composite = (
        GRADE_WEIGHTS["pca_zscore"]  * pca_score  +
        GRADE_WEIGHTS["vol_ratio"]   * vol_score  +
        GRADE_WEIGHTS["body_ratio"]  * body_score +
        GRADE_WEIGHTS["trend_align"] * trend_score+
        GRADE_WEIGHTS["atr_ratio"]   * atr_score  +
        GRADE_WEIGHTS["wick_score"]  * wick_score
    )

    thresholds = GRADE_THRESHOLDS       # [0.80, 0.65, 0.50, 0.35, 0.20]
    letters    = ["A", "B", "C", "D", "E", "F"]
    for letter, thresh in zip(letters, thresholds):
        if composite >= thresh:
            return letter, round(composite, 4)
    return "F", round(composite, 4)


def grade_all(feature_rows: list[dict]) -> list[dict]:
    """Add candle_grade and grade_score fields to each feature dict in-place."""
    for row in feature_rows:
        g, s = grade_candle(row)
        row["candle_grade"] = g
        row["grade_score"]  = s
    return feature_rows


def grade_distribution(feature_rows: list[dict]) -> dict:
    """Returns count per grade letter."""
    from collections import Counter
    counts = Counter(r.get("candle_grade", "?") for r in feature_rows)
    return dict(sorted(counts.items()))
