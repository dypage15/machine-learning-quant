"""
Streamlit dashboard — live view of ML predictions, candle grades, trade log.
Run with: streamlit run ml_system/dashboard.py
"""
import streamlit as st
import numpy as np, pandas as pd, json, os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCA MNQ · ML Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from our system ────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_system.db     import init_db, get_features, get_trades, get_stats, get_latest_predictions
from ml_system.config import PRED_FILE, FEATURE_COLS, DB_PATH

init_db()

# ── Grade colours ──────────────────────────────────────────────────────────
GRADE_COLORS = {"A": "#22c55e", "B": "#84cc16", "C": "#eab308",
                "D": "#f97316", "E": "#ef4444", "F": "#6b7280"}

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 PCA MNQ ML")
    st.caption("Lorentzian + LSTM Intelligence")
    st.divider()

    stats = get_stats()
    st.metric("Candles in DB",  stats["n_candles"])
    st.metric("Feature rows",   stats["n_features"])
    st.metric("Closed trades",  stats["n_trades"])
    if stats["n_trades"] > 0:
        st.metric("Win Rate",   f"{stats['win_rate']:.1%}")
        st.metric("Total P&L",  f"${stats['total_pnl']:,.2f}")

    st.divider()
    if stats["last_model"]:
        lm = stats["last_model"]
        st.caption(f"Last retrain: {lm['model_type']}")
        st.caption(f"At: {lm['trained_at'][:16]}")
        if lm.get("val_accuracy"):
            st.caption(f"Val acc: {lm['val_accuracy']:.3f}")

    st.divider()
    refresh = st.button("🔄 Refresh", use_container_width=True)


# ── Main content ──────────────────────────────────────────────────────────
st.title("📊 PCA MNQ · ML Intelligence Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Next Signal", "📈 Candle Grades", "💰 Trade Log", "🧠 Model Insights"]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: Next Signal
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    # Load latest prediction JSON
    pred_data = None
    if os.path.exists(PRED_FILE):
        with open(PRED_FILE) as f:
            pred_data = json.load(f)

    if pred_data:
        pred     = pred_data.get("prediction", {})
        gen_at   = pred_data.get("generated_at", "")[:16]
        sig_text = pred.get("signal_text", "NEUTRAL")
        ensemble = pred.get("ensemble", 0.0)
        grade    = pred.get("last_grade", "?")
        zscore   = pred_data.get("last_zscore", 0.0)

        col1, col2, col3, col4 = st.columns(4)

        sig_color = "#22c55e" if sig_text=="LONG" else "#ef4444" if sig_text=="SHORT" else "#6b7280"
        col1.markdown(f"""
            <div style='background:{sig_color}22;border:2px solid {sig_color};
                        border-radius:12px;padding:16px;text-align:center'>
                <h1 style='color:{sig_color};margin:0'>{sig_text}</h1>
                <p style='color:#94a3b8;margin:0;font-size:12px'>ML Ensemble Signal</p>
            </div>
        """, unsafe_allow_html=True)

        gc = GRADE_COLORS.get(grade, "#6b7280")
        col2.markdown(f"""
            <div style='background:{gc}22;border:2px solid {gc};
                        border-radius:12px;padding:16px;text-align:center'>
                <h1 style='color:{gc};margin:0'>{grade}</h1>
                <p style='color:#94a3b8;margin:0;font-size:12px'>Candle Grade</p>
            </div>
        """, unsafe_allow_html=True)

        col3.metric("Ensemble Score",  f"{ensemble:+.3f}")
        col3.metric("PCA Z-Score",     f"{zscore:+.2f}")
        col4.metric("Lorentzian",      f"{pred.get('lorentzian',0):+.3f}")
        col4.metric("LSTM Confidence", f"{pred.get('lstm_conf',0):.1%}")

        st.caption(f"Generated at {gen_at} UTC")

        # Confidence bar
        bar_val = (ensemble + 1) / 2  # map -1..1 → 0..1
        st.progress(bar_val, text=f"Directional confidence: {abs(ensemble):.1%}")

    else:
        st.info("No predictions yet. Run the nightly pipeline first.\n\n"
                "```python run_nightly.py```")

    # Recent predictions table
    st.subheader("Recent Predictions")
    preds = get_latest_predictions(20)
    if preds:
        df = pd.DataFrame(preds)
        for col in ["lorentzian","lstm_conf","ensemble"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:+.3f}" if x is not None else "—")
        st.dataframe(df[["created_at","grade","lorentzian","lstm_dir",
                          "lstm_conf","ensemble"]],
                     use_container_width=True, height=300)
    else:
        st.info("Predictions will appear here after the first nightly run.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: Candle Grades
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    feat_rows = get_features(500)
    if feat_rows:
        df = pd.DataFrame(feat_rows)

        # Grade distribution
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Grade Distribution")
            from collections import Counter
            counts = Counter(df["candle_grade"].dropna())
            for letter in ["A","B","C","D","E","F"]:
                cnt = counts.get(letter, 0)
                pct = cnt / len(df) * 100
                gc  = GRADE_COLORS[letter]
                st.markdown(
                    f"<span style='color:{gc};font-size:18px;font-weight:700'>{letter}</span>"
                    f" &nbsp; {cnt} bars &nbsp; <span style='color:#6b7280'>{pct:.1f}%</span>",
                    unsafe_allow_html=True
                )
                st.progress(pct/100)

        with col2:
            st.subheader("Recent Candles")
            display = df.tail(100)[["ts","candle_grade","grade_score",
                                     "pca_zscore","atr_ratio","vol_ratio",
                                     "body_ratio","rsi14"]].copy()
            display["dt"] = pd.to_datetime(display["ts"], unit="s").dt.strftime("%m/%d %H:%M")
            display = display.drop(columns=["ts"]).set_index("dt")
            for col in ["grade_score","pca_zscore","atr_ratio","vol_ratio","body_ratio","rsi14"]:
                if col in display.columns:
                    display[col] = display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

            st.dataframe(display.tail(50), use_container_width=True, height=400)

        # Grade over time
        st.subheader("Grade Score Over Time")
        if "grade_score" in df.columns:
            df["dt"] = pd.to_datetime(df["ts"], unit="s")
            chart_df = df.set_index("dt")[["grade_score","pca_zscore"]].tail(200)
            st.line_chart(chart_df)
    else:
        st.info("No feature data yet. Run the nightly pipeline to generate candle grades.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: Trade Log
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    trades = get_trades(500)
    if trades:
        df = pd.DataFrame(trades)

        # Summary metrics
        closed = df[df["actual_outcome"].notna()]
        if len(closed) > 0:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades",   len(closed))
            col2.metric("Win Rate",       f"{closed['actual_outcome'].mean():.1%}")
            col3.metric("Total P&L",      f"${closed['pnl'].sum():,.2f}")
            col4.metric("Avg P&L/Trade",  f"${closed['pnl'].mean():,.2f}")
            wins   = closed[closed["actual_outcome"]==1]["pnl"]
            losses = closed[closed["actual_outcome"]==0]["pnl"]
            pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999
            col5.metric("Profit Factor",  f"{pf:.2f}")

            # Equity curve
            st.subheader("Equity Curve")
            eq = 100000 + closed["pnl"].cumsum()
            st.line_chart(pd.DataFrame({"Equity ($)": eq.values}))

            # Grade win rate
            st.subheader("Win Rate by Candle Grade")
            if "candle_grade" in closed.columns:
                grade_wr = closed.groupby("candle_grade")["actual_outcome"].agg(["mean","count"])
                grade_wr.columns = ["Win Rate","Count"]
                grade_wr["Win Rate"] = grade_wr["Win Rate"].apply(lambda x: f"{x:.1%}")
                st.dataframe(grade_wr, use_container_width=True)

        # Full trade table
        st.subheader("All Trades")
        display_cols = [c for c in ["session_date","direction","entry_price","exit_price",
                                     "exit_reason","pnl","bars_held","candle_grade",
                                     "lorentzian_vote","lstm_conf","actual_outcome"]
                        if c in df.columns]
        st.dataframe(df[display_cols].tail(100), use_container_width=True, height=400)
    else:
        st.info("No trades logged yet. Trades are recorded automatically when the pipeline runs.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: Model Insights
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Feature Importance (PCA Z-Score vs Outcome)")

    feat_rows = get_features(2000)
    trades    = get_trades(500)

    if feat_rows and len(trades) > 20:
        feat_df  = pd.DataFrame(feat_rows)
        trade_df = pd.DataFrame(trades)
        closed   = trade_df[trade_df["actual_outcome"].notna()]

        if len(closed) > 10:
            # Merge features with trade outcomes by entry timestamp
            merged = closed.merge(feat_df, left_on="entry_ts", right_on="ts", how="inner")
            if len(merged) > 5:
                st.subheader("Feature Correlation with Trade Outcome")
                corrs = {}
                for col in FEATURE_COLS:
                    if col in merged.columns:
                        c = merged[col].corr(merged["actual_outcome"])
                        if not np.isnan(c):
                            corrs[col] = c
                if corrs:
                    corr_df = pd.DataFrame.from_dict(corrs, orient="index",
                                                     columns=["Correlation"])
                    corr_df = corr_df.sort_values("Correlation", ascending=False)
                    st.bar_chart(corr_df)

    # Lorentzian model info
    loren_path = os.path.join(os.path.dirname(__file__), "models", "lorentzian.npz")
    if os.path.exists(loren_path):
        data = np.load(loren_path)
        n_samples = len(data.get("X_train", []))
        st.metric("Lorentzian training samples", n_samples)
    else:
        st.info("Lorentzian model not trained yet.")

    # LSTM history
    lstm_path = os.path.join(os.path.dirname(__file__), "models", "lstm.pt")
    if os.path.exists(lstm_path):
        try:
            import torch
            ckpt = torch.load(lstm_path, map_location="cpu")
            hist = ckpt.get("history", {})
            if hist.get("val_acc"):
                st.subheader("LSTM Validation Accuracy Over Epochs")
                st.line_chart(pd.DataFrame({"Val Accuracy": hist["val_acc"]}))
            if hist.get("train_loss"):
                st.subheader("LSTM Training Loss")
                st.line_chart(pd.DataFrame({"Train Loss": hist["train_loss"]}))
        except Exception as e:
            st.caption(f"Could not load LSTM history: {e}")
    else:
        st.info("LSTM model not trained yet (needs 100+ bars).")

    st.subheader("How the Models Work")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Lorentzian KNN**
- Finds the 8 most similar past bars using Lorentzian distance: `Σ log(1 + |xᵢ − yᵢ|)`
- More robust to outliers than Euclidean distance
- No retraining needed — grows with every bar
- Votes: majority of 8 neighbours determines direction
- Threshold: needs >55% vote fraction to signal
        """)
    with col2:
        st.markdown("""
**LSTM Neural Network**
- Looks at sequences of 20 bars (20 hours of context)
- 2-layer LSTM with 64 hidden units + BatchNorm + Dropout
- Outputs: Short / Neutral / Long + confidence
- Retrained nightly on all historical data
- Warm-starts from previous weights (transfer learning)
        """)

    st.markdown("""
**Ensemble Logic**
```
ensemble = 0.4 × Lorentzian_vote × confidence + 0.6 × LSTM_dir × LSTM_conf
LONG  if ensemble > +0.2
SHORT if ensemble < −0.2
NEUTRAL otherwise
```
    """)


# ── Auto-refresh note ─────────────────────────────────────────────────────
st.caption("Dashboard auto-refreshes on browser reload. "
           "Run `python run_nightly.py` each evening to update predictions.")
