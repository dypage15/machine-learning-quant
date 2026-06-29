"""
PCA MNQ · ML Intelligence Dashboard — Cyberpunk / Neo Tokyo
"""
import streamlit as st
import numpy as np, pandas as pd, json, os
from datetime import datetime

st.set_page_config(
    page_title="PCA MNQ · ML INTELLIGENCE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_system.db     import init_db, get_features, get_trades, get_stats, get_latest_predictions
from ml_system.config import PRED_FILE, FEATURE_COLS, DB_PATH

init_db()

# ── Cyberpunk CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background-color: #000000 !important;
    color: #00ff9f !important;
}
.stApp { background: #000000; }
.stApp::before {
    content: "";
    position: fixed; top:0; left:0; right:0; bottom:0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,
        rgba(0,255,159,0.012) 2px, rgba(0,255,159,0.012) 4px);
    pointer-events: none; z-index: 9999;
}
[data-testid="stSidebar"] {
    background: #03030d !important;
    border-right: 1px solid #00ff9f22 !important;
}
[data-testid="stSidebar"] * { color: #00ff9f !important; }
h1 { font-family:'Orbitron',sans-serif !important; color:#00ff9f !important;
     text-shadow:0 0 20px #00ff9f88; font-size:1.5rem !important; }
h2 { font-family:'Orbitron',sans-serif !important; color:#ff2d78 !important;
     text-shadow:0 0 10px #ff2d7866; font-size:0.85rem !important;
     letter-spacing:3px; margin-top:20px !important; }
h3 { font-family:'Orbitron',sans-serif !important; color:#7b2fff !important; font-size:0.8rem !important; }
[data-testid="metric-container"] {
    background:#03030d !important; border:1px solid #00ff9f33 !important;
    border-radius:3px !important; padding:10px 14px !important;
}
[data-testid="metric-container"] label {
    color:#00ff9f66 !important; font-size:0.6rem !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#00ff9f !important; font-family:'Orbitron',sans-serif !important;
    font-size:1.3rem !important; text-shadow:0 0 8px #00ff9f55;
}
.stTabs [data-baseweb="tab-list"] {
    background:#03030d !important; border-bottom:1px solid #00ff9f22 !important;
}
.stTabs [data-baseweb="tab"] {
    color:#00ff9f55 !important; font-family:'Share Tech Mono',monospace !important;
    font-size:0.7rem !important; letter-spacing:2px !important;
    text-transform:uppercase !important; border-radius:0 !important;
    border-bottom:2px solid transparent !important; padding:8px 18px !important;
}
.stTabs [aria-selected="true"] {
    color:#00ff9f !important; border-bottom:2px solid #00ff9f !important;
    background:#00ff9f08 !important;
}
.stButton button {
    background:transparent !important; border:1px solid #00ff9f !important;
    color:#00ff9f !important; font-family:'Share Tech Mono',monospace !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
}
.stButton button:hover { background:#00ff9f18 !important; box-shadow:0 0 10px #00ff9f33 !important; }
.stProgress > div > div { background:#00ff9f !important; box-shadow:0 0 6px #00ff9f; }
.stProgress > div { background:#0a0a1a !important; }
hr { border-color:#00ff9f1a !important; }
.stCaption, small { color:#00ff9f44 !important; font-size:0.65rem !important; letter-spacing:1px !important; }
[data-testid="stDataFrame"] { border:1px solid #00ff9f1a !important; }
</style>
""", unsafe_allow_html=True)

CLR = {
    "LONG":    "#00ff9f",
    "SHORT":   "#ff2d78",
    "NEUTRAL": "#7b2fff",
    "A":"#00ff9f","B":"#39d0ff","C":"#ffcc00","D":"#ff6b35","E":"#ff2d78","F":"#880033",
}
def _gc(k): return CLR.get(k, "#7b2fff")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 20px'>
        <div style='font-family:Orbitron,sans-serif;font-size:1rem;color:#00ff9f;
                    text-shadow:0 0 16px #00ff9f;letter-spacing:4px'>⚡ PCA·MNQ·ML</div>
        <div style='color:#00ff9f33;font-size:0.55rem;letter-spacing:5px;margin-top:6px'>
            LORENTZIAN + LSTM ENGINE
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    stats = get_stats()
    st.markdown("<div style='color:#00ff9f33;font-size:0.55rem;letter-spacing:4px;margin-bottom:8px'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    st.metric("MNQ BARS",      stats["n_candles"])
    st.metric("FEATURE ROWS",  stats["n_features"])
    st.metric("CLOSED TRADES", stats["n_trades"])
    if stats["n_trades"] > 0:
        st.metric("WIN RATE",  f"{stats['win_rate']:.1%}")
        st.metric("TOTAL P&L", f"${stats['total_pnl']:,.0f}")
    st.divider()
    if stats["last_model"]:
        lm = stats["last_model"]
        st.markdown(f"""
        <div style='color:#00ff9f33;font-size:0.55rem;letter-spacing:4px'>LAST RETRAIN</div>
        <div style='color:#00ff9f;font-size:0.75rem;margin-top:4px'>{lm['model_type'].upper()}</div>
        <div style='color:#00ff9f44;font-size:0.6rem'>{lm['trained_at'][:16]}</div>
        """, unsafe_allow_html=True)
        if lm.get("val_accuracy"):
            st.metric("VAL ACC", f"{lm['val_accuracy']:.3f}")
    st.divider()
    st.button("⟳  REFRESH", use_container_width=True)

# ── Title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='border-bottom:1px solid #00ff9f22;padding-bottom:10px;margin-bottom:2px'>
    <span style='font-family:Orbitron,sans-serif;font-size:1.4rem;color:#00ff9f;
                 text-shadow:0 0 24px #00ff9f;letter-spacing:5px'>
        ⚡ PCA MNQ · ML INTELLIGENCE
    </span>
    <span style='color:#00ff9f2a;font-size:0.6rem;margin-left:16px;letter-spacing:3px'>
        LORENTZIAN KNN + LSTM + PCA SIGNAL ENGINE
    </span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["◈  NEXT SIGNAL", "◈  CANDLE GRADES", "◈  TRADE LOG", "◈  MODEL INTEL"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — NEXT SIGNAL
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    pred_data = None
    if os.path.exists(PRED_FILE):
        with open(PRED_FILE) as f:
            pred_data = json.load(f)

    if pred_data:
        pred     = pred_data.get("prediction", {})
        gen_at   = pred_data.get("generated_at", "")[:16]
        sig_text = pred.get("signal_text", "NEUTRAL")
        ensemble = pred.get("ensemble", 0.0)
        grade    = pred.get("grade", pred_data.get("last_grade", "?"))
        zscore   = pred_data.get("last_zscore", 0.0)
        loren    = pred.get("lorentzian", 0.0)
        lstm_c   = pred.get("lstm_conf", 0.0)
        sc, gc   = _gc(sig_text), _gc(grade)

        st.markdown(f"""
        <div style='display:flex;gap:14px;margin:18px 0'>
            <div style='flex:2;background:#03030d;border:2px solid {sc};border-radius:3px;
                        padding:28px;text-align:center;box-shadow:0 0 30px {sc}22'>
                <div style='font-family:Orbitron,sans-serif;font-size:2.8rem;font-weight:900;
                            color:{sc};text-shadow:0 0 40px {sc};letter-spacing:8px'>{sig_text}</div>
                <div style='color:{sc}55;font-size:0.6rem;letter-spacing:5px;margin-top:8px'>ML ENSEMBLE SIGNAL</div>
                <div style='font-family:Orbitron,sans-serif;font-size:1.8rem;color:{sc};
                            margin-top:12px;text-shadow:0 0 16px {sc}'>{ensemble:+.3f}</div>
                <div style='color:{sc}44;font-size:0.55rem;letter-spacing:4px'>ENSEMBLE SCORE</div>
            </div>
            <div style='flex:1;background:#03030d;border:2px solid {gc};border-radius:3px;
                        padding:28px;text-align:center;box-shadow:0 0 20px {gc}22'>
                <div style='font-family:Orbitron,sans-serif;font-size:2.8rem;font-weight:900;
                            color:{gc};text-shadow:0 0 30px {gc}'>{grade}</div>
                <div style='color:{gc}55;font-size:0.6rem;letter-spacing:5px;margin-top:8px'>CANDLE GRADE</div>
            </div>
            <div style='flex:1;background:#03030d;border:1px solid #7b2fff33;border-radius:3px;
                        padding:28px;text-align:center'>
                <div style='font-family:Orbitron,sans-serif;font-size:1.6rem;
                            color:#7b2fff;text-shadow:0 0 14px #7b2fff'>{zscore:+.2f}σ</div>
                <div style='color:#7b2fff55;font-size:0.6rem;letter-spacing:4px;margin-top:8px'>PCA Z-SCORE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("LORENTZIAN", f"{loren:+.3f}")
        c2.metric("LSTM CONF",  f"{lstm_c:.1%}")
        c3.metric("LSTM DIR",   {-1:"SHORT",0:"NEUTRAL",1:"LONG"}.get(pred.get("lstm_dir",0),"—"))
        c4.metric("GENERATED",  gen_at + " UTC")

        bar_val = (ensemble + 1) / 2
        st.markdown(f"""
        <div style='margin:14px 0 4px'>
            <div style='color:#00ff9f33;font-size:0.55rem;letter-spacing:4px;margin-bottom:5px'>
                DIRECTIONAL CONFIDENCE ── {abs(ensemble):.1%}
            </div>
            <div style='background:#0a0a1a;border-radius:2px;height:5px;border:1px solid #00ff9f1a'>
                <div style='background:{sc};width:{bar_val*100:.1f}%;height:100%;
                            border-radius:2px;box-shadow:0 0 8px {sc}'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#03030d;border:1px solid #ff2d7833;border-radius:3px;
                    padding:40px;text-align:center;margin:32px 0'>
            <div style='color:#ff2d78;font-family:Orbitron,sans-serif;letter-spacing:4px;font-size:0.9rem'>
                NO SIGNAL DATA
            </div>
            <div style='color:#ff2d7844;font-size:0.65rem;margin-top:10px;letter-spacing:3px'>
                RUN: python run_nightly.py --import-json bars.json
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<h2>RECENT PREDICTIONS</h2>", unsafe_allow_html=True)
    preds = get_latest_predictions(20)
    if preds:
        df = pd.DataFrame(preds)
        for col in ["lorentzian","lstm_conf","ensemble"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:+.3f}" if x is not None else "—")
        display_cols = [c for c in ["created_at","signal_text","grade","lorentzian",
                                     "lstm_dir","lstm_conf","ensemble"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, height=260)
    else:
        st.caption("// predictions populate after first pipeline run")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — CANDLE GRADES
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    feat_rows = get_features(500)
    if feat_rows:
        df = pd.DataFrame(feat_rows)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h2>GRADE DISTRIBUTION</h2>", unsafe_allow_html=True)
            from collections import Counter
            counts = Counter(df["candle_grade"].dropna())
            total  = len(df)
            for letter in ["A","B","C","D","E","F"]:
                cnt = counts.get(letter, 0)
                pct = cnt / total * 100 if total else 0
                gc  = _gc(letter)
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:10px;margin:5px 0'>
                    <span style='font-family:Orbitron,sans-serif;font-size:1.1rem;
                                 color:{gc};text-shadow:0 0 8px {gc};width:18px'>{letter}</span>
                    <div style='flex:1;background:#0a0a1a;border-radius:1px;height:4px'>
                        <div style='background:{gc};width:{pct:.1f}%;height:100%;
                                    box-shadow:0 0 5px {gc}'></div>
                    </div>
                    <span style='color:{gc}77;font-size:0.65rem;width:50px;text-align:right'>
                        {cnt} · {pct:.0f}%</span>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("<h2>RECENT CANDLES</h2>", unsafe_allow_html=True)
            disp = df.tail(100)[["ts","candle_grade","grade_score","pca_zscore",
                                  "atr_ratio","vol_ratio","rsi14"]].copy()
            disp["dt"] = pd.to_datetime(disp["ts"], unit="s").dt.strftime("%m/%d %H:%M")
            disp = disp.drop(columns=["ts"]).set_index("dt")
            for col in ["grade_score","pca_zscore","atr_ratio","vol_ratio","rsi14"]:
                if col in disp.columns:
                    disp[col] = disp[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            st.dataframe(disp.tail(50), use_container_width=True, height=360)

        st.markdown("<h2>GRADE SCORE + PCA Z-SCORE TIMELINE</h2>", unsafe_allow_html=True)
        if "grade_score" in df.columns:
            df["dt"] = pd.to_datetime(df["ts"], unit="s")
            st.line_chart(df.set_index("dt")[["grade_score","pca_zscore"]].tail(200),
                          color=["#00ff9f","#7b2fff"])
    else:
        st.markdown("<div style='color:#ff2d78;font-family:Orbitron,sans-serif;text-align:center;"
                    "padding:40px;letter-spacing:4px'>NO FEATURE DATA — RUN THE PIPELINE</div>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADE LOG
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    trades = get_trades(500)
    if trades:
        df     = pd.DataFrame(trades)
        closed = df[df["actual_outcome"].notna()]
        if len(closed) > 0:
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("TRADES",        len(closed))
            c2.metric("WIN RATE",      f"{closed['actual_outcome'].mean():.1%}")
            c3.metric("TOTAL P&L",     f"${closed['pnl'].sum():,.0f}")
            c4.metric("AVG P&L",       f"${closed['pnl'].mean():,.0f}")
            wins   = closed[closed["actual_outcome"]==1]["pnl"]
            losses = closed[closed["actual_outcome"]==0]["pnl"]
            pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999
            c5.metric("PROFIT FACTOR", f"{pf:.2f}")

            if "mfe" in closed.columns and closed["mfe"].notna().any():
                st.markdown("<h2>EXCURSION ANALYSIS (MFE / MAE)</h2>", unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("AVG MFE", f"{closed['mfe'].mean():.1f} pts")
                m2.metric("AVG MAE", f"{closed['mae'].mean():.1f} pts")
                m3.metric("MAX MFE", f"{closed['mfe'].max():.1f} pts")
                m4.metric("MAX MAE", f"{closed['mae'].max():.1f} pts")

            st.markdown("<h2>EQUITY CURVE</h2>", unsafe_allow_html=True)
            eq = 100000 + closed["pnl"].cumsum()
            st.line_chart(pd.DataFrame({"Equity ($)": eq.values}))

            if "candle_grade" in closed.columns:
                st.markdown("<h2>WIN RATE BY GRADE</h2>", unsafe_allow_html=True)
                grade_wr = closed.groupby("candle_grade")["actual_outcome"].agg(["mean","count"])
                grade_wr.columns = ["Win Rate","Count"]
                grade_wr["Win Rate"] = grade_wr["Win Rate"].apply(lambda x: f"{x:.1%}")
                st.dataframe(grade_wr, use_container_width=True)

        st.markdown("<h2>ALL TRADES</h2>", unsafe_allow_html=True)
        dcols = [c for c in ["session_date","direction","entry_price","exit_price",
                              "exit_reason","pnl","bars_held","mfe","mae",
                              "candle_grade","lorentzian_vote","lstm_conf","actual_outcome"]
                 if c in df.columns]
        st.dataframe(df[dcols].tail(100), use_container_width=True, height=360)
    else:
        st.markdown("""
        <div style='background:#03030d;border:1px solid #7b2fff33;border-radius:3px;
                    padding:40px;text-align:center;color:#7b2fff;
                    font-family:Orbitron,sans-serif;letter-spacing:4px'>
            NO TRADES LOGGED YET
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INTEL
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    feat_rows = get_features(2000)
    tr_data   = get_trades(500)

    if feat_rows and len(tr_data) > 20:
        feat_df  = pd.DataFrame(feat_rows)
        trade_df = pd.DataFrame(tr_data)
        closed   = trade_df[trade_df["actual_outcome"].notna()]
        if len(closed) > 10:
            merged = closed.merge(feat_df, left_on="entry_ts", right_on="ts", how="inner")
            if len(merged) > 5:
                st.markdown("<h2>FEATURE → OUTCOME CORRELATION</h2>", unsafe_allow_html=True)
                corrs = {col: merged[col].corr(merged["actual_outcome"])
                         for col in FEATURE_COLS if col in merged.columns
                         and not np.isnan(merged[col].corr(merged["actual_outcome"]))}
                if corrs:
                    corr_df = pd.DataFrame.from_dict(corrs, orient="index", columns=["Correlation"])
                    st.bar_chart(corr_df.sort_values("Correlation", ascending=False))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h2>LORENTZIAN KNN</h2>", unsafe_allow_html=True)
        lp = os.path.join(os.path.dirname(__file__), "models", "lorentzian.npz")
        if os.path.exists(lp):
            d  = np.load(lp)
            n  = len(d.get("X_train", []))
            nf = d.get("X_train", np.array([])).shape[-1] if n > 0 else 0
            st.metric("TRAINING SAMPLES", n)
            st.metric("FEATURE DIMS",     nf)
        else:
            st.caption("// not trained yet")
        st.markdown("""
        <div style='background:#03030d;border:1px solid #00ff9f1a;border-radius:3px;
                    padding:14px;margin-top:10px;font-size:0.68rem;color:#00ff9f77;line-height:1.9'>
            <b style='color:#00ff9f'>DISTANCE:</b> Σ log(1 + |xᵢ − yᵢ|)<br>
            · k=8 nearest neighbours<br>
            · Every 4th bar for diversity<br>
            · 55% vote threshold to signal
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<h2>LSTM NEURAL NET</h2>", unsafe_allow_html=True)
        lp2 = os.path.join(os.path.dirname(__file__), "models", "lstm.pt")
        if os.path.exists(lp2):
            try:
                import torch
                ckpt = torch.load(lp2, map_location="cpu")
                hist = ckpt.get("history", {})
                if hist.get("val_acc"):
                    st.line_chart(pd.DataFrame({"Val Acc": hist["val_acc"]}))
            except Exception as e:
                st.caption(f"// {e}")
        else:
            st.caption("// not trained yet")
        st.markdown("""
        <div style='background:#03030d;border:1px solid #7b2fff1a;border-radius:3px;
                    padding:14px;margin-top:10px;font-size:0.68rem;color:#7b2fff88;line-height:1.9'>
            <b style='color:#7b2fff'>ARCH:</b> 2-layer LSTM · 64h · BN · Dropout<br>
            · 20-bar sequences · 22 features<br>
            · Warm-start nightly retrain<br>
            · 3-class: LONG / NEUTRAL / SHORT
        </div>""", unsafe_allow_html=True)

    st.markdown("<h2>ENSEMBLE FORMULA</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#03030d;border:1px solid #ff2d7822;border-radius:3px;
                padding:18px;font-size:0.72rem;line-height:2.1'>
        <span style='color:#00ff9f'>ensemble</span> =
        <span style='color:#7b2fff'>0.4</span>×Lorentzian_vote×conf +
        <span style='color:#7b2fff'>0.6</span>×LSTM_dir×LSTM_conf<br>
        <span style='color:#00ff9f'>LONG</span>  if ensemble > <span style='color:#ff2d78'>+0.20</span> &nbsp;|&nbsp;
        <span style='color:#ff2d78'>SHORT</span> if ensemble < <span style='color:#ff2d78'>-0.20</span> &nbsp;|&nbsp;
        <span style='color:#7b2fff'>NEUTRAL</span> otherwise<br>
        <span style='color:#00ff9f22;font-size:0.6rem'>
        22 FEATURES: pca_zscore · pca_cs · log_ret×5 · beta_es · trend_align ·
        atr_ratio · body_ratio · upper/lower_wick · rsi14 · bb_pos · vol_ratio ·
        grade_score · hour_sin/cos · dow_sin/cos
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='border-top:1px solid #00ff9f1a;margin-top:24px;padding-top:8px;
            text-align:center;color:#00ff9f1a;font-size:0.55rem;letter-spacing:4px'>
    PCA MNQ ML INTELLIGENCE  ·  RELOAD TO REFRESH  ·  RUN run_nightly.py AFTER MARKET CLOSE
</div>""", unsafe_allow_html=True)
