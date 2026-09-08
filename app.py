# app.py
import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Modular imports
from config import (
    DEFAULT_NUM_BANDS,
    DEFAULT_TIME_SLOTS,
    DEFAULT_SEED,
    DEFAULT_NOISE_FLOOR_DBM,
    DEFAULT_DETECTION_THRESHOLD_DBM,
    LEARNING_RATE,
    GAMMA,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    MEMORY_CAPACITY,
    PER_ALPHA,
    PER_BETA_START,
    SEQ_LEN,
    SWITCH_PENALTY,
)
from environment import RFEnvironment
from ML_scheduler import SmartScheduler, SequentialScanner, RandomScanner
from ui_components import (
    inject_tactical_css,
    plot_waterfall_spectrogram,
    plot_3d_spectrum_topology,
    render_emitter_classifier_card,
    render_pipeline_diagram,
    render_emitter_legend,
    plot_band_activity_histogram,
    plot_polar_band_activity,
    plot_power_trend,
    render_console_log,
    render_mission_summary,
)

# 1. APPLICATION SETUP
st.set_page_config(
    page_title="SmartScan — Interactive Tactical Command",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM COMPONENTS & ANIMATED MARQUEE TICKER
def render_status_ticker(messages):
    ticker_text = " &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; ".join(messages)
    
    st.markdown(f"""
        <style>
        .ticker-wrapper {{
            width: 100%;
            overflow: hidden;
            background: rgba(13, 17, 23, 0.95);
            border: 1px solid rgba(56, 139, 253, 0.3);
            border-radius: 6px;
            padding: 8px 0;
            margin-bottom: 16px;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
            white-space: nowrap;
        }}
        
        .ticker-content {{
            display: inline-block;
            white-space: nowrap;
            padding-left: 100%;
            animation: marquee 25s linear infinite;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 600;
            color: #3fb950;
            letter-spacing: 1px;
        }}
        
        .ticker-wrapper:hover .ticker-content {{
            animation-play-state: paused;
        }}

        @keyframes marquee {{
            0%   {{ transform: translate(0, 0); }}
            100% {{ transform: translate(-100%, 0); }}
        }}
        </style>

        <div class="ticker-wrapper">
            <div class="ticker-content">
                {ticker_text} &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; {ticker_text}
            </div>
        </div>
    """, unsafe_allow_html=True)

# 3. CSS FIXES & RESTORED CARD HOVER STYLES
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains-Mono:wght@400;600;800&family=Inter:wght@400;600;700;800&display=swap');

    :root {
        --bg-main: #06090e;
        --card-bg: rgba(13, 17, 23, 0.85);
        --card-border: rgba(56, 139, 253, 0.25);
        --accent-green: #238636;
        --glow-green: #3fb950;
        --accent-blue: #58a6ff;
        --glow-blue: #79c0ff;
        --accent-amber: #d29922;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
    }

    /* Make header background transparent and allow the toggle button to stay visible */
    [data-testid="stHeader"] {
        background: transparent !important;
        z-index: 100000 !important;
    }

    /* Ensure sidebar collapse/expand trigger is clearly visible */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #3fb950 !important;
        background: rgba(13, 17, 23, 0.9) !important;
        border: 1px solid rgba(56, 139, 253, 0.4) !important;
        border-radius: 6px !important;
        z-index: 100001 !important;
    }

    /* Main container padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* SIDEBAR ALIGNMENT FIX: Pull content right to the top-left corner */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.8rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1527 0%, #05070a 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Moving scanline overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(63,185,80,0.55), transparent);
        z-index: 999;
        animation: scanline-sweep 6s linear infinite;
        pointer-events: none;
    }
    @keyframes scanline-sweep {
        0% { top: 0%; opacity: 0; }
        5% { opacity: 1; }
        95% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }

    /* Top Custom Navigation Bar */
    .smartscan-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        width: 100%;
    }
    .smartscan-nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .smartscan-brand-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.8px;
    }
    .smartscan-brand-title span {
        color: #3fb950;
    }
    .smartscan-nav-items {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .nav-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        background: rgba(56, 139, 253, 0.12);
        border: 1px solid rgba(56, 139, 253, 0.3);
        color: #79c0ff;
        padding: 4px 10px;
        border-radius: 6px;
    }

    /* Sidebar Header Box */
    .sidebar-header-box {
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid rgba(56, 139, 253, 0.35);
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 0px;
        margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    .sidebar-header-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 800;
        color: #3fb950;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0;
        padding: 0;
    }

    /* Animated Radar Sweep Icon */
    .radar-sweep-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, rgba(63,185,80,0.9), rgba(63,185,80,0) 70%);
        position: relative;
        animation: radar-spin 2.2s linear infinite;
        border: 1px solid rgba(63,185,80,0.5);
    }
    @keyframes radar-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .radar-pulse {
        width: 10px;
        height: 10px;
        background-color: #3fb950;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7);
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(63, 185, 80, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
    }

    /* Section Headings */
    .section-heading {
        display: flex;
        align-items: baseline;
        gap: 10px;
        margin: 6px 0 10px 0;
        border-bottom: 1px solid rgba(56, 139, 253, 0.18);
        padding-bottom: 6px;
    }
    .section-heading h2 {
        font-size: 1.25rem;
        font-weight: 800;
        margin: 0;
        color: var(--text-primary);
    }
    .section-heading span.tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--glow-blue);
        background: rgba(56,139,253,0.12);
        border: 1px solid rgba(56,139,253,0.3);
        padding: 2px 7px;
        border-radius: 4px;
    }

    /* PIPELINE & LEGEND CARD STYLES + HOVER EFFECTS */
    .pipeline-card, .legend-card {
        background: rgba(13, 17, 23, 0.85);
        border: 1px solid rgba(56, 139, 253, 0.25);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        position: relative;
    }

    .pipeline-card:hover, .legend-card:hover {
        transform: translateY(-3px);
        border-color: rgba(63, 185, 80, 0.6);
        box-shadow: 0 8px 22px rgba(63, 185, 80, 0.15), 0 0 10px rgba(56, 139, 253, 0.2);
        background: rgba(18, 24, 34, 0.95);
    }

    .pipeline-title, .legend-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        color: #f0f6fc;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }

    .pipeline-desc, .legend-desc {
        font-size: 0.78rem;
        color: #8b949e;
        line-height: 1.4;
    }

    /* Hero Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.85) 100%);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-left: 5px solid #3fb950;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 14px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
    }
    .hero-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .hero-badge {
        background: rgba(35, 134, 54, 0.2);
        color: #3fb950;
        border: 1px solid #238636;
        padding: 3px 8px;
        border-radius: 5px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
        height: 135px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(88, 166, 255, 0.5);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        font-weight: 700;
    }
    .metric-value-large {
        font-size: 1.9rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .pwr-bar-container {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        height: 5px;
        margin-top: 8px;
        overflow: hidden;
    }
    .pwr-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #238636 0%, #3fb950 100%);
        border-radius: 4px;
    }

    /* Console Dashboard Footer */
    .console-footer-dashboard {
        background: rgba(13, 17, 23, 0.85);
        border: 1px solid rgba(56, 139, 253, 0.25);
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Step guide */
    .step-card {
        background: rgba(13, 17, 23, 0.75);
        border: 1px solid rgba(56,139,253,0.2);
        border-radius: 10px;
        padding: 14px;
        height: 115px;
        transition: all 0.25s ease;
    }
    .step-card:hover {
        border-color: rgba(63, 185, 80, 0.5);
        transform: translateY(-2px);
    }
    .step-num {
        font-family: 'JetBrains Mono', monospace;
        color: #3fb950;
        font-weight: 800;
        font-size: 0.8rem;
    }
    .step-title { font-weight: 800; font-size: 0.9rem; margin: 2px 0; }
    .step-desc { font-size: 0.76rem; color: var(--text-secondary); }

    .footer-note {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.75rem;
        padding: 20px 0 10px 0;
        border-top: 1px solid rgba(56,139,253,0.15);
        margin-top: 20px;
    }
    </style>

    <script>
    function playInterceptBeep() {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            gain.gain.setValueAtTime(0.05, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.15);
        } catch(e) {}
    }
    </script>
""", unsafe_allow_html=True)

inject_tactical_css()

# Session State Initializations
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'export_df' not in st.session_state:
    st.session_state.export_df = None

# --- TOP NAVIGATION BAR ---
st.markdown("""
    <div class="smartscan-navbar">
        <div class="smartscan-nav-brand">
            <div class="radar-sweep-icon"></div>
            <div class="smartscan-brand-title">Smart<span>Scan</span> Interactive Engine</div>
            <span class="nav-badge">v2.4 D3QN + LSTM</span>
        </div>
        <div class="smartscan-nav-items">
            <span class="nav-badge">DRDO PS 26055</span>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="radar-pulse"></span>
                <span style="font-family:'JetBrains Mono',monospace; font-size: 0.75rem; color: #3fb950; font-weight:700;">SYSTEM ONLINE</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR BRANDING ---
st.sidebar.markdown("""
    <div class="sidebar-header-box">
        <div class="sidebar-header-title">
            📡 SmartScan Dynamic
        </div>
        <div style="font-size:0.72rem; color:#8b949e; margin-top:2px;">
            Tactical EW Control Center
        </div>
    </div>
""", unsafe_allow_html=True)

sidebar_mode = st.sidebar.radio(
    "🕹️ Operation Mode",
    ["Autonomous Intercept", "Manual Spectrum Audit"],
    key="sidebar_mode_select"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Fine-Tuning")

num_bands_input = st.sidebar.number_input(
    "Frequency Bands (N)",
    min_value=1,
    max_value=1000,
    value=DEFAULT_NUM_BANDS,
    step=1,
    help="Enter channel count directly"
)

num_time_slots = st.sidebar.slider("Simulation Slots", min_value=50, max_value=500, value=DEFAULT_TIME_SLOTS, step=25)
enable_audio = st.sidebar.checkbox("🔊 Audio Feedback on Intercept", value=True)
sim_speed = st.sidebar.slider("Processing Delay (s)", min_value=0.0, max_value=0.2, value=0.01, step=0.01)

st.sidebar.caption(f"⚡ Spectral Range: 30.0 - 512.0 MHz | **{(512.0 - 30.0) / max(1, num_bands_input):.2f} MHz / Band**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Hyperparameters")
st.sidebar.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#8b949e;line-height:1.8;">
Learning Rate&nbsp;&nbsp;&nbsp;<span style="color:#3fb950;">{LEARNING_RATE}</span><br>
Gamma (discount)&nbsp;&nbsp;<span style="color:#3fb950;">{GAMMA}</span><br>
Epsilon start→min&nbsp;<span style="color:#3fb950;">{EPSILON_START} → {EPSILON_MIN}</span><br>
Epsilon decay&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#3fb950;">{EPSILON_DECAY}</span><br>
PER memory cap&nbsp;&nbsp;&nbsp;<span style="color:#3fb950;">{MEMORY_CAPACITY}</span><br>
PER α / β₀&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#3fb950;">{PER_ALPHA} / {PER_BETA_START}</span><br>
Sequence length&nbsp;&nbsp;<span style="color:#3fb950;">{SEQ_LEN}</span><br>
Switch penalty&nbsp;&nbsp;&nbsp;<span style="color:#3fb950;">{SWITCH_PENALTY}</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("SmartScan Command Engine · DRDO PS 26055")

# --- SCROLLING ANIMATED STATUS MARQUEE ---
render_status_ticker([
    "SMARTSCAN ENGINE ONLINE",
    f"D3QN + LSTM · LR {LEARNING_RATE} · γ {GAMMA}",
    "PER REPLAY BUFFER ACTIVE",
    "RAYLEIGH FADING MODEL LOADED",
    "SPECTRAL RANGE 30.0 – 512.0 MHz",
    "AWAITING ENGINE START" if not st.session_state.sim_running else "LIVE INTERCEPT RUN IN PROGRESS",
])

# --- HERO COMMAND DECK ---
st.markdown("""
    <div class="hero-container">
        <div class="hero-title-row">
            <div>
                <span class="hero-badge">DRDO PS 26055</span>
                <h1 style="margin: 6px 0 2px 0; font-size: 1.8rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">
                    SmartScan Interactive Command Engine
                </h1>
                <p style="margin: 0; color: #8b949e; font-size: 0.88rem;">
                    Real-time Spectrum Interception • Deep D3QN + LSTM Hopping Predictor
                </p>
            </div>
            <div style="text-align: right;">
                <div style="display: flex; align-items: center; gap: 8px; background: rgba(13, 17, 23, 0.8); border: 1px solid rgba(63, 185, 80, 0.4); padding: 6px 14px; border-radius: 20px;">
                    <span class="radar-pulse"></span>
                    <span style="color: #3fb950; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.5px;">LIVE SYSTEM READY</span>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Control Bar
c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
with c1:
    preset_mode = st.selectbox(
        "⚡ Environment Preset",
        ["Standard (20 Bands)", "High Density (80 Bands)", "Tactical Noise (Custom / 100 Bands)"],
        key="hero_preset_select"
    )
with c2:
    jammer_strategy = st.selectbox(
        "🛡️ Jammer Strategy",
        ["SWEEP", "BARRAGE", "REACTIVE_SPOOF"],
        key="hero_jammer_select"
    )
with c3:
    enable_fading = st.toggle("Rayleigh Fading", value=True, key="hero_fading_toggle")

with c4:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Start Engine", type="primary", use_container_width=True, key="hero_run"):
            st.session_state.sim_running = True
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=True, key="hero_reset"):
            st.session_state.sim_running = False
            st.session_state.export_df = None

# Adjust band count
if "Standard" in preset_mode:
    num_bands = 20
elif "High Density" in preset_mode:
    num_bands = 80
else:
    num_bands = max(100, int(num_bands_input))

st.markdown("<br>", unsafe_allow_html=True)

# --- MISSION SETUP ---
st.markdown('<div class="section-heading"><h2>Mission Setup</h2><span class="tag">3 STEPS</span></div>', unsafe_allow_html=True)
g1, g2, g3 = st.columns(3)
guide = [
    ("01", "Configure Environment", "Select band-density preset, jammer strategy, and Rayleigh fading above."),
    ("02", "Start Engine", "SmartScan trains its D3QN + LSTM predictor on the spectrum and starts live tuning."),
    ("03", "Analyze Intercepts", "Watch spectrogram, benchmark metrics, and mission console update in real time."),
]
for col, (num, title, desc) in zip([g1, g2, g3], guide):
    with col:
        st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- LIVE METRIC CARDS ---
st.markdown('<div class="section-heading"><h2>Live Receiver Telemetry</h2><span class="tag">REAL-TIME</span></div>', unsafe_allow_html=True)
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1: m_hits = st.empty()
with m_col2: m_pd = st.empty()
with m_col3: m_pwr = st.empty()
with m_col4: m_status = st.empty()

def render_metric_card(container, label, value, subtext="", color="#3fb950", pwr_pct=None):
    if pwr_pct is not None:
        pwr_html = f'<div class="pwr-bar-container"><div class="pwr-bar-fill" style="width: {pwr_pct}%;"></div></div>'
    else:
        pwr_html = '<div class="pwr-bar-container" style="background: transparent; visibility: hidden;"></div>'
    
    card_html = f'<div class="metric-card"><div><div class="metric-label">{label}</div><div class="metric-value-large" style="color: {color};">{value}</div><div style="font-size:0.75rem; color:#8b949e; font-weight:600;">{subtext}</div></div>{pwr_html}</div>'
    container.markdown(card_html, unsafe_allow_html=True)

render_metric_card(m_hits, "SmartScan Intercepts", "0", "Baseline: 0")
render_metric_card(m_pd, "Probability of Intercept (Pd)", "0.0%", "+0.0% vs Sweep")
render_metric_card(m_pwr, "Receiver Power", "-100.0 dBm", "Noise Floor Level", color="#58a6ff", pwr_pct=5)
render_metric_card(m_status, "Receiver State", "IDLE", "Standby Mode", color="#8b949e")

st.markdown("<br>", unsafe_allow_html=True)

# --- SYSTEM PIPELINE ---
st.markdown('<div class="section-heading"><h2>System Architecture</h2><span class="tag">PIPELINE</span></div>', unsafe_allow_html=True)
render_pipeline_diagram()

st.markdown("<br>", unsafe_allow_html=True)

# --- THREAT LEGEND ---
st.markdown('<div class="section-heading"><h2>Emitter &amp; Threat Legend</h2><span class="tag">REFERENCE</span></div>', unsafe_allow_html=True)
render_emitter_legend()

st.markdown("<br>", unsafe_allow_html=True)

# --- WORKSPACE TABS ---
st.markdown('<div class="section-heading"><h2>Command Workspace</h2><span class="tag">INTERACTIVE</span></div>', unsafe_allow_html=True)

tab_waterfall, tab_benchmarks, tab_3d, tab_distribution, tab_power = st.tabs([
    "📡 Live Spectrogram & Trajectory Arrows",
    "📊 Dynamic Benchmark Analytics",
    "🌐 3D Spectrum Surface",
    "📶 Band Activity Distribution",
    "📈 Power Analytics",
])

with tab_waterfall:
    band_range = st.slider(
        "🔍 Interactive Band Zoom Filter",
        min_value=0,
        max_value=num_bands - 1,
        value=(0, num_bands - 1),
        help="Drag sliders to isolate specific frequency channels in real-time"
    )
    plot_area_1 = st.empty()

with tab_benchmarks:
    plot_area_2 = st.empty()

with tab_3d:
    plot_area_3 = st.empty()

with tab_distribution:
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        plot_area_4 = st.empty()
    with dist_col2:
        plot_area_4b = st.empty()

with tab_power:
    plot_area_5 = st.empty()

st.markdown("<br>", unsafe_allow_html=True)

# --- LIVE MISSION CONSOLE ---
st.markdown('<div class="section-heading"><h2>Live Mission Console</h2><span class="tag">LOG</span></div>', unsafe_allow_html=True)
console_area = st.empty()
if not st.session_state.sim_running:
    render_console_log(["[STANDBY] Awaiting engine start command..."])

bottom_status_area = st.empty()
if not st.session_state.sim_running:
    bottom_status_area.markdown("""
        <div class="console-footer-dashboard">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span class="radar-pulse"></span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #8b949e;">
                    SYSTEM STATUS: <strong>STANDBY</strong> — Press <strong>▶ Start Engine</strong> to launch live interception.
                </span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #58a6ff;">
                Channels Configured: <strong>{}</strong> | Engine: <strong>D3QN + LSTM</strong>
            </div>
        </div>
    """.format(num_bands), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- SIMULATION LOOP ---
if st.session_state.sim_running:
    env = RFEnvironment(
        num_bands=num_bands,
        num_time_slots=num_time_slots,
        noise_floor_dbm=DEFAULT_NOISE_FLOOR_DBM,
        detection_threshold_dbm=DEFAULT_DETECTION_THRESHOLD_DBM,
        enable_fading=enable_fading,
        jammer_strategy=jammer_strategy,
        seed=DEFAULT_SEED
    )

    d3qn_scheduler = SmartScheduler(num_bands=num_bands, seq_len=SEQ_LEN)
    seq_scheduler = SequentialScanner(num_bands=num_bands)
    rand_scheduler = RandomScanner(num_bands=num_bands, seed=DEFAULT_SEED)

    with st.spinner(f"Initializing SmartScan D3QN + LSTM Engine across {num_bands} bands..."):
        d3qn_scheduler.train_initial_model(env)

    d3qn_history, seq_history, rand_history = [], [], []
    d3qn_hits, seq_hits, rand_hits = 0, 0, 0
    telemetry_records = []
    console_lines = ["[INIT] SmartScan D3QN + LSTM model trained on synthesized spectrum."]

    for t in range(num_time_slots):
        past_grid = env.grid[:t, :] if t > 0 else np.zeros((1, num_bands))

        d3qn_band = d3qn_scheduler.select_next_band(t, past_grid)
        seq_band = seq_scheduler.select_next_band(t)
        rand_band = rand_scheduler.select_next_band(t)

        d3qn_hit = int(env.grid[t, d3qn_band])
        seq_hit = int(env.grid[t, seq_band])
        rand_hit = int(env.grid[t, rand_band])

        current_pwr = float(env.get_slot_power(t)[d3qn_band])

        d3qn_hits += d3qn_hit
        seq_hits += seq_hit
        rand_hits += rand_hit

        if d3qn_hit and enable_audio:
            st.components.v1.html("<script>playInterceptBeep();</script>", height=0, width=0)

        d3qn_history.append({'time': t, 'band': d3qn_band, 'result': d3qn_hit, 'power': current_pwr})
        seq_history.append({'time': t, 'band': seq_band, 'result': seq_hit})
        rand_history.append({'time': t, 'band': rand_band, 'result': rand_hit})

        active_slots_so_far = np.count_nonzero(np.sum(env.grid[:t+1, :], axis=1) > 0)
        denominator = max(1, active_slots_so_far)

        pd_d3qn = min(100.0, (d3qn_hits / denominator) * 100.0)
        pd_seq  = min(100.0, (seq_hits / denominator) * 100.0)
        pd_rand = min(100.0, (rand_hits / denominator) * 100.0)

        telemetry_records.append({
            "time_slot": t,
            "smartscan_selected_band": d3qn_band,
            "smartscan_hit": d3qn_hit,
            "smartscan_power_dbm": round(current_pwr, 2),
            "seq_selected_band": seq_band,
            "seq_hit": seq_hit,
            "rand_selected_band": rand_band,
            "rand_hit": rand_hit,
            "cumulative_smartscan_pd": round(pd_d3qn, 2),
            "cumulative_seq_pd": round(pd_seq, 2),
            "cumulative_rand_pd": round(pd_rand, 2)
        })

        delta_hits = d3qn_hits - seq_hits
        delta_pd = pd_d3qn - pd_seq
        pwr_bar_pct = max(0, min(100, int((current_pwr + 100) * 1.66)))

        render_metric_card(m_hits, "SmartScan Intercepts", f"{d3qn_hits}", f"+{delta_hits} vs Sweep Baseline")
        render_metric_card(m_pd, "Probability of Intercept (Pd)", f"{pd_d3qn:.1f}%", f"+{delta_pd:.1f}% vs Sweep Baseline")
        render_metric_card(m_pwr, "Receiver Power", f"{current_pwr:.1f} dBm", "Active Channel RF Level", color="#58a6ff", pwr_pct=pwr_bar_pct)

        state_str = "INTERCEPTING" if d3qn_hit else "SCANNING"
        state_color = "#3fb950" if d3qn_hit else "#d29922"
        render_metric_card(m_status, "Receiver State", state_str, f"Channel {d3qn_band} Locked", color=state_color)

        render_emitter_classifier_card(d3qn_band, current_pwr, d3qn_hit, d3qn_history)

        if d3qn_hit:
            console_lines.append(f"[T{t:04d}] CH {d3qn_band:03d} → <span style='color:#3fb950;'>INTERCEPT LOCKED</span> ({current_pwr:.1f} dBm)")
        else:
            console_lines.append(f"[T{t:04d}] CH {d3qn_band:03d} → scanning ({current_pwr:.1f} dBm)")
        with console_area:
            render_console_log(console_lines)

        bottom_status_area.markdown(f"""
            <div class="console-footer-dashboard">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span class="radar-pulse"></span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #3fb950;">
                        LIVE STREAMING — Slot: <strong>{t + 1}/{num_time_slots}</strong> | Current Target: <strong>CH {d3qn_band}</strong> | Efficiency Gain: <strong>+{delta_pd:.1f}%</strong>
                    </span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #79c0ff;">
                    SmartScan Intercepts: <strong>{d3qn_hits}</strong> | Baseline Sweeps: <strong>{seq_hits}</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

        filtered_power_grid = env.power_grid_dbm[:, band_range[0]:band_range[1]+1]
        fig_waterfall = plot_waterfall_spectrogram(
            filtered_power_grid,
            d3qn_history,
            t,
            band_range[1] - band_range[0] + 1,
            seq_history=seq_history
        )

        plot_area_1.plotly_chart(fig_waterfall, use_container_width=True)

        if t % 5 == 0 or t == num_time_slots - 1:
            fig_comp = go.Figure(data=[
                go.Bar(name='Sequential Sweep', x=['Intercepts', 'Pd (%)'], y=[seq_hits, pd_seq], marker_color='#f85149'),
                go.Bar(name='Random Scan', x=['Intercepts', 'Pd (%)'], y=[rand_hits, pd_rand], marker_color='#d29922'),
                go.Bar(name='SmartScan D3QN+LSTM', x=['Intercepts', 'Pd (%)'], y=[d3qn_hits, pd_d3qn], marker_color='#238636')
            ])
            fig_comp.update_layout(
                barmode='group',
                title=dict(text=f"Tactical Benchmark Metrics (Slot t={t} | Band View: {band_range[0]}-{band_range[1]})", font=dict(size=14, color="#f0f6fc")),
                paper_bgcolor="rgba(18, 24, 38, 0.75)",
                plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", size=11),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            plot_area_2.plotly_chart(fig_comp, use_container_width=True)

        if t % 10 == 0 or t == num_time_slots - 1:
            fig_3d = plot_3d_spectrum_topology(filtered_power_grid, t, band_range[1] - band_range[0] + 1)
            plot_area_3.plotly_chart(fig_3d, use_container_width=True)

        if t % 8 == 0 or t == num_time_slots - 1:
            fig_hist = plot_band_activity_histogram(d3qn_history, num_bands)
            plot_area_4.plotly_chart(fig_hist, use_container_width=True)
            fig_polar = plot_polar_band_activity(d3qn_history, num_bands)
            plot_area_4b.plotly_chart(fig_polar, use_container_width=True)

        if t % 6 == 0 or t == num_time_slots - 1:
            fig_power = plot_power_trend(d3qn_history)
            plot_area_5.plotly_chart(fig_power, use_container_width=True)

        if sim_speed > 0:
            time.sleep(sim_speed)

    st.success("SmartScan Interactive Run Finished.")
    st.session_state.export_df = pd.DataFrame(telemetry_records)
    st.session_state.final_stats = (d3qn_hits, seq_hits, rand_hits, pd_d3qn, pd_seq, pd_rand)

# --- MISSION SUMMARY ---
if st.session_state.export_df is not None and 'final_stats' in st.session_state:
    st.markdown('<div class="section-heading"><h2>Mission Summary</h2><span class="tag">RESULT</span></div>', unsafe_allow_html=True)
    render_mission_summary(*st.session_state.final_stats)
    st.markdown("<br>", unsafe_allow_html=True)

# --- TELEMETRY LOG ---
if st.session_state.export_df is not None:
    st.markdown('<div class="section-heading"><h2>Mission Telemetry</h2><span class="tag">EXPORT</span></div>', unsafe_allow_html=True)
    with st.expander("📋 Interactive Mission Telemetry Log", expanded=True):
        st.dataframe(st.session_state.export_df, use_container_width=True)

    csv_data = st.session_state.export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Telemetry Log (CSV)",
        data=csv_data,
        file_name=f"smartscan_mission_telemetry_N{num_bands}.csv",
        mime="text/csv",
        type="primary"
    )
else:
    if not st.session_state.sim_running:
        st.info("Select an environment preset, configure jammer options in the hero control panel, and click **▶ Start Engine**.")

# --- FOOTER ---
st.markdown("""
    <div class="footer-note">
        SmartScan Interactive Command Engine · DRDO PS 26055 · Deep D3QN + LSTM Spectrum Interception Simulator
    </div>
""", unsafe_allow_html=True)