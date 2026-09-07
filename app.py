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
    DEFAULT_DETECTION_THRESHOLD_DBM
)
from environment import RFEnvironment
from ML_scheduler import SmartScheduler, SequentialScanner, RandomScanner
from ui_components import (
    inject_tactical_css,
    plot_waterfall_spectrogram,
    plot_3d_spectrum_topology,
    render_emitter_classifier_card
)

# Application Page Setup
st.set_page_config(
    page_title="SmartScan - Electronic Warfare Suite", 
    page_icon="📡", 
    layout="wide"
)

# Custom Inject for Enhanced Frontend Styling
st.markdown("""
    <style>
    /* Dark Tactical Base Theme Overrides */
    .stApp {
        background-color: #0b0e14;
        color: #c9d1d9;
    }
    
    /* Header Container Styling */
    .main-header {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-left: 5px solid #238636;
        padding: 20px 24px;
        border-radius: 8px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .main-header h1 {
        color: #f0f6fc;
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 0 0 6px 0;
        font-size: 2.1rem;
    }
    .main-header p {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 500;
    }
    .project-badge {
        display: inline-block;
        background-color: #238636;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-right: 10px;
        vertical-align: middle;
    }
    
    /* Metrics Custom Container */
    [data-testid="stMetricValue"] {
        font-family: 'Consolas', 'Courier New', monospace;
        font-weight: 700;
        color: #58a6ff !important;
    }
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar Styling Refinements */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161b22;
        border-radius: 6px 6px 0px 0px;
        border: 1px solid #30363d;
        border-bottom: none;
        color: #8b949e;
        font-weight: 600;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f0f6fc !important;
        border-top: 2px solid #238636 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Inject standard component styles
inject_tactical_css()

# --- HEADER SECTION ---
st.markdown("""
    <div class="main-header">
        <h1><span class="project-badge">DRDO PS 26055</span> SmartScan &mdash; Dynamic Spectrum Intercept Suite</h1>
        <p>Adaptive Reinforcement Learning (D3QN + LSTM) Engine for Dynamic Frequency-Hopping Signal Interception</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### ⚙️ System Configuration")

num_bands = st.sidebar.slider("Frequency Bands (N)", min_value=5, max_value=20, value=DEFAULT_NUM_BANDS)
num_time_slots = st.sidebar.slider("Simulation Time Slots", min_value=50, max_value=500, value=DEFAULT_TIME_SLOTS, step=25)
jammer_strategy = st.sidebar.selectbox("Adversary Jamming Profile", ["SWEEP", "BARRAGE", "REACTIVE_SPOOF"])
enable_fading = st.sidebar.checkbox("Enable Rayleigh Fading Channel", value=True)
sim_speed = st.sidebar.slider("Step Processing Delay (s)", min_value=0.0, max_value=0.2, value=0.01, step=0.01)

st.sidebar.markdown("---")

# Session State Initialization
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'export_df' not in st.session_state:
    st.session_state.export_df = None

col_start, col_reset = st.sidebar.columns(2)
if col_start.button("Execute Run", type="primary", use_container_width=True):
    st.session_state.sim_running = True
if col_reset.button("Reset Session", use_container_width=True):
    st.session_state.sim_running = False
    st.session_state.export_df = None

# --- TOP METRIC ROW ---
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    m_hits = st.empty()
with m_col2:
    m_pd = st.empty()
with m_col3:
    m_pwr = st.empty()
with m_col4:
    m_status = st.empty()

m_hits.metric("SmartScan Total Intercepts", "0")
m_pd.metric("SmartScan Probability (Pd)", "0.0%")
m_pwr.metric("Receiver Power", "-100.0 dBm")
m_status.metric("Receiver State", "IDLE")

st.markdown("<br>", unsafe_allow_html=True)

# Main Interface Tabs
tab_waterfall, tab_benchmarks, tab_3d = st.tabs([
    "📊 Waterfall Spectrogram", 
    "📈 Algorithm Benchmarks", 
    "🌐 3D Spectrum Surface"
])

plot_area_1 = tab_waterfall.empty()
plot_area_2 = tab_benchmarks.empty()
plot_area_3 = tab_3d.empty()

# --- SIMULATION EXECUTION ---
if st.session_state.sim_running:
    # 1. Initialize RF Environment
    env = RFEnvironment(
        num_bands=num_bands,
        num_time_slots=num_time_slots,
        noise_floor_dbm=DEFAULT_NOISE_FLOOR_DBM,
        detection_threshold_dbm=DEFAULT_DETECTION_THRESHOLD_DBM,
        enable_fading=enable_fading,
        jammer_strategy=jammer_strategy,
        seed=DEFAULT_SEED
    )

    # 2. Initialize Schedulers
    d3qn_scheduler = SmartScheduler(num_bands=num_bands)
    seq_scheduler = SequentialScanner(num_bands=num_bands)
    rand_scheduler = RandomScanner(num_bands=num_bands, seed=DEFAULT_SEED)

    # Pre-train sequence predictor on environmental spectrum dynamics
    with st.spinner("Locking SmartScan D3QN + LSTM engine onto frequency-hopping trajectory..."):
        d3qn_scheduler.train_initial_model(env)

    # Tracking Structures
    d3qn_history = []
    seq_history = []
    rand_history = []

    d3qn_hits, seq_hits, rand_hits = 0, 0, 0
    telemetry_records = []

    # --- SIMULATION LOOP ---
    for t in range(num_time_slots):
        # 1. Action Selection
        d3qn_band = d3qn_scheduler.select_next_band(t, env.grid)
        seq_band = seq_scheduler.select_next_band(t)
        rand_band = rand_scheduler.select_next_band(t)

        # 2. Signal Query
        d3qn_hit = int(env.grid[t, d3qn_band])
        seq_hit = int(env.grid[t, seq_band])
        rand_hit = int(env.grid[t, rand_band])

        current_pwr = float(env.get_slot_power(t)[d3qn_band])

        d3qn_hits += d3qn_hit
        seq_hits += seq_hit
        rand_hits += rand_hit

        # 3. History Tracking Updates
        d3qn_history.append({'time': t, 'band': d3qn_band, 'result': d3qn_hit, 'power': current_pwr})
        seq_history.append({'time': t, 'band': seq_band, 'result': seq_hit})
        rand_history.append({'time': t, 'band': rand_band, 'result': rand_hit})

        # 4. Performance Calculations (Normalized & Capped at 100.0%)
        active_slots_so_far = np.count_nonzero(np.sum(env.grid[:t+1, :], axis=1) > 0)
        denominator = max(1, active_slots_so_far)

        pd_d3qn = min(100.0, (d3qn_hits / denominator) * 100.0)
        pd_seq  = min(100.0, (seq_hits / denominator) * 100.0)
        pd_rand = min(100.0, (rand_hits / denominator) * 100.0)

        # 5. Telemetry Data Logging
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

        # 6. UI Metric Card Updates
        delta_hits = d3qn_hits - seq_hits
        delta_pd = pd_d3qn - pd_seq

        m_hits.metric(
            label="SmartScan Total Intercepts", 
            value=f"{d3qn_hits}", 
            delta=f"{'+' if delta_hits >= 0 else ''}{delta_hits} vs Baseline"
        )

        m_pd.metric(
            label="SmartScan Probability (Pd)", 
            value=f"{pd_d3qn:.1f}%", 
            delta=f"{'+' if delta_pd >= 0 else ''}{delta_pd:.1f}% vs Baseline"
        )

        m_pwr.metric("Receiver Power", f"{current_pwr:.1f} dBm")
        m_status.metric("Receiver State", "INTERCEPTING" if d3qn_hit else "SCANNING")

        # Sidebar Classification View
        render_emitter_classifier_card(d3qn_band, current_pwr, d3qn_hit, d3qn_history)

        # Tab 1: Waterfall Plot
        fig_waterfall = plot_waterfall_spectrogram(env.power_grid_dbm, d3qn_history, t, num_bands)
        plot_area_1.plotly_chart(fig_waterfall, use_container_width=True, key=f"waterfall_{t}")

        # Tab 2: Strategy Comparison
        if t % 5 == 0 or t == num_time_slots - 1:
            fig_comp = go.Figure(data=[
                go.Bar(name='Sequential Sweep', x=['Intercepts', 'Probability of Intercept (%)'], y=[seq_hits, pd_seq], marker_color='#f85149'),
                go.Bar(name='Random Scan', x=['Intercepts', 'Probability of Intercept (%)'], y=[rand_hits, pd_rand], marker_color='#d29922'),
                go.Bar(name='SmartScan Agent (Ours)', x=['Intercepts', 'Probability of Intercept (%)'], y=[d3qn_hits, pd_d3qn], marker_color='#238636')
            ])
            fig_comp.update_layout(
                barmode='group',
                title=dict(text=f"Performance Benchmark Comparison Matrix (Time Slot t={t})", font=dict(size=14, color="#f0f6fc")),
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", size=11),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            plot_area_2.plotly_chart(fig_comp, use_container_width=True, key=f"comp_{t}")

        # Tab 3: 3D Surface Topology
        if t % 10 == 0 or t == num_time_slots - 1:
            fig_3d = plot_3d_spectrum_topology(env.power_grid_dbm, t, num_bands)
            plot_area_3.plotly_chart(fig_3d, use_container_width=True, key=f"topology_{t}")

        if sim_speed > 0:
            time.sleep(sim_speed)

    st.success("SmartScan Simulation Execution Completed.")
    st.session_state.export_df = pd.DataFrame(telemetry_records)

# --- DATA EXPORT & BENCHMARK SUMMARY ---
if st.session_state.export_df is not None:
    st.markdown("### Mission Telemetry & Benchmark Summary")
    
    df_summary = pd.DataFrame({
        "Scanning Strategy": ["Sequential Sweep Baseline", "Random Scan Baseline", "SmartScan D3QN + LSTM Agent"],
        "Total Intercepted Transmissions": [
            st.session_state.export_df["seq_hit"].sum(),
            st.session_state.export_df["rand_hit"].sum(),
            st.session_state.export_df["smartscan_hit"].sum()
        ],
        "Probability of Intercept (Pd)": [
            f"{st.session_state.export_df['cumulative_seq_pd'].iloc[-1]:.2f}%",
            f"{st.session_state.export_df['cumulative_rand_pd'].iloc[-1]:.2f}%",
            f"{st.session_state.export_df['cumulative_smartscan_pd'].iloc[-1]:.2f}%"
        ]
    })
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    csv_data = st.session_state.export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download SmartScan Mission Telemetry CSV",
        data=csv_data,
        file_name="smartscan_mission_telemetry.csv",
        mime="text/csv",
        type="primary"
    )
else:
    if not st.session_state.sim_running:
        st.info("Configure tactical parameters in the sidebar and click **Execute Run** to initialize the SmartScan engine.")