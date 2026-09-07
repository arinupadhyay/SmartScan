# app.py
import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Modular file imports
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

# Page Configuration & CSS Injection
st.set_page_config(page_title="SmartScan: EW Live Tactical Interface", layout="wide")
inject_tactical_css()

st.title("🛡️ SmartScan: Autonomous EW Frequency Intercept Suite")
st.markdown("<b>DRDO SIH Submission Build</b> | D3QN + LSTM Dynamic Receiver Scheduling vs Baseline Scanners", unsafe_allow_html=True)

# --- 1. INPUT LAYER: SIDEBAR USER CONTROLS ---
st.sidebar.header("⚙️ Simulation & Input Parameters")

num_bands = st.sidebar.slider("Number of Frequency Bands", min_value=5, max_value=20, value=DEFAULT_NUM_BANDS)
num_time_slots = st.sidebar.slider("Time Slots (t)", min_value=50, max_value=500, value=DEFAULT_TIME_SLOTS, step=25)
jammer_strategy = st.sidebar.selectbox("Adversary ECM Strategy", ["SWEEP", "BARRAGE", "REACTIVE_SPOOF"])
enable_fading = st.sidebar.checkbox("Enable Rayleigh Fast Fading", value=True)
sim_speed = st.sidebar.slider("Simulation Delay (s)", min_value=0.0, max_value=0.2, value=0.01, step=0.01)

# Initialize Session State Variables
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'export_df' not in st.session_state:
    st.session_state.export_df = None

start_col, reset_col = st.sidebar.columns(2)
if start_col.button("🚀 Run Mission & Benchmark"):
    st.session_state.sim_running = True
if reset_col.button("🔄 Reset"):
    st.session_state.sim_running = False
    st.session_state.export_df = None

# --- 3. DISPLAY LAYER: TOP HUD METRIC CARDS ---
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    m_hits = st.empty()
with metric_col2:
    m_pd = st.empty()
with metric_col3:
    m_pwr = st.empty()
with metric_col4:
    m_status = st.empty()

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["🛰️ Real-Time Waterfall", "📊 Benchmark Comparison", "🌐 3D Spectral Topology"])
plot_area_1 = tab1.empty()
plot_area_2 = tab2.empty()
plot_area_3 = tab3.empty()

# --- SIMULATION EXECUTION LOOP ---
if st.session_state.sim_running:
    # 1. Input Processing & Environment Initialization
    env = RFEnvironment(
        num_bands=num_bands,
        num_time_slots=num_time_slots,
        noise_floor_dbm=DEFAULT_NOISE_FLOOR_DBM,
        detection_threshold_dbm=DEFAULT_DETECTION_THRESHOLD_DBM,
        enable_fading=enable_fading,
        jammer_strategy=jammer_strategy,
        seed=DEFAULT_SEED
    )

    # 2. Schedulers Initialization
    d3qn_scheduler = SmartScheduler(num_bands=num_bands)
    seq_scheduler = SequentialScanner(num_bands=num_bands)
    rand_scheduler = RandomScanner(num_bands=num_bands, seed=DEFAULT_SEED)

    with st.spinner("Initializing D3QN + LSTM Replay Memory & Training Warmup..."):
        d3qn_scheduler.train_initial_model(env, warmup_steps=40)

    # 2. STORAGE LAYER: IN-MEMORY BENCHMARK BUFFERS
    d3qn_history = []
    seq_history = []
    rand_history = []

    d3qn_hits, seq_hits, rand_hits = 0, 0, 0
    telemetry_records = []

    for t in range(num_time_slots):
        # Action Selection
        d3qn_band = d3qn_scheduler.select_next_band(t, d3qn_history)
        seq_band = seq_scheduler.select_next_band(t)
        rand_band = rand_scheduler.select_next_band(t)

        # Environment Query
        d3qn_hit = int(env.grid[t, d3qn_band])
        seq_hit = int(env.grid[t, seq_band])
        rand_hit = int(env.grid[t, rand_band])

        current_pwr = float(env.get_slot_power(t)[d3qn_band])

        d3qn_hits += d3qn_hit
        seq_hits += seq_hit
        rand_hits += rand_hit

        # History Buffers
        d3qn_history.append({'time': t, 'band': d3qn_band, 'result': d3qn_hit, 'power': current_pwr})
        seq_history.append({'time': t, 'band': seq_band, 'result': seq_hit})
        rand_history.append({'time': t, 'band': rand_band, 'result': rand_hit})

        # Calculate Running Probability of Intercept
        total_active_opportunities = np.sum(env.grid[:t+1, :])
        pd_d3qn = (d3qn_hits / max(1, total_active_opportunities)) * 100
        pd_seq = (seq_hits / max(1, total_active_opportunities)) * 100
        pd_rand = (rand_hits / max(1, total_active_opportunities)) * 100

        # Exportable Telemetry Record (Storage)
        telemetry_records.append({
            "time_slot": t,
            "d3qn_selected_band": d3qn_band,
            "d3qn_hit": d3qn_hit,
            "d3qn_power_dbm": round(current_pwr, 2),
            "seq_selected_band": seq_band,
            "seq_hit": seq_hit,
            "rand_selected_band": rand_band,
            "rand_hit": rand_hit,
            "cumulative_d3qn_pd": round(pd_d3qn, 2),
            "cumulative_seq_pd": round(pd_seq, 2),
            "cumulative_rand_pd": round(pd_rand, 2)
        })

        # 3. DISPLAY LAYER UPDATES
        m_hits.metric("D3QN Hits", f"{d3qn_hits}", delta=f"+{d3qn_hits - seq_hits} vs Seq")
        m_pd.metric("D3QN $P_d$", f"{pd_d3qn:.1f}%", delta=f"{pd_d3qn - pd_seq:.1f}% vs Seq")
        m_pwr.metric("Rx PSD Power", f"{current_pwr:.1f} dBm")
        m_status.metric("AI System State", "TRACKING" if d3qn_hit else "SCANNING")

        render_emitter_classifier_card(d3qn_band, current_pwr, d3qn_hit, d3qn_history)

        # Tab 1: Real-Time Spectrogram
        fig_waterfall = plot_waterfall_spectrogram(env.power_grid_dbm, d3qn_history, t, num_bands)
        plot_area_1.plotly_chart(fig_waterfall, use_container_width=True, key=f"waterfall_{t}")

        # Tab 2: Strategy Comparison
        if t % 5 == 0 or t == num_time_slots - 1:
            fig_comp = go.Figure(data=[
                go.Bar(name='Sequential Sweep', x=['Hits', 'Probability of Intercept (%)'], y=[seq_hits, pd_seq], marker_color='#FF3366'),
                go.Bar(name='Random Scan', x=['Hits', 'Probability of Intercept (%)'], y=[rand_hits, pd_rand], marker_color='#FFCC00'),
                go.Bar(name='Smart D3QN + LSTM', x=['Hits', 'Probability of Intercept (%)'], y=[d3qn_hits, pd_d3qn], marker_color='#00FFCC')
            ])
            fig_comp.update_layout(
                barmode='group',
                title=f"📊 Live Strategy Performance Comparison (Slot t={t})",
                template="plotly_dark",
                height=400
            )
            plot_area_2.plotly_chart(fig_comp, use_container_width=True, key=f"comp_{t}")

        # Tab 3: 3D Spectrum Topology
        if t % 10 == 0 or t == num_time_slots - 1:
            fig_3d = plot_3d_spectrum_topology(env.power_grid_dbm, t, num_bands)
            plot_area_3.plotly_chart(fig_3d, use_container_width=True, key=f"topology_{t}")

        if sim_speed > 0:
            time.sleep(sim_speed)

    st.success("Mission Run Complete. Telemetry dataset compiled.")

    # Save to session state for persistence & CSV export
    st.session_state.export_df = pd.DataFrame(telemetry_records)

# --- 4. PERSISTENT DISPLAY & DATA EXPORT SECTION ---
if st.session_state.export_df is not None:
    st.subheader("📋 Executive Mission Summary & Telemetry Export")
    
    df_results = pd.DataFrame({
        "Scanning Strategy": ["Sequential Sweep", "Random Scan", "Smart D3QN + LSTM (Ours)"],
        "Total Intercepted Hits": [
            st.session_state.export_df["seq_hit"].sum(),
            st.session_state.export_df["rand_hit"].sum(),
            st.session_state.export_df["d3qn_hit"].sum()
        ],
        "Probability of Intercept (Pd)": [
            f"{st.session_state.export_df['cumulative_seq_pd'].iloc[-1]:.2f}%",
            f"{st.session_state.export_df['cumulative_rand_pd'].iloc[-1]:.2f}%",
            f"{st.session_state.export_df['cumulative_d3qn_pd'].iloc[-1]:.2f}%"
        ]
    })
    st.table(df_results)

    # File Export (Data Persistence requirement)
    csv_data = st.session_state.export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Export Mission Telemetry CSV",
        data=csv_data,
        file_name="smartscan_mission_telemetry.csv",
        mime="text/csv",
        help="Click to download step-by-step scan history and metric comparison for judging verification."
    )
else:
    if not st.session_state.sim_running:
        st.info("Click **'🚀 Run Mission & Benchmark'** in the sidebar to execute real-time EW comparison.")