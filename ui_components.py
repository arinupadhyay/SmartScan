# ui_components.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np

def inject_tactical_css():
    """Injects a clean, professional engineering dark theme."""
    st.markdown("""
        <style>
        /* Base Layout & Clean Dark Background */
        .stApp {
            background-color: #0e1117;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Clean Header Bar */
        .main-header {
            padding: 1rem 0rem 1.5rem 0rem;
            border-bottom: 1px solid #1e293b;
            margin-bottom: 1.5rem;
        }
        .main-header h1 {
            color: #f8fafc;
            font-size: 1.6rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .main-header p {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #161b22;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 12px 16px;
        }
        div[data-testid="stMetricLabel"] {
            color: #8b949e !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stMetricValue"] {
            color: #f0f6fc !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #21262d;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            color: #8b949e;
            font-size: 0.88rem;
            font-weight: 500;
            padding: 0px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #161b22 !important;
            color: #58a6ff !important;
            border-bottom: 2px solid #58a6ff !important;
        }

        /* Clean Card Boxes */
        .info-card {
            background-color: #161b22;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 1rem;
        }
        .info-card-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .info-card-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #f0f6fc;
        }

        /* Buttons */
        .stButton>button {
            width: 100%;
            background-color: #238636;
            color: #ffffff;
            border: 1px solid rgba(240,246,252,0.1);
            border-radius: 6px;
            font-weight: 500;
            padding: 6px 16px;
            transition: background-color 0.15s ease;
        }
        .stButton>button:hover {
            background-color: #2ea043;
            border-color: rgba(240,246,252,0.2);
            color: #ffffff;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0d1117;
            border-right: 1px solid #21262d;
        }
        </style>
    """, unsafe_allow_html=True)


def plot_waterfall_spectrogram(power_grid_dbm, scan_history, current_time, num_bands, seq_history=None):
    """Renders a clean 2D Spectrogram with receiver scan path overlay(s)."""
    fig = go.Figure()

    # Power Spectral Density Heatmap
    fig.add_trace(go.Heatmap(
        z=power_grid_dbm[:current_time+1, :].T,
        x=list(range(current_time + 1)),
        y=[f"B{i+1}" for i in range(num_bands)],
        colorscale="Viridis",
        colorbar=dict(title="PSD (dBm)", thickness=12, len=0.9),
        showscale=True
    ))

    # Overlay Primary (SmartScan) Receiver Path
    if scan_history:
        times = [record['time'] for record in scan_history]
        bands = [f"B{record['band']+1}" for record in scan_history]
        results = [record['result'] for record in scan_history]

        fig.add_trace(go.Scatter(
            x=times,
            y=bands,
            mode='lines+markers',
            name='SmartScan Tune',
            line=dict(color='#f0f6fc', width=1.5),
            marker=dict(
                size=6,
                color=['#3fb950' if r == 1 else '#f85149' for r in results],
                symbol=['circle' if r == 1 else 'x' for r in results]
            )
        ))

    # Optional Overlay: Baseline (e.g. Sequential Sweep) Receiver Path
    if seq_history:
        times = [record['time'] for record in seq_history]
        bands = [f"B{record['band']+1}" for record in seq_history]

        fig.add_trace(go.Scatter(
            x=times,
            y=bands,
            mode='lines',
            name='Baseline Sweep',
            line=dict(color='#d29922', width=1, dash='dot'),
            opacity=0.6
        ))

    fig.update_layout(
        title=dict(text="Spectrum Waterfall & Receiver Track", font=dict(size=14, color="#f0f6fc")),
        xaxis=dict(title="Time Slot (t)", gridcolor="#21262d", zeroline=False),
        yaxis=dict(title="Frequency Band", gridcolor="#21262d", zeroline=False),
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        height=380,
        showlegend=bool(seq_history),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_3d_spectrum_topology(power_grid_dbm, current_time, num_bands):
    """Renders a 3D Surface Plot of the RF spectrum."""
    time_window = min(current_time + 1, 50)
    start_t = max(0, current_time + 1 - time_window)
    
    z_data = power_grid_dbm[start_t:current_time+1, :]
    x_data = [f"B{i+1}" for i in range(num_bands)]
    y_data = list(range(start_t, current_time + 1))

    fig = go.Figure(data=[go.Surface(
        z=z_data,
        x=x_data,
        y=y_data,
        colorscale="Viridis",
        showscale=False
    )])

    fig.update_layout(
        title=dict(text="3D Power Spectral Density Surface", font=dict(size=14, color="#f0f6fc")),
        scene=dict(
            xaxis=dict(title="Band", backgroundcolor="#0d1117", gridcolor="#21262d"),
            yaxis=dict(title="Time", backgroundcolor="#0d1117", gridcolor="#21262d"),
            zaxis=dict(title="Power (dBm)", backgroundcolor="#0d1117", gridcolor="#21262d"),
        ),
        paper_bgcolor="#161b22",
        font=dict(color="#8b949e", size=10),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400
    )
    return fig


def render_emitter_classifier_card(current_band, current_power, is_hit, history):
    """Renders a simple, clean signal classification status block."""
    if not is_hit:
        classification = "No Signal Detected"
        status_color = "#8b949e"
    else:
        if current_power > -60.0:
            classification = "High-Power Pulse / Radar"
            status_color = "#f85149"
        elif current_power > -75.0:
            classification = "Tactical Comms / FHSS"
            status_color = "#58a6ff"
        else:
            classification = "Low-Power / Noise Floor"
            status_color = "#d29922"

    st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">Active Tuned Channel Target</div>
            <div class="info-card-value" style="color: {status_color};">
                Band {current_band + 1} &mdash; {classification}
            </div>
            <div style="font-size: 0.8rem; color: #8b949e; margin-top: 4px;">
                Signal Strength: {current_power:.1f} dBm
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# ADDITIONAL UI COMPONENTS (previously missing — caused ImportError)
# ==========================================

def render_pipeline_diagram():
    """Renders a simple card-based view of the system processing pipeline."""
    stages = [
        ("📡", "RF Environment", "Synthesizes radar, FHSS, and burst emitters plus jammer + fading + AWGN."),
        ("🧠", "D3QN + LSTM Predictor", "Learns hop patterns from recent spectrum history to predict the next active band."),
        ("🎯", "Scheduler / Receiver", "Tunes the virtual receiver to the predicted band each time slot."),
        ("📊", "Telemetry & Analytics", "Logs hits/misses, Pd, and power to drive live dashboards and CSV export."),
    ]
    cols = st.columns(len(stages))
    for col, (icon, title, desc) in zip(cols, stages):
        with col:
            st.markdown(f"""
                <div class="pipeline-card">
                    <div class="pipeline-title">{icon} {title}</div>
                    <div class="pipeline-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)


def render_emitter_legend():
    """Renders a legend describing each emitter / threat type and its visual cue."""
    legend_items = [
        ("Radar_Chirp", "#f85149", "High-power periodic pulse train (search/track radar)."),
        ("FHSS", "#58a6ff", "Fast pseudo-random frequency-hopping tactical comms."),
        ("Burst_Comms", "#d29922", "Low duty-cycle, short unpredictable comm bursts."),
        ("Jammer (ECM)", "#a371f7", "Adversarial interference: SWEEP, BARRAGE, or REACTIVE_SPOOF."),
    ]
    cols = st.columns(len(legend_items))
    for col, (name, color, desc) in zip(cols, legend_items):
        with col:
            st.markdown(f"""
                <div class="legend-card">
                    <div class="legend-title">
                        <span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>
                        {name}
                    </div>
                    <div class="legend-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)


def plot_band_activity_histogram(history, num_bands):
    """Bar chart of how many times each band was tuned vs. hit."""
    tuned_counts = np.zeros(num_bands, dtype=int)
    hit_counts = np.zeros(num_bands, dtype=int)

    for record in history:
        b = record['band']
        if 0 <= b < num_bands:
            tuned_counts[b] += 1
            if record.get('result', 0):
                hit_counts[b] += 1

    band_labels = [f"B{i+1}" for i in range(num_bands)]

    fig = go.Figure(data=[
        go.Bar(name='Tuned', x=band_labels, y=tuned_counts, marker_color='#58a6ff'),
        go.Bar(name='Intercepted', x=band_labels, y=hit_counts, marker_color='#3fb950'),
    ])
    fig.update_layout(
        barmode='overlay',
        title=dict(text="Band Tuning / Intercept Distribution", font=dict(size=14, color="#f0f6fc")),
        xaxis=dict(title="Band", gridcolor="#21262d"),
        yaxis=dict(title="Count", gridcolor="#21262d"),
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_polar_band_activity(history, num_bands):
    """Polar bar chart showing intercept density spread across bands."""
    hit_counts = np.zeros(num_bands, dtype=int)
    for record in history:
        b = record['band']
        if 0 <= b < num_bands and record.get('result', 0):
            hit_counts[b] += 1

    band_labels = [f"B{i+1}" for i in range(num_bands)]

    fig = go.Figure(data=[
        go.Barpolar(
            r=hit_counts,
            theta=band_labels,
            marker_color=hit_counts,
            marker_colorscale="Viridis",
            opacity=0.85
        )
    ])
    fig.update_layout(
        title=dict(text="Polar Intercept Density", font=dict(size=14, color="#f0f6fc")),
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(color="#8b949e", gridcolor="#21262d"),
            angularaxis=dict(color="#8b949e", gridcolor="#21262d")
        ),
        paper_bgcolor="#161b22",
        font=dict(color="#8b949e", size=10),
        height=360,
        showlegend=False
    )
    return fig


def plot_power_trend(history):
    """Line chart of received signal power over time for the primary receiver."""
    times = [r['time'] for r in history]
    powers = [r.get('power', np.nan) for r in history]
    results = [r.get('result', 0) for r in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=powers,
        mode='lines',
        name='Receiver Power',
        line=dict(color='#58a6ff', width=1.5)
    ))
    hit_times = [t for t, r in zip(times, results) if r]
    hit_powers = [p for p, r in zip(powers, results) if r]
    if hit_times:
        fig.add_trace(go.Scatter(
            x=hit_times,
            y=hit_powers,
            mode='markers',
            name='Intercept',
            marker=dict(color='#3fb950', size=7, symbol='circle')
        ))

    fig.update_layout(
        title=dict(text="Receiver Power Trend", font=dict(size=14, color="#f0f6fc")),
        xaxis=dict(title="Time Slot (t)", gridcolor="#21262d"),
        yaxis=dict(title="Power (dBm)", gridcolor="#21262d"),
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def render_console_log(lines, max_lines=200):
    """Renders a scrollable, terminal-style console log."""
    display_lines = lines[-max_lines:]
    log_html = "<br>".join(display_lines)
    st.markdown(f"""
        <div style="
            background-color: #0d1117;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 12px 14px;
            height: 220px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: #8b949e;
            line-height: 1.6;
        ">
            {log_html}
        </div>
    """, unsafe_allow_html=True)


def render_mission_summary(d3qn_hits, seq_hits, rand_hits, pd_d3qn, pd_seq, pd_rand):
    """Renders end-of-run summary cards comparing all three schedulers."""
    col1, col2, col3 = st.columns(3)

    rows = [
        (col1, "SmartScan D3QN+LSTM", d3qn_hits, pd_d3qn, "#3fb950"),
        (col2, "Sequential Sweep", seq_hits, pd_seq, "#f85149"),
        (col3, "Random Scan", rand_hits, pd_rand, "#d29922"),
    ]

    for col, label, hits, pd_val, color in rows:
        with col:
            st.markdown(f"""
                <div class="info-card">
                    <div class="info-card-title">{label}</div>
                    <div class="info-card-value" style="color: {color};">
                        {hits} Intercepts
                    </div>
                    <div style="font-size: 0.8rem; color: #8b949e; margin-top: 4px;">
                        Pd: {pd_val:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)