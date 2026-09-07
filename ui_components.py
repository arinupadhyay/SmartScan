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


def plot_waterfall_spectrogram(power_grid_dbm, scan_history, current_time, num_bands):
    """Renders a clean 2D Spectrogram with receiver scan path overlay."""
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

    # Overlay Receiver Path
    if scan_history:
        times = [record['time'] for record in scan_history]
        bands = [f"B{record['band']+1}" for record in scan_history]
        results = [record['result'] for record in scan_history]

        fig.add_trace(go.Scatter(
            x=times,
            y=bands,
            mode='lines+markers',
            name='Receiver Tune',
            line=dict(color='#f0f6fc', width=1.5),
            marker=dict(
                size=6,
                color=['#3fb950' if r == 1 else '#f85149' for r in results],
                symbol=['circle' if r == 1 else 'x' for r in results]
            )
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
        showlegend=False
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