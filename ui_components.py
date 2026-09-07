# ui_components.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

def inject_tactical_css():
    """Injects custom CSS to give Streamlit a military command console aesthetic."""
    st.markdown("""
        <style>
        .stApp {
            background-color: #080a0f;
            color: #00ffcc;
            font-family: 'Courier New', Courier, monospace;
        }
        section[data-testid="stSidebar"] {
            background-color: #0d1117;
            border-right: 1px solid #1f293d;
        }
        .tactical-card {
            background-color: #0d131d;
            border: 1px solid #00ffcc;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 10px;
            box-shadow: 0 0 10px rgba(0, 255, 204, 0.15);
        }
        .tactical-title {
            color: #00ffcc;
            font-size: 0.85rem;
            font-weight: bold;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
        .tactical-value {
            color: #ffffff;
            font-size: 1.4rem;
            font-weight: bold;
            margin-top: 4px;
        }
        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: bold;
        }
        .badge-radar { background-color: #ff3366; color: #fff; }
        .badge-fhss { background-color: #00ccff; color: #000; }
        .badge-burst { background-color: #ffcc00; color: #000; }
        .badge-jammer { background-color: #ff0000; color: #fff; }
        .badge-unknown { background-color: #555555; color: #fff; }
        </style>
    """, unsafe_allow_html=True)


def plot_waterfall_spectrogram(env_power_grid, history, current_t, num_bands):
    """Renders a 2D Waterfall Spectrogram with receiver scan overlay."""
    active_power = env_power_grid[:current_t + 1, :num_bands]
    
    times = [entry['time'] for entry in history]
    bands = [entry['band'] for entry in history]
    results = [entry['result'] for entry in history]

    hit_times = [t for t, r in zip(times, results) if r == 1]
    hit_bands = [b for b, r in zip(times, results) if r == 1]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=active_power.T,
        x=list(range(current_t + 1)),
        y=[f"B{b+1}" for b in range(num_bands)],
        colorscale='Viridis',
        colorbar=dict(title="PSD (dBm)", len=0.8),
        showscale=True,
        name='Power Spectrum'
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=[f"B{b+1}" for b in bands],
        mode='lines+markers',
        line=dict(color='#00FFCC', width=2),
        marker=dict(size=4, color='#00FFCC'),
        name='Receiver Track'
    ))

    if hit_times:
        fig.add_trace(go.Scatter(
            x=hit_times,
            y=[f"B{b+1}" for b in hit_bands],
            mode='markers',
            marker=dict(symbol='star', size=12, color='#FFD700', line=dict(color='#FFFFFF', width=1)),
            name='Intercepted Hits'
        ))

    fig.update_layout(
        title="🛰️ Real-Time Waterfall Spectrogram & Trajectory Overlay",
        xaxis_title="Time Slot (t)",
        yaxis_title="Frequency Band",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def plot_3d_spectrum_topology(env_power_grid, current_t, num_bands):
    """Renders a 3D Surface Plot of Power Spectral Density (PSD)."""
    active_power = env_power_grid[:current_t + 1, :num_bands]
    
    x_time = np.arange(current_t + 1)
    y_bands = np.arange(num_bands) + 1

    fig = go.Figure(data=[
        go.Surface(
            z=active_power.T,
            x=x_time,
            y=y_bands,
            colorscale='Thermal',
            colorbar=dict(title="PSD (dBm)")
        )
    ])

    fig.update_layout(
        title="🌐 3D Spectral Power Topology",
        scene=dict(
            xaxis_title="Time Slot (t)",
            yaxis_title="Frequency Band (B)",
            zaxis_title="Power (dBm)",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))
        ),
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig


def render_emitter_classifier_card(current_band, hit_power_dbm, is_hit, history):
    """Classifies intercepted signals in real time based on PSD and temporal history."""
    if not is_hit:
        st.sidebar.markdown("""
            <div class="tactical-card">
                <div class="tactical-title">Emitter Classification</div>
                <div class="tactical-value" style="color: #666;">NO SIGNAL DETECTED</div>
            </div>
        """, unsafe_allow_html=True)
        return

    recent_band_hits = [
        h['result'] for h in history[-10:] if h['band'] == current_band
    ]
    persistence_ratio = sum(recent_band_hits) / max(1, len(recent_band_hits))

    if hit_power_dbm > -45.0 and persistence_ratio > 0.6:
        emitter_type = "Search / Track Radar"
        badge_class = "badge-radar"
        confidence = 94.2
    elif hit_power_dbm > -65.0 and persistence_ratio > 0.8:
        emitter_type = "Active ECM Jammer"
        badge_class = "badge-jammer"
        confidence = 98.7
    elif hit_power_dbm > -75.0 and persistence_ratio < 0.4:
        emitter_type = "FHSS Tactical Radio"
        badge_class = "badge-fhss"
        confidence = 88.5
    elif hit_power_dbm > -85.0:
        emitter_type = "Tactical Burst Comms"
        badge_class = "badge-burst"
        confidence = 82.1
    else:
        emitter_type = "Unknown Transmission"
        badge_class = "badge-unknown"
        confidence = 55.0

    st.sidebar.markdown(f"""
        <div class="tactical-card">
            <div class="tactical-title">🎯 Intercepted Signal Classification</div>
            <div class="tactical-value">
                <span class="status-badge {badge_class}">{emitter_type}</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.8rem; color: #aaa;">
                <b>Target Band:</b> B{current_band + 1}<br>
                <b>Received PSD:</b> {hit_power_dbm:.1f} dBm<br>
                <b>Classifier Confidence:</b> {confidence}%
            </div>
        </div>
    """, unsafe_allow_html=True)