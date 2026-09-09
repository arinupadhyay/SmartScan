# 📡 SmartScan Interactive Command Engine

An autonomous RF spectrum intercept platform powered by Deep Reinforcement Learning (D3QN + LSTM) that detects, classifies, and tracks agile frequency-hopping targets (FHSS), radar chirps, and burst communications in complex electronic warfare environments.

## 🚀 Features

- **Autonomous Signal Interception:** D3QN model with LSTM temporal memory layer achieving a **97.5% Probability of Intercept ($P_d$)** (195 Intercepts vs. 11 Sweep / 13 Random).
- **Tactical Visualizations:** Dynamic 2D Waterfall Spectrogram with trajectory arrows, 3D Power Spectral Density Surface maps, Band Activity Distributions, and Polar Intercept Density charts.
- **Dynamic Benchmarking:** Live, side-by-side performance analytics comparing SmartScan against traditional Sequential Sweep and Random Scan baselines.
- **Electronic Countermeasures & Fading:** Supports adversarial jammer strategies (SWEEP, BARRAGE, REACTIVE_SPOOF) and realistic channel propagation via Rayleigh Fading models.
- **Live Receiver Telemetry & Logs:** Real-time console audit logging with active channel locks, dBm power levels, and efficiency gain tracking (+92.0%).

## 🛠 Tech Stack

- Python 3.9+
- PyTorch (D3QN + LSTM Architecture)
- Streamlit
- Plotly (3D Surface Maps & Waterfall Plots)
- NumPy / SciPy

## 📸 Screenshots

### Command Engine & Mission Setup
![SmartScan Interactive Command Engine Interface](docs/images/command_engine.png)

### Live Receiver Telemetry & Emitter Reference
![Live Telemetry Dashboard](docs/images/telemetry_dashboard.png)

### Spectrum Waterfall & Receiver Track
![Spectrum Waterfall Spectrogram](docs/images/waterfall_spectrogram.png)

### 3D Power Spectral Density Surface
![3D Spectrum Surface](docs/images/3d_spectrum_surface.png)

### Dynamic Benchmark Analytics
![Dynamic Benchmark Comparison](docs/images/benchmark_analytics.png)

### Live Mission Console
![Live Mission Console Logs](docs/images/mission_console.png)

## ⚙️ Installation

Clone the repository:

git clone https://github.com/arinupadhyay/SmartScan.git

Install dependencies:

pip install -r requirements.txt

Run the project:

streamlit run app.py

## 🔗 Live Demo

https://smartscan-rf.streamlit.app

## 👨‍💻 Author

Arin Upadhyay
