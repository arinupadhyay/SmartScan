# 📡 SmartScan: AI-Driven Spectrum Intercept & Signal Intelligence

SmartScan is an autonomous RF spectrum intercept platform powered by Deep Reinforcement Learning (DRL) and Spatio-Temporal feature extraction. It is designed to detect, classify, and track agile frequency-hopping signals, jammers, and target emissions across wideband frequency spectrums in real-time.

Built using **PyTorch**, **Streamlit**, and **Plotly**, SmartScan provides continuous monitoring, automated threat identification, and dynamic spectrum visualization through an intuitive tactical dashboard.

---

## 🎯 Key Features

* **Autonomous Spectrum Intercept:** Leverages a **Dueling Double Deep Q-Network (D3QN)** combined with an **LSTM temporal memory layer** to learn frequency-hopping sequences and maximize detection probability ($P_d$).
* **Real-Time Visualizations:**
  * **2D Waterfall Spectrogram:** Live heatmap showing signal power levels over time across channels.
  * **3D Power Topology Map:** Interactive 3D surface plot mapping temporal frequency dynamics.
  * **Radial Coverage & Emitter Metrics:** Multi-perspective telemetry tracking detection rates ($P_d$) and false alarm probabilities.
* **Algorithmic Benchmark:** Real-time side-by-side performance comparisons between **SmartScan (D3QN+LSTM)**, **Sequential Sweep**, and **Random Scan**.
* **Emitter Classification Engine:** Automated classification card identifying target signals, interferences, and jamming vectors.
* **Tactical Operator UI:** Dark-mode optimized dashboard with low-latency rendering and console audit logs.

---

## 🏗️ System Architecture

```text
               +----------------------------------+
               |    RF Environment Simulation     |
               | (Power Grid Matrix: Time x Band) |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   D3QN + LSTM Neural Network     |
               | (Prioritized Experience Replay)  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    Streamlit Tactical UI         |
               | (Waterfall, 3D Mesh, Telemetry)  |
               +----------------------------------+
