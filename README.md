# Vibration TCM Deep Learning Pipeline V2 🚀

> **Real-time Industrial IoT Monitoring with Ensemble AI Models**  
> From 1D CNN proof-of-concept to production-ready 4-model ensemble system

## 📖 Overview

**Vibration TCM V2** is an advanced real-time condition monitoring system that combines **Deep Learning** and **Machine Learning** models for intelligent fault detection in industrial equipment. While demonstrated with vibration sensors, the architecture is **applicable to any IoT time-series monitoring scenario**.

### Key Features

✨ **4-Model Ensemble**: CNN1D, LSTM, CNN2D, + ML (Random Forest)  
⚡ **Real-Time**: Sub-second inference with MQTT streaming  
🎯 **Smart Voting**: Confidence-based tie-breaking for balanced decisions  
📊 **Live Dashboard**: Streamlit interface with real-time metrics  
📱 **Multi-Channel Alerts**: Telegram, database logging, or custom endpoints  
🔧 **Unified Launcher**: Single command replaces 7 separate terminals  

---

## 🎯 V2 Improvements (From Original)

### **V1 → V2 Evolution**

| Feature | V1 (Original) | V2 (Current) |
|---------|--------------|--------------|
| **Models** | 2 models (1D CNN, 2D CNN) | **4 models** (CNN1D, LSTM, CNN2D, ML-RF) |
| **Architecture** | Batch processing | **Real-time streaming** |
| **Decision Logic** | Simple majority | **Confidence-based voting** |
| **Deployment** | Manual analysis | **Production-ready system** |
| **Notifications** | None | **Telegram + Database + Custom** |
| **Monitoring** | Basic training logs | **Live dashboard + alerts** |
| **Launcher** | Manual scripts | **Batch scripts (+scenarios)** |
| **Speed** | Batch mode | **Sub-second real-time** |

### What's New in V2

1. **Ensemble Intelligence**  
   - Added LSTM for temporal patterns  
   - Integrated lightweight ML (RF) for realtime speed  
   - Confidence-based tie-breaking for 2-2 splits

2. **Streaming Architecture**  
   - MQTT pub/sub for loose coupling  
   - Independent model processes (parallel)  
   - Real-time dashboard with live metrics

3. **Production Features**  
   - Database logging (SQLite with 2 tables)  
   - Telegram alerts with cooldown  
   - Data validation tools  
   - Comprehensive error handling

4. **Developer Experience**  
   - Batch scripts for easy launching
   - Multiple deployment scenarios  
   - Batch mode data monitor  
   - Complete documentation

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Vibration TCM V2 System                   │
│                                                             │
│  Hardware Layer          Communication       AI/ML Layer    │
│  ┌──────────┐            ┌──────────┐      ┌────────────┐   │
│  │ IMU/Gyro │──BLE/USB──▶│   MQTT   │◀────▶│ DL Models│   │
│  │ Sensor   │            │  Broker  │      │ (3 models) │   │
│  └──────────┘            └────┬─────┘      └────────────┘   │
│                               │             ┌────────────┐  │
│                               └────────────▶│ ML Model   │ │
│                                             │ (RF-Top20) │  │
│                                             └─────┬──────┘  │
│                                                   │         │
│  Presentation Layer      Data Layer              │          │
│  ┌──────────────┐       ┌──────────────┐         │          │
│  │  Dashboard   │◀──────│   Voting     │◀────────┘         │
│  │  (Streamlit) │       │   Engine     │                    │
│  └──────────────┘       └──────┬───────┘                    │
│                                │                            │
│  Notification Layer     │      │       Storage Layer        │
│  ┌──────────────┐       │      │        ┌──────────────┐    │
│  │  Telegram    │◀──────┘      └──────▶│   SQLite DB  │    │
│  │    Bot       │                       │  (2 tables)  │    │
│  └──────────────┘                       └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Sensor (USB) ─┬─▶ Gateway ─▶ MQTT (raw) ──┬─▶ DL Inference
              │                            │    (3 models)
              │                            │       │
              │                            └─▶ ML Inference  
              │                                   (1 model)
              │                                       │
              │                                       ▼
              │                               ┌──────────────┐
              │                               │ Voting Logic │
              │                               │  (4 models)  │
              │                               └──────┬───────┘
              │                                      │
              │                    ┌─────────────────┴─────────────────┐
              │                    ▼                 ▼                 ▼
              │              Dashboard          Data Logger      Alert Monitor
              │                  │                   │                 │
              └──────────────────┴───────────────────┴─────────────────┘
                                      |
                                      ▼
                               User/Admin
```

### Component Breakdown

| Component | Role | Technology | Performance |
|-----------|------|------------|-------------|
| **Sensor Gateway** | Data acquisition | Python + PySerial | 10 Hz |
| **DL Inference** | Deep learning models | TensorFlow/Keras | ~150ms |
| **ML Inference** | Lightweight classifier | scikit-learn | ~2ms |
| **Voting Engine** | Ensemble decision | Custom logic | <1ms |
| **Dashboard** | Real-time visualization | Streamlit + Plotly | Live updates |
| **Data Logger** | Persistence | SQLite | Async writes |
| **Alert Monitor** | Notifications | Telegram Bot API | Cooldown: 5min |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MQTT Broker (Mosquitto)
- IMU/Gyro sensor or simulation data

### Installation

```bash
# Clone repository
git clone https://github.com/MYoussef11/Vibration-TCM-Deep-Learning-Pipeline
cd Vibration-TCM-Deep-Learning-Pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional: Telegram)
cp .env.example .env
nano .env  # Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
```

### Launch System

**Windows (Recommended)**:
```bash
# Full system with sensor
.\start_full_system.bat

# Demo mode (no sensor needed)
.\start_demo.bat
```

**Manual Launch (All Platforms)**:

```bash
# Terminal 1: Dashboard
streamlit run dashboard.py

# Terminal 2: DL Models
python scripts/stream_inference.py

# Terminal 3: ML Model
python scripts/stream_ml_inference.py

# Terminal 4: Sensor Gateway
python scripts/stream_gateway.py --port COM8

# Terminal 5: Data Logger (optional)
python scripts/data_logger.py
```

---

## 📊 Deployment Scenarios

### Windows Batch Scripts
| Script | Components | Use Case |
|--------|------------|----------|
| `start_full_system.bat` | All 5 | Complete system with monitoring |
| `start_demo.bat` | 4 (no sensor) | Presentations, UI testing |

---

## 🎓 How It Works

### 1. Sensor Data Acquisition

```
IMU Sensor → USB → Gateway Script → MQTT Topic (sensors/vibration/raw)
```

The gateway reads 6-axis data (ax, ay, az, wx, wy, wz) at Hz.

### 2. Inference Pipeline

**Deep Learning Path** (3 models):
- **CNN1D**: 1D convolutions on time series
- **LSTM**: Recurrent layers for temporal dependencies  
- **CNN2D**: 2D convolutions on spectrograms

**Machine Learning Path** (1 model):
- **Random Forest**: Top 20 features, optimized for speed

### 3. Voting & Decision

```python
# Confidence-based tie-breaking
if good_votes == faulty_votes:  # e.g., 2-2 split
    avg_good = mean([conf for model in good_voters])
    avg_faulty = mean([conf for model in faulty_voters])
    final_decision = "Faulty" if avg_faulty > avg_good else "Good"
```

### 4. Output & Alerts

- **Dashboard**: Live metrics, model agreement, charts
- **Database**: Logged predictions for analysis
- **Telegram**: Alerts after N consecutive faults

---

## 🌐 Broader IoT Applicability

While demonstrated with **vibration analysis**, this system is designed for **any IoT time-series monitoring**:

### Adaptable Scenarios

| Domain | Sensor Type | Use Case |
|--------|-------------|----------|
| **Industrial** | Vibration, Current, Temp | Predictive maintenance |
| **Energy** | Power meters | Grid anomaly detection |
| **Agriculture** | Soil moisture, pH | Crop health monitoring |
| **Healthcare** | Wearables (ECG, PPG) | Patient vitals tracking |
| **Smart Buildings** | HVAC, Occupancy | Energy optimization |
| **Transportation** | GPS, Accelerometers | Fleet management |

### Customization Points

1. **Sensor Interface**: Swap `stream_gateway.py` for your protocol (Modbus, BLE, etc.)
2. **Features**: Modify `feature_extractor.py` for domain-specific metrics
3. **Models**: Retrain with your labeled data
4. **Notifications**: Replace Telegram with email, SMS, webhooks, cloud services
5. **Dashboard**: Customize Streamlit UI for your KPIs

**The architecture is protocol-agnostic and extensible!**

---

## 📁 Project Structure

```
Vibration-TCM-Deep-Learning-Pipeline/
├── dashboard.py                   # Streamlit dashboard (4 models)
├── scripts/
│   ├── stream_gateway.py          # Sensor data acquisition
│   ├── stream_inference.py        # DL models inference
│   ├── stream_ml_inference.py     # ML model inference
│   ├── data_logger.py             # Database logging
│   ├── alert_monitor.py           # Telegram alerts
│   ├── data_monitor.py            # Batch mode validation
│   ├── feature_extractor.py       # Top-20 features
│   ├── telegram_notifier.py       # Telegram integration
│   ├── train_top20_model.py       # ML training
│   └── tune_classical_models.py   # ML hyperparameter tuning
├── models/                        # Trained models (DL + ML)
├── data/                          # Raw sensor data
│   └── vibration_logs.db          # SQLite database
├── reports/                       # Training results
│   └── phase3/ml_binary/          # ML model artifacts
├── notebooks/                     # Jupyter analysis
└── requirements.txt               # Python dependencies
```

---

## 🛠️ Configuration

### Environment Variables (`.env`)

```bash
# Telegram (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# MQTT (Default: localhost)
MQTT_BROKER=localhost
MQTT_PORT=1883
```

---

## 📈 Performance Metrics

### Model Accuracy

> **Note**: Accuracy varies by dataset and training configuration. Retrain models with your specific data for best results.

- **Deep Learning Models**: Typical range 85-95% on test data
- **ML Model (RF-20)**: Optimized for speed over accuracy (trade-off for real-time)

### Inference Speed

- **DL Models**: ~100-200ms (3 models combined, varies by hardware)
- **ML Model**: ~2-5ms (feature extraction + inference)
- **Total latency**: Sub-second end-to-end

### System Capacity

- **Throughput**: 5-10 predictions/second (adjustable)
- **Data rate**: 10 Hz raw sensor data
- **Window size**: Configurable (default: 20 samples)

---


## ✨ Acknowledgements

- **V1**: Original 1D vs 2D CNN comparison
- **V2**: Ensemble models + streaming + production features


---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/MYoussef11/Vibration-TCM-Deep-Learning-Pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MYoussef11/Vibration-TCM-Deep-Learning-Pipeline/discussions)

---

**Note**: This is a **Proof of Concept (POC)** demonstrating the architecture. In production:
- Use secure MQTT (TLS)
- Implement authentication
- Add model versioning
- Set up monitoring/logging
- Scale with containers (Docker/Kubernetes)
