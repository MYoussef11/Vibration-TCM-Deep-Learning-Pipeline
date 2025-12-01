"""
Operational Dashboard for Vibration TCM.
Visualizes real-time sensor data and AI predictions.
Run with: streamlit run dashboard.py
"""

import json
import time
from collections import deque
from datetime import datetime

import pandas as pd
import paho.mqtt.client as mqtt
import plotly.graph_objects as go
import streamlit as st

# Setup Page
st.set_page_config(
    page_title="Vibration TCM Dashboard",
    page_icon="⚡",
    layout="wide",
)

# Constants
MQTT_BROKER = "localhost"
MQTT_TOPIC_RAW = "sensors/vibration/raw"
MQTT_TOPIC_PRED = "sensors/vibration/prediction"
MQTT_TOPIC_ML_PRED = "sensors/vibration/ml_prediction"
HISTORY_LEN = 200

import queue

# Global queue for MQTT messages (thread-safe)
_mqtt_queue = queue.Queue()

# Initialize session state
if "raw_data" not in st.session_state:
    st.session_state.raw_data = deque(maxlen=HISTORY_LEN)
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0
if "fault_count" not in st.session_state:
    st.session_state.fault_count = 0
if "ml_prediction" not in st.session_state:
    st.session_state.ml_prediction = None

def on_message(client, userdata, msg):
    # Push to global queue (thread-safe, always available)
    _mqtt_queue.put(msg)

@st.cache_resource
def setup_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.subscribe([(MQTT_TOPIC_RAW, 0), (MQTT_TOPIC_PRED, 0), (MQTT_TOPIC_ML_PRED, 0)])
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"MQTT Connection Failed: {e}")
        return None

client = setup_mqtt()

# UI Layout
st.title("⚡ Vibration AI Monitor")

# Top Statistics Cards - Create placeholders
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
stat_placeholder1 = stat_col1.empty()
stat_placeholder2 = stat_col2.empty()
stat_placeholder3 = stat_col3.empty()
stat_placeholder4 = stat_col4.empty()

st.divider()

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Real-Time Vibration Signal")
    chart_placeholder = st.empty()

with col2:
    st.subheader("AI Diagnostics")
    metrics_placeholder = st.empty()

st.subheader("Recent Alerts")
alerts_placeholder = st.empty()

# Main Loop
while True:
    # Update Statistics at top
    uptime = datetime.now() - st.session_state.start_time
    hours = int(uptime.total_seconds() // 3600)
    minutes = int((uptime.total_seconds() % 3600) // 60)
    stat_placeholder1.metric("⏱️ Uptime", f"{hours}h {minutes}m")
    
    stat_placeholder2.metric("📊 Total Predictions", st.session_state.total_predictions)
    
    if st.session_state.total_predictions > 0:
        fault_rate = (st.session_state.fault_count / st.session_state.total_predictions) * 100
        stat_placeholder3.metric("⚠️ Fault Rate", f"{fault_rate:.1f}%")
    else:
        stat_placeholder3.metric("⚠️ Fault Rate", "0.0%")
    
    if st.session_state.predictions:
        latest = st.session_state.predictions[-1]
        labels = [res["label"] for res in latest["predictions"].values()]
        if len(set(labels)) == 1:
            agreement = "✅ All Agree"
        else:
            agreement = "⚠️ Disagree"
        stat_placeholder4.metric("🤝 Model Agreement", agreement)
    else:
        stat_placeholder4.metric("🤝 Model Agreement", "N/A")
    
    # Process Queue
    updated = False
    while not _mqtt_queue.empty():
        msg = _mqtt_queue.get()
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic
            
            if topic == MQTT_TOPIC_RAW:
                if payload.get("type") == "accel":
                    st.session_state.raw_data.append({
                        "time": datetime.now(),
                        "ax": payload["ax"],
                        "ay": payload["ay"],
                        "az": payload["az"]
                    })
                    updated = True
                    
            elif topic == MQTT_TOPIC_PRED:
                st.session_state.predictions.append(payload)
                st.session_state.total_predictions += 1
                
                # Count faults with 4-model voting (if ML available)
                preds = payload["predictions"]
                ml_pred = st.session_state.ml_prediction
                
                # Collect all votes
                votes = {"Good": [], "Faulty": []}
                for model_name, res in preds.items():
                    label = res["label"]
                    conf = res["confidence"]
                    votes[label].append((model_name, conf))
                
                # Add ML vote if available and recent (within 2 seconds)
                if ml_pred:
                    from datetime import datetime
                    ml_time = datetime.fromisoformat(ml_pred["timestamp"])
                    dl_time = datetime.fromisoformat(payload["timestamp"])
                    if abs((ml_time - dl_time).total_seconds()) < 2:
                        ml_label = ml_pred["label"]
                        ml_conf = ml_pred["confidence"]
                        votes[ml_label].append(("ml", ml_conf))
                
                # Determine fault with tie-breaking
                good_count = len(votes["Good"])
                faulty_count = len(votes["Faulty"])
                
                if faulty_count > good_count:
                    is_fault = True
                elif good_count > faulty_count:
                    is_fault = False
                else:  # Tie - use confidence-based voting
                    avg_good_conf = sum(c for _, c in votes["Good"]) / len(votes["Good"]) if votes["Good"] else 0
                    avg_faulty_conf = sum(c for _, c in votes["Faulty"]) / len(votes["Faulty"]) if votes["Faulty"] else 0
                    is_fault = avg_faulty_conf > avg_good_conf
                
                if is_fault:
                    st.session_state.fault_count += 1
                
                # Keep only last 50 predictions
                if len(st.session_state.predictions) > 50:
                    st.session_state.predictions.pop(0)
                    
                updated = True
                    
            elif topic == MQTT_TOPIC_ML_PRED:
                # Store latest ML prediction
                st.session_state.ml_prediction = payload
                updated = True
        except Exception as e:
            print(f"Error processing message: {e}")

    # Update Chart
    if st.session_state.raw_data:
        df = pd.DataFrame(st.session_state.raw_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["ax"], mode='lines', name='Accel X'))
        fig.add_trace(go.Scatter(y=df["ay"], mode='lines', name='Accel Y'))
        fig.add_trace(go.Scatter(y=df["az"], mode='lines', name='Accel Z'))
        fig.update_layout(
            height=300, 
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Time",
            yaxis_title="Acceleration (g)"
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Update Metrics
    if st.session_state.predictions:
        latest = st.session_state.predictions[-1]
        preds = latest["predictions"]
        
        with metrics_placeholder.container():
            # Display DL models
            for model_name, result in preds.items():
                label = result["label"]
                conf = result["confidence"]
                
                color = "green" if label == "Good" else "red"
                st.markdown(f"""
                **{model_name.upper()}**  
                :{color}[{label}] ({conf:.1%})
                """)
                st.progress(conf)
                st.divider()
            
            # Display ML model if available
            if st.session_state.ml_prediction:
                ml_pred = st.session_state.ml_prediction
                label = ml_pred["label"]
                conf = ml_pred["confidence"]
                timing = ml_pred.get("timing", {})
                total_ms = timing.get("total_ms", 0)
                
                color = "green" if label == "Good" else "red"
                st.markdown(f"""
                **ML (RF-20)** 🌟  
                :{color}[{label}] ({conf:.1%})  
                _Time: {total_ms:.1f}ms_
                """)
                st.progress(conf)
            else:
                st.info("ML: Waiting for prediction...")

    # Update Alerts Table
    if st.session_state.predictions:
        # Flatten for table
        rows = []
        for p in reversed(st.session_state.predictions):
            row = {"Time": p["timestamp"]}
            for m, res in p["predictions"].items():
                row[f"{m.upper()}"] = f"{res['label']} ({res['confidence']:.2f})"
            rows.append(row)
        
        alerts_placeholder.dataframe(pd.DataFrame(rows).head(5), hide_index=True)

    time.sleep(0.5)
