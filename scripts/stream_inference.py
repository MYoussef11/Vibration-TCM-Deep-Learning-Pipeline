"""
Inference Engine for Real-Time Vibration Analysis.
Subscribes to MQTT raw data, buffers it, and runs ML/DL models.
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
from scipy import signal

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcm_pipeline import PreprocessingConfig

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
MQTT_TOPIC_RAW = "sensors/vibration/raw"
MQTT_TOPIC_PRED = "sensors/vibration/prediction"
WINDOW_SIZE = 20  # Samples (2 seconds at 10Hz) - Adjusted to match existing models
SAMPLING_RATE = 10.0
MODELS_DIR = PROJECT_ROOT / "reports" / "phase3" / "dl_tuning"

class InferenceEngine:
    def __init__(self, broker: str):
        self.broker = broker
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.models = {}
        self.last_inference_time = 0
        self.inference_interval = 1.0  # Run inference every 1 second

        # Load Models
        self.load_models()

        # MQTT Setup
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def load_models(self):
        """Load trained Keras models."""
        try:
            # Load DL Models
            for name in ["cnn1d", "lstm", "cnn2d"]:
                model_path = MODELS_DIR / f"best_model_{name}.keras"
                if model_path.exists():
                    logger.info(f"Loading {name} from {model_path}...")
                    self.models[name] = tf.keras.models.load_model(model_path)
                else:
                    logger.warning(f"Model {name} not found at {model_path}")
            
            if not self.models:
                logger.error("No models loaded! Please train models first.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"Connected to MQTT Broker (rc={rc})")
        client.subscribe(MQTT_TOPIC_RAW)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            # We only care about Acceleration for now
            if payload.get("type") == "accel":
                # Current simple strategy: Store latest Accel and Gyro state
                self.latest_accel = [payload["ax"], payload["ay"], payload["az"]]
                
            elif payload.get("type") == "gyro":
                self.latest_gyro = [payload["wx"], payload["wy"], payload["wz"]]
                
                # When we get a gyro packet, let's assume we have a full "sample" 
                # (Accel + Gyro) and push to buffer.
                if hasattr(self, 'latest_accel'):
                    sample = self.latest_accel + self.latest_gyro
                    self.buffer.append(sample)
                    
                    # Check if it's time to run inference
                    if time.time() - self.last_inference_time > self.inference_interval:
                        self.run_inference()

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def compute_spectrogram(self, window):
        """Compute spectrogram for 2D CNN."""
        # Window shape: (20, 6)
        # We need to match the training preprocessing
        n_fft = 1024  # Match the existing models
        overlap = 0.5  # Standard overlap
        
        # Explicit padding if window is smaller than n_fft (matches build_phase3_datasets.py)
        if window.shape[0] < n_fft:
            pad_width = n_fft - window.shape[0]
            # Pad time dimension (axis 0)
            window = np.pad(window, ((0, pad_width), (0, 0)), mode='constant')

        # Helper from build_phase3_datasets.py logic
        hop = max(1, int(n_fft * (1 - overlap)))
        noverlap = n_fft - hop
        
        channel_specs = []
        for i in range(6): # 6 channels
            f, t, Zxx = signal.stft(
                window[:, i],
                fs=SAMPLING_RATE,
                nperseg=n_fft,
                noverlap=noverlap,
                padded=True,
                boundary=None
            )
            magnitude = np.abs(Zxx)
            channel_specs.append(magnitude)
        
        # Average across channels (as done in training)
        averaged = np.stack(channel_specs, axis=0).mean(axis=0)
        
        # Log scale
        spec = 20 * np.log10(np.maximum(averaged, 1e-12))
        
        # Add dimensions: (Freq, Time, 1) -> (1, Freq, Time, 1) for batch
        return spec[np.newaxis, ..., np.newaxis]

    def run_inference(self):
        if len(self.buffer) < WINDOW_SIZE:
            return

        self.last_inference_time = time.time()
        
        # Prepare Data
        window = np.array(list(self.buffer)) # Shape (100, 6)
        
        # 1. Prepare Inputs
        # 1D Models expect (1, 100, 6)
        X_time = window[np.newaxis, ...]
        
        # 2D Models expect Spectrogram
        X_spec = self.compute_spectrogram(window)
        
        results = {}
        
        # 2. Run Models
        for name, model in self.models.items():
            try:
                if name == "cnn2d":
                    pred = model.predict(X_spec, verbose=0)
                else:
                    pred = model.predict(X_time, verbose=0)
                
                # pred is [[prob_good, prob_faulty]] (if binary)
                # or [[prob_good, prob_mod, prob_bad]] (if 3-class)
                # We assume Binary (Good=0, Faulty=1) based on latest training
                
                confidence = float(np.max(pred))
                label_idx = int(np.argmax(pred))
                
                label_map = {0: "Good", 1: "Faulty"}
                label = label_map.get(label_idx, "Unknown")
                
                results[name] = {
                    "label": label,
                    "confidence": confidence,
                    "probs": pred[0].tolist()
                }
            except Exception as e:
                logger.error(f"Inference failed for {name}: {e}")

        # 3. Publish Results
        if results:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "predictions": results
            }
            self.client.publish(MQTT_TOPIC_PRED, json.dumps(payload))
            logger.info(f"Inference: {results}")

    def start(self):
        logger.info("Starting Inference Engine...")
        try:
            self.client.connect(self.broker, 1883, 60)
            self.client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Stopping Inference Engine...")
            self.client.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Inference Engine")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT Broker address")
    args = parser.parse_args()

    engine = InferenceEngine(args.broker)
    engine.start()

if __name__ == "__main__":
    main()
