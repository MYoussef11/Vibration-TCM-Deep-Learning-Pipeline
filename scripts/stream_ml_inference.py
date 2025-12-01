"""
ML Model Inference Engine for Streaming Vibration Data.
Runs in parallel with DL models, using top 20 features.
"""

import json
import logging
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import paho.mqtt.client as mqtt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feature_extractor import SimplifiedFeatureExtractor

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MQTT_BROKER = "localhost"
MQTT_TOPIC_RAW = "sensors/vibration/raw"
MQTT_TOPIC_ML_PRED = "sensors/vibration/ml_prediction"
WINDOW_SIZE = 20
SAMPLING_RATE = 10


class MLInferenceEngine:
    def __init__(self, model_path, broker=MQTT_BROKER):
        self.broker = broker
        self.model = self._load_model(model_path)
        self.feature_extractor = SimplifiedFeatureExtractor(sampling_rate=SAMPLING_RATE)
        
        # Data buffer
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.last_accel = None
        self.last_gyro = None
        
        # MQTT Setup
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        # Stats
        self.predictions_count = 0
        
    def _load_model(self, model_path):
        """Load ML model."""
        logger.info(f"Loading ML model from {model_path}")
        model = joblib.load(model_path)
        logger.info(f"✅ ML model loaded: {type(model).__name__}")
        return model
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"Connected to MQTT Broker (rc={rc})")
        client.subscribe(MQTT_TOPIC_RAW)
        logger.info(f"Subscribed to {MQTT_TOPIC_RAW}")
        
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            # Collect complete samples (accel + gyro)
            if payload.get("type") == "accel":
                self.last_accel = [payload["ax"], payload["ay"], payload["az"]]
            elif payload.get("type") == "gyro":
                self.last_gyro = [payload["wx"], payload["wy"], payload["wz"]]
            
            # When we have both, add to buffer
            if self.last_accel is not None and self.last_gyro is not None:
                sample = self.last_accel + self.last_gyro
                self.buffer.append(sample)
                
                # Reset for next sample
                self.last_accel = None
                self.last_gyro = None
                
                # Run inference when buffer is full
                if len(self.buffer) == WINDOW_SIZE:
                    self._run_inference()
                    
        except Exception as e:
            logger.error(f"Error in on_message: {e}")
            
    def _run_inference(self):
        """Run ML model inference on current window."""
        try:
            # Convert buffer to numpy array
            window = np.array(list(self.buffer))
            
            # Extract features
            start_time = time.time()
            features = self.feature_extractor.extract_array(window).reshape(1, -1)
            feature_time = (time.time() - start_time) * 1000
            
            # Run prediction
            start_time = time.time()
            prediction = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            inference_time = (time.time() - start_time) * 1000
            
            # Map prediction to label
            label = "Good" if prediction == 0 else "Faulty"
            confidence = float(proba[prediction])
            
            # Prepare result
            result = {
                "timestamp": datetime.now().isoformat(),
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    "Good": float(proba[0]),
                    "Faulty": float(proba[1])
                },
                "timing": {
                    "feature_extraction_ms": round(feature_time, 2),
                    "inference_ms": round(inference_time, 2),
                    "total_ms": round(feature_time + inference_time, 2)
                }
            }
            
            # Publish to MQTT
            self.client.publish(
                MQTT_TOPIC_ML_PRED,
                json.dumps(result),
                qos=0
            )
            
            self.predictions_count += 1
            logger.info(f"ML Prediction #{self.predictions_count}: {label} ({confidence:.1%}) | "
                       f"Time: {result['timing']['total_ms']:.1f}ms")
                       
        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            
    def start(self):
        """Start the ML inference engine."""
        logger.info("Starting ML Inference Engine...")
        logger.info(f"Window size: {WINDOW_SIZE} samples")
        logger.info(f"Features: {len(self.feature_extractor.feature_names)}")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            self.client.connect(self.broker, 1883, 60)
            self.client.loop_forever()
        except KeyboardInterrupt:
            self.stop()
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)
        
    def stop(self):
        """Stop the inference engine."""
        logger.info("Stopping ML Inference Engine...")
        self.client.disconnect()
        logger.info(f"Total predictions: {self.predictions_count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Inference Engine")
    parser.add_argument(
        "--model",
        type=str,
        default="reports/phase3/ml_top20/rf_top20_model.pkl",
        help="Path to trained ML model"
    )
    parser.add_argument(
        "--broker",
        type=str,
        default=MQTT_BROKER,
        help="MQTT Broker address"
    )
    args = parser.parse_args()
    
    # Resolve model path
    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Start engine
    engine = MLInferenceEngine(model_path, broker=args.broker)
    engine.start()


if __name__ == "__main__":
    main()
