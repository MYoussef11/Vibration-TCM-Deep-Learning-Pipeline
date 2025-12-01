"""
Data Logger for Vibration TCM Streaming Pipeline.
Subscribes to MQTT prediction topic and logs all results to SQLite database.
"""

import argparse
import json
import logging
import signal
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
MQTT_TOPIC_PRED = "sensors/vibration/prediction"
MQTT_TOPIC_ML_PRED = "sensors/vibration/ml_prediction"
DB_PATH = PROJECT_ROOT / "data" / "vibration_logs.db"


class DataLogger:
    def __init__(self, broker: str, db_path: Path):
        self.broker = broker
        self.db_path = db_path
        self.conn = None
        self.running = True
        
        # Setup database
        self._init_database()
        
        # Setup MQTT
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def _init_database(self):
        """Initialize SQLite database with schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cnn1d_label TEXT,
                cnn1d_confidence REAL,
                lstm_label TEXT,
                lstm_confidence REAL,
                cnn2d_label TEXT,
                cnn2d_confidence REAL,
                is_fault INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on timestamp for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON predictions(timestamp)
        """)
        
        # Create ML predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                label TEXT,
                confidence REAL,
                prob_good REAL,
                prob_faulty REAL,
                feature_time_ms REAL,
                inference_time_ms REAL,
                total_time_ms REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (timestamp) REFERENCES predictions(timestamp)
            )
        """)
        
        # Create index on ML timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_timestamp 
            ON ml_predictions(timestamp)
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"Connected to MQTT Broker (rc={rc})")
        client.subscribe([(MQTT_TOPIC_PRED, 0), (MQTT_TOPIC_ML_PRED, 0)])
        logger.info(f"Subscribed to {MQTT_TOPIC_PRED} and {MQTT_TOPIC_ML_PRED}")
        
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic == MQTT_TOPIC_PRED:
                self._log_prediction(payload)
            elif msg.topic == MQTT_TOPIC_ML_PRED:
                self._log_ml_prediction(payload)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def _log_prediction(self, payload: dict):
        """Log prediction to database."""
        timestamp = payload.get("timestamp")
        predictions = payload.get("predictions", {})
        
        # Extract model results
        cnn1d = predictions.get("cnn1d", {})
        lstm = predictions.get("lstm", {})
        cnn2d = predictions.get("cnn2d", {})
        
        # Determine if it's a fault (majority vote)
        labels = [cnn1d.get("label"), lstm.get("label"), cnn2d.get("label")]
        fault_count = sum(1 for label in labels if label == "Faulty")
        is_fault = 1 if fault_count > len(labels) / 2 else 0
        
        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, 
                cnn1d_label, cnn1d_confidence,
                lstm_label, lstm_confidence,
                cnn2d_label, cnn2d_confidence,
                is_fault
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            cnn1d.get("label"), cnn1d.get("confidence"),
            lstm.get("label"), lstm.get("confidence"),
            cnn2d.get("label"), cnn2d.get("confidence"),
            is_fault
        ))
        self.conn.commit()
        
        logger.info(f"Logged prediction: Fault={bool(is_fault)}, Models={labels}")
    
    def _log_ml_prediction(self, payload: dict):
        """Log ML prediction to database."""
        timestamp = payload.get("timestamp")
        label = payload.get("label")
        confidence = payload.get("confidence")
        probs = payload.get("probabilities", {})
        timing = payload.get("timing", {})
        
        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ml_predictions (
                timestamp, label, confidence,
                prob_good, prob_faulty,
                feature_time_ms, inference_time_ms, total_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            label,
            confidence,
            probs.get("Good"),
            probs.get("Faulty"),
            timing.get("feature_extraction_ms"),
            timing.get("inference_ms"),
           timing.get("total_ms")
        ))
        self.conn.commit()
        
        logger.info(f"Logged ML prediction: {label} ({confidence:.1%}), Time: {timing.get('total_ms', 0):.1f}ms")
        
    def start(self):
        """Start the logger."""
        logger.info("Starting Data Logger...")
        
        # Setup signal handlers for graceful shutdown
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
        """Stop the logger and cleanup."""
        logger.info("Stopping Data Logger...")
        if self.conn:
            self.conn.close()
        self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Data Logger for Vibration TCM")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT Broker address")
    parser.add_argument(
        "--db", 
        type=Path, 
        default=DB_PATH, 
        help="SQLite database path"
    )
    args = parser.parse_args()
    
    logger = DataLogger(args.broker, args.db)
    logger.start()


if __name__ == "__main__":
    main()
