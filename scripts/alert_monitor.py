"""
Automated Alert Monitor for Vibration TCM System.
Monitors predictions in real-time and sends alerts when thresholds are exceeded.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.telegram_notifier import TelegramNotifier

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MQTT_TOPIC_PRED = "sensors/vibration/prediction"
MQTT_TOPIC_ML_PRED = "sensors/vibration/ml_prediction"


class AlertMonitor:
    def __init__(self, broker: str, telegram_notifier: TelegramNotifier, 
                 fault_threshold: int = 3, time_window: int = 60):
        """
        Initialize alert monitor.
        
        Args:
            broker: MQTT broker address
            telegram_notifier: TelegramNotifier instance
            fault_threshold: Number of consecutive faults before alerting
            time_window: Time window in seconds to track faults
        """
        self.broker = broker
        self.notifier = telegram_notifier
        self.fault_threshold = fault_threshold
        self.time_window = time_window
        
        # Tracking
        self.recent_faults = deque(maxlen=100)  # Last 100 predictions
        self.last_alert_time = None
        self.alert_cooldown = 300  # 5 minutes between alerts
        self.ml_prediction = None  # Store latest ML prediction
        
        # MQTT Setup
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"Connected to MQTT Broker (rc={rc})")
        client.subscribe([(MQTT_TOPIC_PRED, 0), (MQTT_TOPIC_ML_PRED, 0)])
        logger.info(f"Subscribed to DL and ML prediction topics")
        
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            # Store ML prediction if received
            if msg.topic == MQTT_TOPIC_ML_PRED:
                self.ml_prediction = payload
                logger.debug("ML prediction received and stored")
                return
            
            # Process DL prediction
            if msg.topic == MQTT_TOPIC_PRED:
                self._check_for_alerts(payload)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def _check_for_alerts(self, payload: dict):
        """Check if alert conditions are met."""
        predictions = payload.get("predictions", {})
        timestamp = payload.get("timestamp")
        
        # Count fault votes
        fault_votes = sum(
            1 for res in predictions.values() 
            if res.get("label") == "Faulty"
        )
        is_fault = fault_votes > len(predictions) / 2
        
        # Track fault
        self.recent_faults.append({
            'timestamp': timestamp,
            'is_fault': is_fault,
            'predictions': predictions
        })
        
        # Check consecutive faults
        recent_count = min(len(self.recent_faults), self.fault_threshold)
        consecutive_faults = all(
            entry['is_fault'] 
            for entry in list(self.recent_faults)[-recent_count:]
        )
        
        if consecutive_faults and len(self.recent_faults) >= self.fault_threshold:
            self._send_alert(predictions)
            
    def _send_alert(self, predictions: dict):
        """Send alert if conditions are met and cooldown has passed."""
        now = time.time()
        
        # Check cooldown
        if self.last_alert_time:
            time_since_last = now - self.last_alert_time
            if time_since_last < self.alert_cooldown:
                logger.info(f"Alert cooldown active ({time_since_last:.0f}s)")
                return
        
        # Build alert message with DL models
        model_results = []
        for name, res in predictions.items():
            model_results.append(f"  • {name.upper()}: {res['label']} ({res['confidence']:.1%})")
        
        # Add ML model if available and recent
        if self.ml_prediction:
            try:
                ml_time = datetime.fromisoformat(self.ml_prediction['timestamp'])
                current_time = datetime.now()
                time_diff = abs((current_time - ml_time).total_seconds())
                
                # Only include if recent (within 5 seconds)
                if time_diff < 5:
                    label = self.ml_prediction.get('label', 'Unknown')
                    conf = self.ml_prediction.get('confidence', 0)
                    timing = self.ml_prediction.get('timing', {})
                    total_ms = timing.get('total_ms', 0)
                    model_results.append(f"  • ML (RF-20): {label} ({conf:.1%}) - {total_ms:.1f}ms")
            except:
                pass  # Skip ML if timestamp parsing fails
        
        alert_msg = f"""⚠️ **FAULT DETECTED** ⚠️

**{self.fault_threshold} consecutive fault predictions detected!**

**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Model Results**:
{chr(10).join(model_results)}

**Recommendation**: Inspect equipment immediately.
"""
        
        # Send alert
        success = self.notifier.send_alert(alert_msg)
        
        if success:
            self.last_alert_time = now
            logger.info("Alert sent successfully")
        else:
            logger.error("Failed to send alert")
            
    def start(self):
        """Start the monitor."""
        logger.info("Starting Alert Monitor...")
        logger.info(f"Fault threshold: {self.fault_threshold} consecutive faults")
        logger.info(f"Alert cooldown: {self.alert_cooldown}s")
        
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
        """Stop the monitor."""
        logger.info("Stopping Alert Monitor...")
        self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Alert Monitor for Vibration TCM")
    parser.add_argument(
        "--broker", 
        type=str, 
        default="localhost", 
        help="MQTT Broker address"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Number of consecutive faults before alerting (default: 3)"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=300,
        help="Seconds between alerts (default: 300)"
    )
    args = parser.parse_args()
    
    # Get Telegram credentials
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("❌ Error: Telegram credentials not found!")
        print("Please configure .env file with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        sys.exit(1)
    
    try:
        # Initialize components
        notifier = TelegramNotifier(bot_token, chat_id)
        monitor = AlertMonitor(
            args.broker, 
            notifier,
            fault_threshold=args.threshold
        )
        monitor.alert_cooldown = args.cooldown
        
        # Start monitoring
        monitor.start()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
