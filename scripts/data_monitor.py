"""
Data Stream Monitor - Debug/Validation Tool
Shows real-time data flow through the entire pipeline
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# MQTT Topics
TOPIC_RAW = "sensors/vibration/raw"
TOPIC_PRED = "sensors/vibration/prediction"


class DataMonitor:
    def __init__(self, broker: str, batch_size: int = None, pause_seconds: int = 3, interval: int = 1):
        """
        Initialize data monitor.
        
        Args:
            broker: MQTT broker address
            batch_size: Show N packets then pause (None = continuous)
            pause_seconds: Seconds to pause between batches
            interval: Show every Nth packet (1 = all packets)
        """
        self.broker = broker
        self.batch_size = batch_size
        self.pause_seconds = pause_seconds
        self.interval = interval
        self.stats = {
            'raw_packets': 0,
            'predictions': 0,
            'accel_packets': 0,
            'gyro_packets': 0,
            'last_accel': None,
            'last_gyro': None,
            'last_prediction': None
        }
        self.packets_shown = 0
        
        # MQTT Setup
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"Connected to MQTT Broker (rc={rc})")
        client.subscribe([(TOPIC_RAW, 0), (TOPIC_PRED, 0)])
        logger.info(f"Subscribed to: {TOPIC_RAW}, {TOPIC_PRED}")
        print("\n" + "="*80)
        print("DATA STREAM MONITOR ACTIVE")
        print("="*80)
        print("Watching for sensor data and predictions...")
        print("Press Ctrl+C to stop\n")
        
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            if msg.topic == TOPIC_RAW:
                self._handle_raw(payload)
            elif msg.topic == TOPIC_PRED:
                self._handle_prediction(payload)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def _handle_raw(self, payload: dict):
        """Handle raw sensor data."""
        self.stats['raw_packets'] += 1
        data_type = payload.get('type')
        
        # Check interval filter
        if self.interval > 1:
            if data_type == 'accel' and self.stats['accel_packets'] % self.interval != 0:
                return
            if data_type == 'gyro' and self.stats['gyro_packets'] % self.interval != 0:
                return
        
        if data_type == 'accel':
            self.stats['accel_packets'] += 1
            self.stats['last_accel'] = payload
            print(f"\n📡 ACCEL #{self.stats['accel_packets']}")
            print(f"   Time: {payload.get('timestamp', 'N/A')}")
            print(f"   Data: ax={payload.get('ax', 0):.3f}, "
                  f"ay={payload.get('ay', 0):.3f}, "
                  f"az={payload.get('az', 0):.3f}")
            self.packets_shown += 1
                  
        elif data_type == 'gyro':
            self.stats['gyro_packets'] += 1
            self.stats['last_gyro'] = payload
            print(f"\n🔄 GYRO  #{self.stats['gyro_packets']}")
            print(f"   Time: {payload.get('timestamp', 'N/A')}")
            print(f"   Data: wx={payload.get('wx', 0):.3f}, "
                  f"wy={payload.get('wy', 0):.3f}, "
                  f"wz={payload.get('wz', 0):.3f}")
            self.packets_shown += 1
                  
        # Check for complete sample (accel + gyro)
        if self.stats['last_accel'] and self.stats['last_gyro']:
            sample = [
                self.stats['last_accel']['ax'],
                self.stats['last_accel']['ay'],
                self.stats['last_accel']['az'],
                self.stats['last_gyro']['wx'],
                self.stats['last_gyro']['wy'],
                self.stats['last_gyro']['wz']
            ]
            print(f"   ✅ Complete Sample: {sample}")
            
        # Check if batch is complete
        if self.batch_size and self.packets_shown >= self.batch_size:
            self._pause_for_batch()
            
    def _pause_for_batch(self):
        """Pause and wait for user input between batches."""
        self.print_stats()
        print(f"\n⏸️  Batch complete. Press Enter to continue (or Ctrl+C to stop)...")
        try:
            input()
            self.packets_shown = 0
            print("\n" + "="*80)
            print("Resuming monitoring...")
            print("="*80 + "\n")
        except KeyboardInterrupt:
            raise
            
    def _handle_prediction(self, payload: dict):
        """Handle prediction results."""
        self.stats['predictions'] += 1
        self.stats['last_prediction'] = payload
        
        print(f"\n🤖 PREDICTION #{self.stats['predictions']}")
        print(f"   Time: {payload.get('timestamp', 'N/A')}")
        
        predictions = payload.get('predictions', {})
        for model, result in predictions.items():
            label = result.get('label', 'N/A')
            conf = result.get('confidence', 0)
            symbol = "✅" if label == "Good" else "⚠️"
            print(f"   {symbol} {model.upper()}: {label} ({conf:.1%})")
            
        # Check agreement
        labels = [res['label'] for res in predictions.values()]
        if len(set(labels)) == 1:
            print(f"   🤝 All models agree: {labels[0]}")
        else:
            print(f"   ⚡ Models disagree: {labels}")
            
    def print_stats(self):
        """Print current statistics."""
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"Total Raw Packets:  {self.stats['raw_packets']}")
        print(f"  - Accel:          {self.stats['accel_packets']}")
        print(f"  - Gyro:           {self.stats['gyro_packets']}")
        print(f"Total Predictions:  {self.stats['predictions']}")
        print("="*80 + "\n")
        
    def start(self):
        """Start the monitor."""
        try:
            self.client.connect(self.broker, 1883, 60)
            self.client.loop_forever()
        except KeyboardInterrupt:
            logger.info("\nStopping monitor...")
            self.print_stats()
            self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Data Stream Monitor")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT Broker")
    parser.add_argument(
        "--batch", 
        type=int, 
        help="Show N packets then pause (default: continuous)"
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=3,
        help="Seconds to pause between batches (default: 3)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Show every Nth packet (default: 1 = all packets)"
    )
    args = parser.parse_args()
    
    monitor = DataMonitor(
        args.broker,
        batch_size=args.batch,
        pause_seconds=args.pause,
        interval=args.interval
    )
    monitor.start()


if __name__ == "__main__":
    main()
