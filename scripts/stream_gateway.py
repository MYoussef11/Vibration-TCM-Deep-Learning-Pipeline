"""
Sensor Gateway for WitMotion IMU.
Reads data from Serial Port and publishes to MQTT.

Protocol:
- Acceleration (0x51): 0x55 0x51 AxL AxH AyL AyH AzL AzH TL TH SUM
- Angular Velocity (0x52): 0x55 0x52 WxL WxH WyL WyH WzL WzH TL TH SUM
"""

import argparse
import json
import logging
import struct
import time
from datetime import datetime

import paho.mqtt.client as mqtt
import serial
import serial.tools.list_ports

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
BAUD_RATE = 115200
MQTT_TOPIC_RAW = "sensors/vibration/raw"
HEADER_BYTE = 0x55
TYPE_ACCEL = 0x51
TYPE_GYRO = 0x52
TYPE_ANGLE = 0x53


def find_witmotion_port():
    """Auto-detect WitMotion sensor port (heuristic)."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # Common USB-Serial chips
        if "CH340" in p.description or "CP210" in p.description or "USB-SERIAL" in p.description:
            return p.device
    return None


def parse_packet(data):
    """Parse 11-byte WitMotion packet."""
    if len(data) != 11:
        return None
    
    # Checksum validation (Sum of first 10 bytes = 11th byte)
    checksum = sum(data[:10]) & 0xFF
    if checksum != data[10]:
        logger.warning("Checksum error")
        return None

    packet_type = data[1]
    
    # Parse payload (8 bytes: L H L H L H L H)
    # <h means little-endian short (2 bytes)
    values = struct.unpack("<hhhh", data[2:10])
    
    result = {}
    if packet_type == TYPE_ACCEL:
        # Acceleration: a = value / 32768.0 * 16.0 * 9.8 (g -> m/s^2)
        # WitMotion standard: output is g, range 16g
        scale = 16.0 * 9.8 / 32768.0
        result = {
            "type": "accel",
            "ax": values[0] * scale,
            "ay": values[1] * scale,
            "az": values[2] * scale,
            "temp": values[3] / 100.0
        }
    elif packet_type == TYPE_GYRO:
        # Angular Velocity: w = value / 32768.0 * 2000.0 (deg/s)
        scale = 2000.0 / 32768.0
        result = {
            "type": "gyro",
            "wx": values[0] * scale,
            "wy": values[1] * scale,
            "wz": values[2] * scale,
            "temp": values[3] / 100.0
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="WitMotion Sensor Gateway")
    parser.add_argument("--port", type=str, help="Serial port (e.g., COM3). Auto-detect if not provided.")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT Broker address")
    args = parser.parse_args()

    # 1. Setup MQTT
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(args.broker, 1883, 60)
        client.loop_start()
        logger.info(f"Connected to MQTT Broker at {args.broker}")
    except Exception as e:
        logger.error(f"Failed to connect to MQTT: {e}")
        return

    # 2. Setup Serial
    port = args.port or find_witmotion_port()
    if not port:
        logger.error("No suitable serial port found. Please specify --port.")
        return

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        logger.info(f"Connected to Serial Port: {port}")
    except Exception as e:
        logger.error(f"Failed to open serial port {port}: {e}")
        return

    # 3. Main Loop
    buffer = b""
    try:
        while True:
            # Read chunks
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buffer += chunk

            # Process buffer
            while len(buffer) >= 11:
                # Find header 0x55
                idx = buffer.find(HEADER_BYTE)
                if idx == -1:
                    # No header, discard all but last byte (might be start of header)
                    buffer = buffer[-1:]
                    break
                
                if idx > 0:
                    # Discard garbage before header
                    buffer = buffer[idx:]
                
                if len(buffer) < 11:
                    break # Wait for more data

                # Extract packet
                packet = buffer[:11]
                parsed = parse_packet(packet)
                
                if parsed:
                    # Add timestamp
                    parsed["timestamp"] = datetime.now().isoformat()
                    
                    # Publish to MQTT
                    payload = json.dumps(parsed)
                    client.publish(MQTT_TOPIC_RAW, payload)
                    # logger.debug(f"Published: {payload}")

                # Remove processed packet
                buffer = buffer[11:]

    except KeyboardInterrupt:
        logger.info("Stopping Gateway...")
    finally:
        ser.close()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
