# robot/serial_comm.py
"""
Encapsulates serial communication with the SCARA controller.
Reads lines in a background thread and puts parsed messages into the shared queue.
"""

import serial
import threading
import logging
import time
from utils.safe_queue import SafeQueue

class SerialComm:
    def __init__(self, port=None, baudrate=115200, timeout=0.05, queue=None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.queue = queue or SafeQueue()
        self.ser = None
        self._running = False
        self._thread = None

    def connect(self, port=None):
        if port:
            self.port = port
        if not self.port:
            raise ValueError("No port specified")
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout, dsrdtr=False)
        time.sleep(2)          # Allow Arduino/GRBL to reset
        self.ser.reset_input_buffer()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logging.info(f"SerialComm connected to {self.port}")

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        logging.info("SerialComm disconnected")

    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def send(self, cmd):
        """Send a G-code command (appends newline)."""
        if self.is_connected():
            try:
                self.ser.write(f"{cmd}\n".encode())
                # logging.debug(f"Sent: {cmd}")
            except serial.SerialException as e:
                logging.error(f"Serial write error: {e}")
        else:
            logging.warning(f"Not connected – command ignored: {cmd}")

    def _read_loop(self):
        """Continuously read lines from serial and push to queue."""
        while self._running and self.is_connected():
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self._parse_line(line)
            except serial.SerialException:
                break
            except Exception as e:
                logging.error(f"Serial read error: {e}")
            time.sleep(0.01)

    def _parse_line(self, line):
        """Parse known feedback lines and put a message dict into the queue."""
        logging.debug(f"Serial <<< {line}")
        if line.startswith("SW:"):
            states = [int(x) for x in line.replace("SW:", "").split(",")]
            self.queue.put({"type": "robot_switches", "data": states})
        elif line.startswith("POS:"):
            positions = line.replace("POS:", "").split(",")
            self.queue.put({"type": "robot_position", "data": positions})
        elif line == "HOMING_COMPLETE":
            self.queue.put({"type": "robot_homing_complete", "data": None})
        elif line == "HOMING_ABORTED":
            self.queue.put({"type": "robot_homing_aborted", "data": None})
        elif line == "ESTOP":
            self.queue.put({"type": "robot_estop", "data": None})
        # Ignore other lines (OK, ERR, etc.)