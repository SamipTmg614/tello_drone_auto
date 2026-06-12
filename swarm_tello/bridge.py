"""
SLAVE BRIDGE - runs on the slave laptop alongside slave_dashboard.py
Connects to master over ethernet and pushes:
  - Live telemetry (battery, height, temp, IMU...)
  - Mission logs
  - Mission status (running, current step)

It reads state from slave_dashboard.py via shared `shared_state` dict.
Run both together:
    python slave_dashboard.py   (in one terminal)
    python bridge.py            (in another terminal)

Or import and start bridge as a thread from slave_dashboard.py directly.

Config — edit these:
"""

MASTER_IP   = "192.168.1.100"   # ethernet IP of master laptop
MASTER_PORT = 9100              # port master bridge_server listens on
DRONE_ID    = "drone2"          # label shown on master dashboard
PUSH_INTERVAL = 1.0             # seconds between telemetry pushes

import socket
import json
import time
import threading

# This dict is populated by slave_dashboard.py when imported together.
# If running standalone, it falls back to zeros.
shared_state = {
    "stats":   {},
    "log":     [],
    "running": False,
    "current": -1,
}

_sock      = None
_sock_lock = threading.Lock()

def _send(msg: dict):
    global _sock
    with _sock_lock:
        if _sock is None:
            return
        try:
            _sock.sendall((json.dumps(msg) + "\n").encode())
        except Exception:
            _sock = None

def _connect_loop():
    global _sock
    while True:
        try:
            print(f"[BRIDGE] Connecting to master {MASTER_IP}:{MASTER_PORT}...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((MASTER_IP, MASTER_PORT))
            s.settimeout(None)
            with _sock_lock:
                _sock = s
            print("[BRIDGE] Connected to master.")
            _send({"type": "hello", "id": DRONE_ID})

            # Keep reading (master may send pings or commands later)
            while True:
                try:
                    data = s.recv(1024)
                    if not data:
                        break
                except Exception:
                    break
        except Exception as e:
            print(f"[BRIDGE] Connection error: {e}")
        finally:
            with _sock_lock:
                _sock = None
        print("[BRIDGE] Retrying in 3s...")
        time.sleep(3)

def _push_loop():
    last_log_len = 0
    while True:
        time.sleep(PUSH_INTERVAL)
        # Push telemetry
        _send({
            "type":    "stats",
            "id":      DRONE_ID,
            "data":    shared_state.get("stats", {}),
            "running": shared_state.get("running", False),
            "current": shared_state.get("current", -1),
        })
        # Push any new log lines
        log = shared_state.get("log", [])
        if len(log) > last_log_len:
            new_lines = log[last_log_len:]
            for line in new_lines:
                _send({"type": "log", "id": DRONE_ID, "msg": line})
            last_log_len = len(log)

def start():
    """Call this to start the bridge in background threads."""
    threading.Thread(target=_connect_loop, daemon=True).start()
    threading.Thread(target=_push_loop,    daemon=True).start()
    print("[BRIDGE] Started.")

if __name__ == "__main__":
    start()
    print("[BRIDGE] Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[BRIDGE] Stopped.")