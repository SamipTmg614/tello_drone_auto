"""
MASTER DASHBOARD - runs on master laptop
  - Controls Tello 1 directly (same engine as before)
  - Runs a TCP bridge server on port 9100 that the slave connects to
  - Slave telemetry, logs, and mission status appear live in the dashboard
  - Full mission builder for Drone 1
  - Read-only live panel for Drone 2 (slave)

Usage:
    python master_dashboard.py
    Dashboard at http://localhost:5000
"""

from djitellopy import Tello
from flask import Flask, Response, jsonify, request
import cv2
import threading
import time
import os
import av
import socket
import json
import urllib.request
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ---- Tello 1 (local) ------------------------------------------------------
tello = Tello()
tello.connect()
tello.streamon()
frame_reader = tello.get_frame_read()

# ---- Recording ------------------------------------------------------------
RECORD_DIR = "recordings"
os.makedirs(RECORD_DIR, exist_ok=True)
record_state = {"active": False, "writer": None, "filename": None,
                "lock": threading.Lock()}

def start_recording(frame):
    h, w = frame.shape[:2]
    fname = os.path.join(RECORD_DIR, datetime.now().strftime("rec_%Y%m%d_%H%M%S.avi"))
    writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
    record_state["writer"] = writer; record_state["filename"] = fname
    record_state["active"] = True;   log1(f"Recording started: {fname}")

def stop_recording():
    with record_state["lock"]:
        if record_state["writer"]:
            record_state["writer"].release(); record_state["writer"] = None
        record_state["active"] = False
        fname = record_state["filename"]; record_state["filename"] = None
    log1(f"Recording saved: {fname}")

# ---- Slave state (received over ethernet) ---------------------------------
slave = {
    "id":      "drone2",
    "connected": False,
    "stats":   {},
    "log":     [],
    "running": False,
    "current": -1,
}
slave_lock = threading.Lock()

# ---- Drone 2 video proxy --------------------------------------------------
SLAVE_IP        = "192.168.1.101"
SLAVE_VIDEO_URL = f"http://{SLAVE_IP}:5000/video"

d2_frame_lock  = threading.Lock()
d2_latest_jpeg = None   # None = no frame yet (shows placeholder)

def d2_proxy_thread():
    """Pull slave MJPEG stream, store latest JPEG for /video2 to serve."""
    global d2_latest_jpeg
    while True:
        try:
            print(f"[PROXY] Connecting to {SLAVE_VIDEO_URL}...")
            req = urllib.request.urlopen(SLAVE_VIDEO_URL, timeout=10)
            buf = b""
            while True:
                chunk = req.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    start = buf.find(b'\xff\xd8')
                    end   = buf.find(b'\xff\xd9')
                    if start == -1 or end == -1 or end < start:
                        break
                    jpeg = buf[start:end + 2]
                    buf  = buf[end + 2:]
                    with d2_frame_lock:
                        d2_latest_jpeg = jpeg
        except Exception as e:
            print(f"[PROXY] Slave video lost: {e} — retrying in 3s")
            with d2_frame_lock:
                d2_latest_jpeg = None
            time.sleep(3)

def gen_d2_frames():
    """Serve Drone 2 frames as MJPEG to the browser."""
    placeholder = None
    while True:
        with d2_frame_lock:
            jpeg = d2_latest_jpeg
        if jpeg:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(1 / 20)
        else:
            if placeholder is None:
                img = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(img, "Drone 2 — No Signal", (55, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
                _, buf = cv2.imencode('.jpg', img)
                placeholder = buf.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(1)

# ---- Mission engine (Drone 1) ---------------------------------------------
mission_status = {"running": False, "log": [], "current": -1, "cancel": False}
mission_lock   = threading.Lock()

STABILIZE_AFTER_TAKEOFF = 4
MISSION_SPEED   = 40
MAX_MOVE_CHUNK  = 250
MOVE_RETRIES    = 2
RETRY_SETTLE    = 1.5
MIN_BATTERY     = 20
USE_RC_FALLBACK = True
RC_SPEED        = 30
RC_CM_PER_S     = 30
RC_DEG_PER_S    = 45
RC_AXES = {
    "up":      (0,  0,  1, 0), "down":    (0,  0, -1, 0),
    "forward": (0,  1,  0, 0), "back":    (0, -1,  0, 0),
    "left":    (-1, 0,  0, 0), "right":   (1,  0,  0, 0),
    "cw":      (0,  0,  0, 1), "ccw":     (0,  0,  0, -1),
}

def log1(msg):
    print(f"[D1] {msg}")
    mission_status["log"].append(msg)
    if len(mission_status["log"]) > 40: mission_status["log"].pop(0)

MOVE_CMDS = {
    "up": tello.move_up, "down": tello.move_down,
    "forward": tello.move_forward, "back": tello.move_back,
    "left": tello.move_left, "right": tello.move_right,
}
ROTATE_CMDS = {
    "cw": tello.rotate_clockwise, "ccw": tello.rotate_counter_clockwise,
}
FLIP_DIRS = {"l", "r", "f", "b"}
ALL_TYPES = set(MOVE_CMDS) | set(ROTATE_CMDS) | {"flip","wait","takeoff","land","emergency"}
LABELS = {
    "takeoff":"Takeoff","land":"Land","up":"Up","down":"Down",
    "forward":"Forward","back":"Back","left":"Left","right":"Right",
    "cw":"Rotate CW","ccw":"Rotate CCW","flip":"Flip","wait":"Wait","emergency":"EMERGENCY",
}

def step_desc(step):
    t = step.get("type"); v = step.get("value")
    label = LABELS.get(t, t)
    if t in ("takeoff","land"): return label
    if t == "flip": return f"{label} {v}"
    if t == "wait": return f"{label} {v}s"
    return f"{label} {v}{'deg' if t in ROTATE_CMDS else 'cm'}"

def validate_steps(steps):
    if not isinstance(steps, list) or not steps: return "no steps"
    for i, s in enumerate(steps):
        t = s.get("type") if isinstance(s, dict) else None
        if t not in ALL_TYPES: return f"step {i+1}: unknown type '{t}'"
        if t in MOVE_CMDS or t in ROTATE_CMDS or t == "wait":
            try: float(s.get("value"))
            except: return f"step {i+1}: '{t}' needs a number"
        if t == "flip" and s.get("value") not in FLIP_DIRS:
            return f"step {i+1}: flip needs l/r/f/b"
    return None

def _rc_move(direction, amount, per_sec):
    lr, fb, ud, yaw = RC_AXES[direction]; s = RC_SPEED
    dur = amount / max(1, per_sec); end = time.time() + dur
    log1(f"  rc-fallback {direction} ~{amount} (~{dur:.1f}s)")
    while time.time() < end:
        if mission_status.get("cancel"): return
        tello.send_rc_control(lr*s, fb*s, ud*s, yaw*s); time.sleep(0.05)
    tello.send_rc_control(0, 0, 0, 0)

def _do_with_retry(direction, amount, per_sec):
    precise = MOVE_CMDS.get(direction) or ROTATE_CMDS.get(direction)
    for attempt in range(MOVE_RETRIES + 1):
        try: precise(amount); return
        except Exception as e:
            if mission_status.get("cancel"): raise
            if attempt < MOVE_RETRIES:
                log1(f"  {direction} {amount} failed: {e} — settle {RETRY_SETTLE}s & retry")
                time.sleep(RETRY_SETTLE)
            elif USE_RC_FALLBACK:
                _rc_move(direction, amount, per_sec); return
            else: raise

def exec_step(step):
    t = step.get("type"); v = step.get("value")
    if t == "takeoff":   tello.takeoff()
    elif t == "land":    tello.land()
    elif t == "emergency": tello.emergency()
    elif t in MOVE_CMDS:
        dist = max(20, min(500, int(float(v))))
        n = (dist + MAX_MOVE_CHUNK - 1) // MAX_MOVE_CHUNK
        base, extra = divmod(dist, n)
        for k in range(n):
            if mission_status.get("cancel"): return
            _do_with_retry(t, base + (1 if k < extra else 0), RC_CM_PER_S)
            if k < n - 1: time.sleep(0.4)
    elif t in ROTATE_CMDS:
        _do_with_retry(t, max(1, min(360, int(float(v)))), RC_DEG_PER_S)
    elif t == "flip":
        if v not in FLIP_DIRS: raise ValueError(f"bad flip dir '{v}'")
        tello.flip(v)
    elif t == "wait": time.sleep(float(v))
    else: raise ValueError(f"unknown step '{t}'")

def abort_mission(stop_cmd="land"):
    mission_status["cancel"] = True
    try: tello.send_command_without_return(stop_cmd); log1(f"!! Override: {stop_cmd.upper()}")
    except Exception as e: log1(f"abort send failed: {e}")

def run_mission(steps):
    if not mission_lock.acquire(blocking=False): log1("Already running."); return
    try:
        mission_status.update({"running": True, "cancel": False, "log": [], "current": -1})
        log1(f"Mission start — {len(steps)} steps")
        try:
            batt = tello.get_battery(); log1(f"Battery {batt}%")
            if batt < MIN_BATTERY: log1(f"Battery too low — aborting."); return
        except Exception as e: log1(f"battery check failed: {e}")
        try: tello.set_speed(MISSION_SPEED)
        except: pass
        for i, step in enumerate(steps):
            if mission_status["cancel"]: log1("Aborted."); break
            mission_status["current"] = i
            log1(f"[{i+1}/{len(steps)}] {step_desc(step)}")
            exec_step(step)
            if step.get("type") == "takeoff":
                log1(f"stabilizing {STABILIZE_AFTER_TAKEOFF}s..."); time.sleep(STABILIZE_AFTER_TAKEOFF)
            if mission_status["cancel"]: log1("Aborted."); break
            time.sleep(0.3)
        else: log1("Mission complete.")
    except Exception as e:
        log1(f"Mission error: {e}")
        if not mission_status["cancel"]:
            try: tello.land(); log1("Auto-land.")
            except: pass
    finally:
        mission_status.update({"current": -1, "running": False, "cancel": False})
        mission_lock.release()

# ---- Bridge server (receives slave data) ----------------------------------
BRIDGE_PORT = 9100

def handle_slave_conn(conn, addr):
    print(f"[BRIDGE] Slave connected from {addr}")
    with slave_lock:
        slave["connected"] = True
    buf = ""
    try:
        while True:
            chunk = conn.recv(4096).decode(errors="ignore")
            if not chunk: break
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line: continue
                try:
                    msg = json.loads(line)
                    mtype = msg.get("type")
                    with slave_lock:
                        if mtype == "hello":
                            slave["id"] = msg.get("id", "drone2")
                        elif mtype == "stats":
                            slave["stats"]   = msg.get("data", {})
                            slave["running"] = msg.get("running", False)
                            slave["current"] = msg.get("current", -1)
                        elif mtype == "log":
                            slave["log"].append(msg.get("msg", ""))
                            if len(slave["log"]) > 40: slave["log"].pop(0)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[BRIDGE] Slave disconnected: {e}")
    finally:
        with slave_lock:
            slave["connected"] = False
        conn.close()

def bridge_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", BRIDGE_PORT))
    srv.listen(5)
    print(f"[BRIDGE] Listening on port {BRIDGE_PORT}")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle_slave_conn, args=(conn, addr), daemon=True).start()

# ---- Video ----------------------------------------------------------------
def gen_frames():
    while True:
        try:
            frame = frame_reader.frame
            if frame is None: time.sleep(0.05); continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with record_state["lock"]:
                if record_state["active"] and record_state["writer"]:
                    record_state["writer"].write(rgb)
            if record_state["active"]:
                cv2.circle(rgb, (20, 20), 8, (0, 0, 255), -1)
                cv2.putText(rgb, "REC", (34, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', rgb)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except av.error.InvalidDataError: time.sleep(0.05)
        except Exception as e: print(f"Frame error: {e}"); time.sleep(0.05)

# ---- Routes ---------------------------------------------------------------
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(gen_d2_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        'battery':     tello.get_battery(),
        'height':      tello.get_height(),
        'temp':        tello.get_temperature(),
        'speed_x':     tello.get_speed_x(),
        'speed_y':     tello.get_speed_y(),
        'speed_z':     tello.get_speed_z(),
        'pitch':       tello.get_pitch(),
        'roll':        tello.get_roll(),
        'yaw':         tello.get_yaw(),
        'flight_time': tello.get_flight_time(),
        'barometer':   tello.get_barometer(),
    })

@app.route('/slave/stats')
def slave_stats():
    with slave_lock:
        return jsonify(slave)

@app.route('/record/toggle', methods=['POST'])
def record_toggle():
    with record_state["lock"]:
        currently = record_state["active"]
    if currently:
        stop_recording(); return jsonify({"recording": False})
    else:
        frame = frame_reader.frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with record_state["lock"]: start_recording(rgb)
        return jsonify({"recording": True, "file": record_state["filename"]})

@app.route('/record/status')
def record_status():
    return jsonify({"recording": record_state["active"],
                    "filename":  record_state["filename"] or ""})

@app.route('/mission/start', methods=['POST'])
def start_mission():
    if mission_status["running"]: return jsonify({"status": "busy"}), 409
    data = request.get_json(silent=True) or {}
    steps = data.get("steps", [])
    err = validate_steps(steps)
    if err: return jsonify({"status": "error", "msg": err}), 400
    threading.Thread(target=run_mission, args=(steps,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/mission/abort', methods=['POST'])
def abort_route():
    if not mission_status["running"]: return jsonify({"status": "idle"})
    abort_mission("land"); return jsonify({"status": "aborting"})

@app.route('/mission/status')
def mission_status_route():
    return jsonify({"running": mission_status["running"],
                    "log":     mission_status["log"],
                    "current": mission_status["current"]})

@app.route('/control/<cmd>', methods=['POST'])
def control(cmd):
    if cmd not in ALL_TYPES: return jsonify({"status": "error", "msg": f"unknown cmd '{cmd}'"}), 400
    if cmd in ("land", "emergency"):
        if mission_status["running"]: abort_mission(cmd)
        else:
            try: exec_step({"type": cmd})
            except Exception as e: return jsonify({"status": "error", "msg": str(e)}), 500
        return jsonify({"status": "ok"})
    if mission_status["running"]: return jsonify({"status": "busy"}), 409
    v = request.args.get("value")
    step = {"type": cmd}
    if cmd in MOVE_CMDS:     step["value"] = v or 30
    elif cmd in ROTATE_CMDS: step["value"] = v or 45
    elif cmd == "flip":      step["value"] = v or "f"
    try: exec_step(step); return jsonify({"status": "ok"})
    except Exception as e:   return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/')
def index():
    return DASHBOARD_HTML

# ---- HTML -----------------------------------------------------------------
DASHBOARD_HTML = '''<!DOCTYPE html>
<html>
<head>
<title>Tello Swarm Master</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0d0d0d; color:#e0e0e0; font-family:'Segoe UI',sans-serif; padding:20px; }
h2 { font-size:18px; font-weight:500; letter-spacing:.05em; color:#aaa; margin-bottom:16px; text-transform:uppercase; }

/* ---- Drone header tabs ---- */
.drone-tabs { display:flex; gap:12px; margin-bottom:20px; }
.drone-tab {
    padding:10px 20px; border-radius:8px; font-size:13px; font-weight:600;
    border:1px solid #222; background:#161616; color:#666; cursor:pointer;
    display:flex; align-items:center; gap:8px;
}
.drone-tab.active { border-color:#2563eb; color:#60a5fa; background:#0d1a33; }
.drone-tab .dot { width:8px; height:8px; border-radius:50%; background:#333; }
.drone-tab.d1 .dot { background:#22c55e; }
.drone-tab.d2 .dot { background:#f59e0b; }
.drone-tab.d2.offline .dot { background:#333; }
.drone-tab.d2.offline { opacity:.5; }

/* ---- Two-drone layout ---- */
.swarm-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.drone-col { display:flex; flex-direction:column; gap:14px; }
.drone-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 14px; background:#161616; border:1px solid #222;
    border-radius:10px; font-size:13px;
}
.drone-header .name { font-weight:700; font-size:15px; }
.drone-header .badge {
    padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600;
}
.badge-online  { background:#14532d; color:#4ade80; }
.badge-offline { background:#1c1c1c; color:#555; }
.badge-running { background:#451a03; color:#f59e0b; animation:pulse 1s infinite; }

/* ---- Stat cards ---- */
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(100px,1fr)); gap:8px; }
.card { background:#161616; border:1px solid #222; border-radius:10px; padding:12px 14px; }
.card .label { font-size:10px; color:#555; margin-bottom:4px; text-transform:uppercase; letter-spacing:.08em; }
.card .value { font-size:20px; font-weight:600; color:#fff; }
.card .unit  { font-size:11px; color:#555; margin-left:2px; }

/* ---- Video ---- */
.video-wrap { position:relative; }
.video-wrap img { width:100%; border-radius:10px; border:1px solid #222; display:block; }
.btn-record {
    position:absolute; top:10px; right:10px;
    background:rgba(20,20,20,.85); border:1px solid #333; border-radius:8px;
    padding:7px 12px; color:#e0e0e0; font-size:12px; font-weight:600;
    cursor:pointer; display:flex; align-items:center; gap:6px;
    backdrop-filter:blur(4px);
}
.btn-record.recording { border-color:#ef4444; color:#ef4444; }
.rec-dot { width:8px; height:8px; border-radius:50%; background:#555; }
.btn-record.recording .rec-dot { background:#ef4444; animation:blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
.rec-filename {
    position:absolute; bottom:10px; left:10px;
    background:rgba(0,0,0,.7); border-radius:6px; padding:3px 8px;
    font-size:11px; font-family:monospace; color:#ef4444; display:none;
}
.rec-filename.visible { display:block; }

/* ---- Panels ---- */
.panel { background:#161616; border:1px solid #222; border-radius:10px; padding:16px; }
.panel h3 { font-size:12px; color:#555; text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px; }

/* ---- Log ---- */
.log-box {
    background:#0d0d0d; border:1px solid #1a1a1a; border-radius:8px;
    padding:10px 12px; font-size:11px; font-family:monospace; color:#4ade80;
    min-height:80px; max-height:160px; overflow-y:auto; line-height:1.7;
}

/* ---- Buttons ---- */
.btn { border:none; border-radius:6px; padding:8px 10px; font-size:13px;
       font-weight:600; cursor:pointer; color:#fff; background:#333;
       transition:background .15s; }
.btn:hover { background:#444; }
.btn-mission { background:#2563eb; color:#fff; border:none; border-radius:8px;
               padding:11px; font-size:13px; font-weight:600; cursor:pointer;
               width:100%; transition:background .2s; }
.btn-mission:hover { background:#1d4ed8; }
.btn-mission:disabled { background:#1e3a6e; color:#555; cursor:not-allowed; }
.btn-emergency { width:100%; padding:12px; background:#dc2626; color:#fff;
                 border:none; border-radius:8px; font-size:14px; font-weight:700;
                 cursor:pointer; margin-top:8px; }
.btn-emergency:hover { background:#b91c1c; }
.btn-cut { width:100%; padding:7px; background:#3a0d0d; color:#f87171;
           border:1px solid #7f1d1d; border-radius:8px; font-size:12px;
           font-weight:600; cursor:pointer; margin-top:5px; }
.btn-cut:hover { background:#511414; }

/* ---- Manual pad ---- */
.pad { display:grid; grid-template-columns:repeat(3,1fr); gap:5px; }
.pad .btn { padding:9px 0; background:#1e1e1e; font-size:12px; }
.pad .btn:hover { background:#2a2a2a; }
.pad .btn.takeoff { background:#166534; }
.pad .btn.land    { background:#7f1d1d; }
.pad .spacer { visibility:hidden; }
.manual-val { display:flex; align-items:center; gap:6px; margin-top:8px;
              font-size:11px; color:#666; }
.manual-val input { width:54px; background:#0d0d0d; border:1px solid #2a2a2a;
                    color:#e0e0e0; border-radius:6px; padding:5px; font-size:12px; }

/* ---- Builder ---- */
.builder-row { display:flex; gap:6px; margin-bottom:10px; }
.builder-row select, .builder-row input {
    background:#0d0d0d; border:1px solid #2a2a2a; color:#e0e0e0;
    border-radius:6px; padding:7px; font-size:12px; }
.builder-row select { flex:1; min-width:0; }
.builder-row input  { width:58px; }
.steps-list { display:flex; flex-direction:column; gap:5px; margin-bottom:10px;
              max-height:200px; overflow-y:auto; }
.step-item { display:flex; align-items:center; gap:7px; background:#0d0d0d;
             border:1px solid #1f1f1f; border-radius:6px; padding:6px 9px;
             font-size:12px; color:#ccc; }
.step-item.active { border-color:#f59e0b; background:#1a1407; }
.step-item .num   { color:#555; font-size:10px; width:16px; flex-shrink:0; }
.step-item .desc  { flex:1; }
.ico-btn { background:none; border:none; color:#666; cursor:pointer; font-size:12px; padding:1px 3px; }
.ico-btn:hover { color:#fff; }
.empty-hint { color:#444; font-size:11px; padding:6px 0; }

/* ---- Slave read-only panel ---- */
.slave-offline-msg { color:#555; font-size:13px; padding:20px 0; text-align:center; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
              background:#22c55e; margin-right:5px; vertical-align:middle; }
.status-dot.running { background:#f59e0b; animation:pulse 1s infinite; }
</style>
</head>
<body>
<h2>&#9679; Tello Swarm Dashboard</h2>

<div class="swarm-grid">

  <!-- ===================== DRONE 1 (local) ===================== -->
  <div class="drone-col">
    <div class="drone-header">
      <span class="name">🚁 Drone 1 <span style="font-size:11px;color:#555;">(this machine)</span></span>
      <span class="badge badge-online" id="d1badge">Online</span>
    </div>

    <div class="grid" id="d1stats">
      <div class="card"><div class="label">Battery</div><div class="value" id="d1battery">--<span class="unit">%</span></div></div>
      <div class="card"><div class="label">Height</div><div class="value" id="d1height">--<span class="unit">cm</span></div></div>
      <div class="card"><div class="label">Temp</div><div class="value" id="d1temp">--<span class="unit">°C</span></div></div>
      <div class="card"><div class="label">Pitch</div><div class="value" id="d1pitch">--<span class="unit">°</span></div></div>
      <div class="card"><div class="label">Roll</div><div class="value" id="d1roll">--<span class="unit">°</span></div></div>
      <div class="card"><div class="label">Yaw</div><div class="value" id="d1yaw">--<span class="unit">°</span></div></div>
      <div class="card"><div class="label">Flight Time</div><div class="value" id="d1flighttime">--<span class="unit">s</span></div></div>
      <div class="card"><div class="label">Barometer</div><div class="value" id="d1baro">--<span class="unit">cm</span></div></div>
    </div>

    <div class="video-wrap">
      <img src="/video">
      <button class="btn-record" id="recBtn" onclick="toggleRecord()">
        <span class="rec-dot"></span><span id="recLabel">Record</span>
      </button>
      <div class="rec-filename" id="recFilename"></div>
    </div>

    <div class="panel">
      <h3>Manual Control — Drone 1</h3>
      <div class="pad">
        <button class="btn takeoff" onclick="ctrl('takeoff')">🛫</button>
        <button class="btn" onclick="ctrl('up')">⬆</button>
        <button class="btn land" onclick="ctrl('land')">🛬</button>
        <button class="btn" onclick="ctrl('ccw')">⟲</button>
        <button class="btn" onclick="ctrl('forward')">▲</button>
        <button class="btn" onclick="ctrl('cw')">⟳</button>
        <button class="btn" onclick="ctrl('left')">◀</button>
        <button class="btn" onclick="ctrl('back')">▼</button>
        <button class="btn" onclick="ctrl('right')">▶</button>
        <button class="btn" onclick="ctrl('flip','f')">⤿</button>
        <button class="btn" onclick="ctrl('down')">⬇</button>
        <div class="spacer"></div>
      </div>
      <div class="manual-val">
        move <input type="number" id="manualDist" value="30" min="20" max="500"> cm
        · turn <input type="number" id="manualAng" value="45" min="1" max="360"> °
      </div>
    </div>

    <div class="panel">
      <h3>Mission Builder — Drone 1</h3>
      <div class="builder-row">
        <select id="cmdType" onchange="onTypeChange()">
          <option value="takeoff">Takeoff</option>
          <option value="up">Up (cm)</option>
          <option value="down">Down (cm)</option>
          <option value="forward">Forward (cm)</option>
          <option value="back">Back (cm)</option>
          <option value="left">Left (cm)</option>
          <option value="right">Right (cm)</option>
          <option value="cw">Rotate CW (deg)</option>
          <option value="ccw">Rotate CCW (deg)</option>
          <option value="flip">Flip (l/r/f/b)</option>
          <option value="wait">Wait (s)</option>
          <option value="land">Land</option>
        </select>
        <input type="text" id="cmdVal" placeholder="val">
        <button class="btn" onclick="addStep()">+ Add</button>
      </div>
      <div class="steps-list" id="stepsList"></div>
      <button class="btn-mission" id="missionBtn" onclick="runMission()">▶ Run Mission</button>
      <div style="display:flex;gap:8px;margin-top:8px;">
        <button class="btn" style="flex:1" onclick="loadSquare()">▢ Square</button>
        <button class="btn" style="flex:1" onclick="clearSteps()">✕ Clear</button>
      </div>
      <button class="btn-emergency" onclick="emergencyLand()">🛬 EMERGENCY LAND D1</button>
      <button class="btn-cut" onclick="emergencyCut()">⛔ Cut motors D1</button>
      <div style="margin-top:12px;">
        <div style="font-size:11px;color:#555;margin-bottom:5px;text-transform:uppercase;">
          <span class="status-dot" id="d1statusDot"></span>
          <span id="d1statusText">Idle</span>
        </div>
        <div class="log-box" id="d1log">Waiting...</div>
      </div>
    </div>
  </div>

  <!-- ===================== DRONE 2 (slave — read-only) ===================== -->
  <div class="drone-col">
    <div class="drone-header">
      <span class="name">🚁 Drone 2 <span style="font-size:11px;color:#555;" id="d2id">(slave)</span></span>
      <span class="badge badge-offline" id="d2badge">Offline</span>
    </div>

    <div class="grid">
      <div class="card"><div class="label">Battery</div><div class="value" id="d2battery">--<span class="unit">%</span></div></div>
      <div class="card"><div class="label">Height</div><div class="value" id="d2height">--<span class="unit">cm</span></div></div>
      <div class="card"><div class="label">Temp</div><div class="value" id="d2temp">--<span class="unit">°C</span></div></div>
      <div class="card"><div class="label">Pitch</div><div class="value" id="d2pitch">--<span class="unit">°</span></div></div>
      <div class="card"><div class="label">Roll</div><div class="value" id="d2roll">--<span class="unit">°</span></div></div>
      <div class="card"><div class="label">Yaw</div><div class="value" id="d2yaw">--<span class="unit">°</span></div></div>
      <div class="card"><div class="label">Flight Time</div><div class="value" id="d2flighttime">--<span class="unit">s</span></div></div>
      <div class="card"><div class="label">Barometer</div><div class="value" id="d2baro">--<span class="unit">cm</span></div></div>
    </div>

    <div class="panel" style="text-align:center;">
      <h3>Live Feed — Drone 2</h3>
      <div class="video-wrap" style="text-align:left;">
        <img src="/video2" alt="Drone 2 Feed" style="width:100%;border-radius:8px;border:1px solid #222;display:block;background:#111;">
        <div class="feed-badge" id="d2feedbadge" style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,.7);border-radius:6px;padding:4px 10px;font-size:11px;font-weight:600;color:#555;">Offline</div>
      </div>
    </div>

    <div class="panel">
      <h3>Mission Status — Drone 2 <span style="color:#444;font-weight:400;">(read-only)</span></h3>
      <div style="font-size:11px;color:#555;margin-bottom:8px;text-transform:uppercase;">
        <span class="status-dot" id="d2statusDot"></span>
        <span id="d2statusText">Offline</span>
      </div>
      <div class="log-box" id="d2log">Waiting for slave connection...</div>
      <div style="margin-top:10px;font-size:11px;color:#444;">
        Control Drone 2 missions from the slave dashboard at
        <a href="http://192.168.1.101:5000" target="_blank" style="color:#60a5fa;">192.168.1.101:5000</a>
      </div>
    </div>

  </div>
</div>

<script>
// ---- Drone 1 stats ----
async function updateD1Stats() {
  const r = await fetch('/stats'); const d = await r.json();
  const s = (v, u) => v + '<span class="unit">' + u + '</span>';
  document.getElementById('d1battery').innerHTML    = s(d.battery, '%');
  document.getElementById('d1height').innerHTML     = s(d.height, 'cm');
  document.getElementById('d1temp').innerHTML       = s(d.temp, '°C');
  document.getElementById('d1pitch').innerHTML      = s(d.pitch, '°');
  document.getElementById('d1roll').innerHTML       = s(d.roll, '°');
  document.getElementById('d1yaw').innerHTML        = s(d.yaw, '°');
  document.getElementById('d1flighttime').innerHTML = s(d.flight_time, 's');
  document.getElementById('d1baro').innerHTML       = s(d.barometer, 'cm');
}

// ---- Drone 2 stats (from slave bridge) ----
async function updateD2Stats() {
  const r = await fetch('/slave/stats'); const d = await r.json();
  const s  = (v, u) => (v !== undefined ? v : '--') + '<span class="unit">' + u + '</span>';
  const sd = d.stats || {};
  document.getElementById('d2battery').innerHTML    = s(sd.battery, '%');
  document.getElementById('d2height').innerHTML     = s(sd.height, 'cm');
  document.getElementById('d2temp').innerHTML       = s(sd.temp, '°C');
  document.getElementById('d2pitch').innerHTML      = s(sd.pitch, '°');
  document.getElementById('d2roll').innerHTML       = s(sd.roll, '°');
  document.getElementById('d2yaw').innerHTML        = s(sd.yaw, '°');
  document.getElementById('d2flighttime').innerHTML = s(sd.flight_time, 's');
  document.getElementById('d2baro').innerHTML       = s(sd.barometer, 'cm');

  const online = d.connected;
  const badge  = document.getElementById('d2badge');
  badge.textContent = online ? (d.running ? 'Running' : 'Online') : 'Offline';
  badge.className   = 'badge ' + (online ? (d.running ? 'badge-running' : 'badge-online') : 'badge-offline');
  if (d.id) document.getElementById('d2id').textContent = d.id;

  // Feed badge on video panel
  const fb = document.getElementById('d2feedbadge');
  if (fb) {
    fb.textContent = online ? (d.running ? '● Running' : '● Online') : '○ Offline';
    fb.style.color = online ? (d.running ? '#f59e0b' : '#4ade80') : '#555';
  }

  const dot = document.getElementById('d2statusDot');
  const txt = document.getElementById('d2statusText');
  dot.className = 'status-dot' + (d.running ? ' running' : '');
  txt.textContent = online ? (d.running ? 'Running' : 'Idle') : 'Offline';

  const logEl = document.getElementById('d2log');
  if (d.log && d.log.length) {
    logEl.innerHTML = d.log.map(l => '<div>' + l + '</div>').join('');
    logEl.scrollTop = logEl.scrollHeight;
  }
}

// ---- Record ----
async function toggleRecord() {
  const r = await fetch('/record/toggle', { method:'POST' });
  const d = await r.json();
  const btn = document.getElementById('recBtn');
  const lbl = document.getElementById('recLabel');
  const fn  = document.getElementById('recFilename');
  btn.classList.toggle('recording', d.recording);
  lbl.textContent = d.recording ? 'Stop' : 'Record';
  if (d.recording && d.file) {
    const parts = d.file.replace(/\\/g,'/').split('/');
    fn.textContent = '● ' + parts[parts.length-1];
    fn.classList.add('visible');
  } else { fn.classList.remove('visible'); }
}

// ---- Manual control ----
const MOVE = ['up','down','forward','back','left','right'];
const ROT  = ['cw','ccw'];
function ctrl(cmd, val) {
  let url = '/control/' + cmd;
  if (val !== undefined)        url += '?value=' + encodeURIComponent(val);
  else if (MOVE.includes(cmd)) url += '?value=' + document.getElementById('manualDist').value;
  else if (ROT.includes(cmd))  url += '?value=' + document.getElementById('manualAng').value;
  fetch(url, { method:'POST' }).catch(()=>{});
}

// ---- Mission builder ----
let steps = [];
const TYPE_META = {
  takeoff:{v:false},land:{v:false},
  up:{v:true,def:50},down:{v:true,def:50},forward:{v:true,def:50},back:{v:true,def:50},
  left:{v:true,def:50},right:{v:true,def:50},cw:{v:true,def:90},ccw:{v:true,def:90},
  flip:{v:true,def:'f'},wait:{v:true,def:2},
};
const LBL = {takeoff:'Takeoff',land:'Land',up:'Up',down:'Down',forward:'Forward',
  back:'Back',left:'Left',right:'Right',cw:'Rotate CW',ccw:'Rotate CCW',flip:'Flip',wait:'Wait'};

function onTypeChange() {
  const t = document.getElementById('cmdType').value;
  const inp = document.getElementById('cmdVal');
  const m = TYPE_META[t];
  if (m.v) { inp.style.visibility='visible'; inp.value=m.def; }
  else     { inp.style.visibility='hidden';  inp.value=''; }
}
function descOf(s) {
  const n = LBL[s.type]||s.type;
  if (s.type==='takeoff'||s.type==='land') return n;
  if (s.type==='flip') return n+' '+s.value;
  if (s.type==='wait') return n+' '+s.value+'s';
  return n+' '+s.value+(ROT.includes(s.type)?'°':'cm');
}
function renderSteps(cur) {
  const box = document.getElementById('stepsList');
  if (!steps.length) { box.innerHTML='<div class="empty-hint">No steps.</div>'; return; }
  box.innerHTML = steps.map((s,i) =>
    '<div class="step-item '+(i===cur?'active':'')+'">' +
    '<span class="num">'+(i+1)+'</span><span class="desc">'+descOf(s)+'</span>' +
    '<button class="ico-btn" onclick="moveStep('+i+',-1)">▲</button>' +
    '<button class="ico-btn" onclick="moveStep('+i+',1)">▼</button>' +
    '<button class="ico-btn" onclick="removeStep('+i+')">✕</button></div>').join('');
}
function addStep() {
  const t = document.getElementById('cmdType').value;
  const m = TYPE_META[t]; const step = {type:t};
  if (m.v) {
    const val = document.getElementById('cmdVal').value.trim();
    if (t==='flip') { if(!['l','r','f','b'].includes(val)){alert('flip needs l/r/f/b');return;} step.value=val; }
    else { if(val===''||isNaN(val)){alert('enter a number');return;} step.value=Number(val); }
  }
  steps.push(step); renderSteps(-1);
}
function removeStep(i) { steps.splice(i,1); renderSteps(-1); }
function moveStep(i,d) {
  const j=i+d; if(j<0||j>=steps.length) return;
  [steps[i],steps[j]]=[steps[j],steps[i]]; renderSteps(-1);
}
function clearSteps() { steps=[]; renderSteps(-1); }
function loadSquare() {
  steps=[{type:'takeoff'},{type:'up',value:500}];
  for(let i=0;i<4;i++){steps.push({type:'forward',value:200});if(i<3)steps.push({type:'ccw',value:90});}
  steps.push({type:'down',value:500},{type:'land'}); renderSteps(-1);
}
async function runMission() {
  if(!steps.length){alert('add steps first');return;}
  const r = await fetch('/mission/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({steps})});
  if(!r.ok){const d=await r.json().catch(()=>({}));alert(d.msg||'error');}
}
async function emergencyLand() { try{await fetch('/control/land',{method:'POST'});}catch(e){} }
async function emergencyCut()  {
  if(!confirm('CUT MOTORS? Drone drops immediately.')) return;
  try{await fetch('/control/emergency',{method:'POST'});}catch(e){}
}
async function pollD1Mission() {
  const r=await fetch('/mission/status'); const d=await r.json();
  const btn=document.getElementById('missionBtn');
  const dot=document.getElementById('d1statusDot');
  const txt=document.getElementById('d1statusText');
  const log=document.getElementById('d1log');
  btn.disabled=d.running; btn.textContent=d.running?'⏳ Running...':'▶ Run Mission';
  dot.className='status-dot'+(d.running?' running':'');
  txt.textContent=d.running?'Running':'Idle';
  renderSteps(d.running?d.current:-1);
  if(d.log&&d.log.length){log.innerHTML=d.log.map(l=>'<div>'+l+'</div>').join('');log.scrollTop=log.scrollHeight;}
}

onTypeChange(); renderSteps(-1);
setInterval(updateD1Stats, 1000);
setInterval(updateD2Stats, 1000);
setInterval(pollD1Mission, 800);
updateD1Stats(); updateD2Stats(); pollD1Mission();
</script>
</body>
</html>'''

# ---- Boot -----------------------------------------------------------------
if __name__ == '__main__':
    threading.Thread(target=bridge_server,   daemon=True).start()
    threading.Thread(target=d2_proxy_thread, daemon=True).start()
    print("Master dashboard at: http://localhost:5000")
    print(f"Bridge server:       port {BRIDGE_PORT}")
    print(f"Drone 2 video proxy: {SLAVE_VIDEO_URL}")
    app.run(host='0.0.0.0', port=5000, threaded=True)