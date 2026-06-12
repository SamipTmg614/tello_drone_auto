"""
SLAVE DASHBOARD - runs on slave laptop
  - Full independent Flask dashboard for Tello 2
  - Same features as single-drone dashboard (mission builder, manual control,
    recording, emergency land)
  - Shares state with bridge.py so master can see live telemetry + logs

Usage:
    python slave_dashboard.py
    Dashboard at http://localhost:5000

Edit TELLO_* and RECORD_DIR below if needed.
"""

from djitellopy import Tello
from flask import Flask, Response, jsonify, request
import cv2
import threading
import time
import os
import av
from datetime import datetime
import bridge   # <-- shares state with master via ethernet

app = Flask(__name__)

# ---- Tello ----------------------------------------------------------------
tello = Tello()
tello.connect()
tello.streamon()
time.sleep(2)   # wait for drone to start sending video frames
frame_reader = tello.get_frame_read()

# ---- Recording ------------------------------------------------------------
RECORD_DIR   = "recordings"
os.makedirs(RECORD_DIR, exist_ok=True)
record_state = {"active": False, "writer": None, "filename": None,
                "lock": threading.Lock()}

def start_recording(frame):
    h, w = frame.shape[:2]
    fname = os.path.join(RECORD_DIR, datetime.now().strftime("rec_%Y%m%d_%H%M%S.avi"))
    writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
    record_state["writer"]   = writer
    record_state["filename"] = fname
    record_state["active"]   = True
    log(f"Recording started: {fname}")

def stop_recording():
    with record_state["lock"]:
        if record_state["writer"]:
            record_state["writer"].release()
            record_state["writer"] = None
        record_state["active"]   = False
        fname = record_state["filename"]
        record_state["filename"] = None
    log(f"Recording saved: {fname}")

# ---- Mission engine -------------------------------------------------------
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

def log(msg):
    print(msg)
    mission_status["log"].append(msg)
    if len(mission_status["log"]) > 40:
        mission_status["log"].pop(0)
    # share with bridge so master sees it
    bridge.shared_state["log"] = mission_status["log"][:]

MOVE_CMDS = {
    "up": tello.move_up, "down": tello.move_down,
    "forward": tello.move_forward, "back": tello.move_back,
    "left": tello.move_left, "right": tello.move_right,
}
ROTATE_CMDS = {
    "cw": tello.rotate_clockwise, "ccw": tello.rotate_counter_clockwise,
}
FLIP_DIRS = {"l", "r", "f", "b"}
ALL_TYPES = set(MOVE_CMDS) | set(ROTATE_CMDS) | {"flip", "wait", "takeoff", "land", "emergency"}

LABELS = {
    "takeoff":"Takeoff","land":"Land","up":"Up","down":"Down",
    "forward":"Forward","back":"Back","left":"Left","right":"Right",
    "cw":"Rotate CW","ccw":"Rotate CCW","flip":"Flip","wait":"Wait","emergency":"EMERGENCY",
}

def step_desc(step):
    t = step.get("type"); v = step.get("value")
    label = LABELS.get(t, t)
    if t in ("takeoff","land"): return label
    if t == "flip":  return f"{label} {v}"
    if t == "wait":  return f"{label} {v}s"
    unit = "deg" if t in ROTATE_CMDS else "cm"
    return f"{label} {v}{unit}"

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
    lr, fb, ud, yaw = RC_AXES[direction]
    s = RC_SPEED; dur = amount / max(1, per_sec)
    log(f"  rc-fallback {direction} ~{amount} (~{dur:.1f}s)")
    end = time.time() + dur
    while time.time() < end:
        if mission_status.get("cancel"): return
        tello.send_rc_control(lr*s, fb*s, ud*s, yaw*s)
        time.sleep(0.05)
    tello.send_rc_control(0, 0, 0, 0)

def _do_with_retry(direction, amount, per_sec):
    precise = MOVE_CMDS.get(direction) or ROTATE_CMDS.get(direction)
    for attempt in range(MOVE_RETRIES + 1):
        try:
            precise(amount); return
        except Exception as e:
            if mission_status.get("cancel"): raise
            if attempt < MOVE_RETRIES:
                log(f"  {direction} {amount} failed: {e} — settle {RETRY_SETTLE}s & retry")
                time.sleep(RETRY_SETTLE)
            elif USE_RC_FALLBACK:
                log(f"  rc fallback for {direction} {amount}")
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
    try:
        tello.send_command_without_return(stop_cmd)
        log(f"!! Operator override: {stop_cmd.upper()}")
    except Exception as e:
        log(f"abort send failed: {e}")

def run_mission(steps):
    if not mission_lock.acquire(blocking=False):
        log("Mission already running."); return
    try:
        mission_status.update({"running": True, "cancel": False, "log": [], "current": -1})
        bridge.shared_state["running"] = True
        log(f"Mission start — {len(steps)} step(s)")
        try:
            batt = tello.get_battery()
            log(f"Battery {batt}%")
            if batt < MIN_BATTERY:
                log(f"Battery < {MIN_BATTERY}% — aborting."); return
        except Exception as e:
            log(f"battery check failed: {e}")
        try: tello.set_speed(MISSION_SPEED)
        except: pass

        for i, step in enumerate(steps):
            if mission_status["cancel"]: log("Mission aborted."); break
            mission_status["current"] = i
            bridge.shared_state["current"] = i
            log(f"[{i+1}/{len(steps)}] {step_desc(step)}")
            exec_step(step)
            if step.get("type") == "takeoff":
                log(f"stabilizing {STABILIZE_AFTER_TAKEOFF}s...")
                time.sleep(STABILIZE_AFTER_TAKEOFF)
            if mission_status["cancel"]: log("Mission aborted."); break
            time.sleep(0.3)
        else:
            log("Mission complete.")
    except Exception as e:
        log(f"Mission error: {e}")
        if not mission_status["cancel"]:
            try: tello.land(); log("Auto-land after error.")
            except: pass
    finally:
        mission_status.update({"current": -1, "running": False, "cancel": False})
        bridge.shared_state.update({"running": False, "current": -1})
        mission_lock.release()

# ---- Stats loop (feeds bridge) --------------------------------------------
def stats_loop():
    while True:
        try:
            bridge.shared_state["stats"] = {
                "battery":     tello.get_battery(),
                "height":      tello.get_height(),
                "temp":        tello.get_temperature(),
                "speed_x":     tello.get_speed_x(),
                "speed_y":     tello.get_speed_y(),
                "speed_z":     tello.get_speed_z(),
                "pitch":       tello.get_pitch(),
                "roll":        tello.get_roll(),
                "yaw":         tello.get_yaw(),
                "flight_time": tello.get_flight_time(),
                "barometer":   tello.get_barometer(),
            }
        except: pass
        time.sleep(1)

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
        except av.error.InvalidDataError:
            time.sleep(0.05)
        except Exception as e:
            print(f"Frame error: {e}"); time.sleep(0.05)

# ---- Routes ---------------------------------------------------------------
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(bridge.shared_state.get("stats", {}))

@app.route('/record/toggle', methods=['POST'])
def record_toggle():
    with record_state["lock"]:
        currently = record_state["active"]
    if currently:
        stop_recording()
        return jsonify({"recording": False})
    else:
        frame = frame_reader.frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with record_state["lock"]:
            start_recording(rgb)
        return jsonify({"recording": True, "file": record_state["filename"]})

@app.route('/record/status')
def record_status():
    return jsonify({"recording": record_state["active"],
                    "filename": record_state["filename"] or ""})

@app.route('/mission/start', methods=['POST'])
def start_mission():
    if mission_status["running"]:
        return jsonify({"status": "busy"}), 409
    data  = request.get_json(silent=True) or {}
    steps = data.get("steps", [])
    err   = validate_steps(steps)
    if err: return jsonify({"status": "error", "msg": err}), 400
    threading.Thread(target=run_mission, args=(steps,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/mission/abort', methods=['POST'])
def abort_route():
    if not mission_status["running"]:
        return jsonify({"status": "idle"})
    abort_mission("land")
    return jsonify({"status": "aborting"})

@app.route('/mission/status')
def mission_status_route():
    return jsonify({"running":  mission_status["running"],
                    "log":      mission_status["log"],
                    "current":  mission_status["current"]})

@app.route('/control/<cmd>', methods=['POST'])
def control(cmd):
    if cmd not in ALL_TYPES:
        return jsonify({"status": "error", "msg": f"unknown cmd '{cmd}'"}), 400
    if cmd in ("land", "emergency"):
        if mission_status["running"]: abort_mission(cmd)
        else:
            try: exec_step({"type": cmd})
            except Exception as e: return jsonify({"status": "error", "msg": str(e)}), 500
        return jsonify({"status": "ok"})
    if mission_status["running"]:
        return jsonify({"status": "busy"}), 409
    v    = request.args.get("value")
    step = {"type": cmd}
    if cmd in MOVE_CMDS:   step["value"] = v or 30
    elif cmd in ROTATE_CMDS: step["value"] = v or 45
    elif cmd == "flip":    step["value"] = v or "f"
    try: exec_step(step); return jsonify({"status": "ok"})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/')
def index():
    return open("slave_ui.html").read()

# ---- Boot -----------------------------------------------------------------
if __name__ == '__main__':
    bridge.start()                                          # start ethernet bridge
    threading.Thread(target=stats_loop, daemon=True).start()
    print("Slave dashboard at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)