from djitellopy import Tello
from flask import Flask, Response, jsonify, request
import cv2
import threading
import time

app = Flask(__name__)

tello = Tello()
tello.connect()
tello.streamon()
frame_reader = tello.get_frame_read()

# ---- Mission engine -------------------------------------------------------
mission_status = {"running": False, "log": [], "current": -1, "cancel": False}
mission_lock = threading.Lock()

# Let the drone settle after takeoff before the next move so the climb is stable.
# Firing a move ~0.3s after takeoff (before position hold locks) is what makes it
# lurch/drift backward instead of climbing cleanly.
STABILIZE_AFTER_TAKEOFF = 4      # seconds
MISSION_SPEED = 40               # cm/s for distance moves — gentler = more predictable

def log(msg):
    print(msg)
    mission_status["log"].append(msg)
    if len(mission_status["log"]) > 40:
        mission_status["log"].pop(0)

# command tables: map a step "type" to the drone action
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
    "takeoff": "Takeoff", "land": "Land", "up": "Up", "down": "Down",
    "forward": "Forward", "back": "Back", "left": "Left", "right": "Right",
    "cw": "Rotate CW", "ccw": "Rotate CCW", "flip": "Flip", "wait": "Wait",
    "emergency": "EMERGENCY",
}

def step_desc(step):
    t = step.get("type")
    v = step.get("value")
    label = LABELS.get(t, t)
    if t in ("takeoff", "land"):
        return label
    if t == "flip":
        return f"{label} {v}"
    if t == "wait":
        return f"{label} {v}s"
    unit = "deg" if t in ROTATE_CMDS else "cm"
    return f"{label} {v}{unit}"

def validate_steps(steps):
    if not isinstance(steps, list) or not steps:
        return "no steps"
    for i, s in enumerate(steps):
        t = s.get("type") if isinstance(s, dict) else None
        if t not in ALL_TYPES:
            return f"step {i + 1}: unknown type '{t}'"
        if t in MOVE_CMDS or t in ROTATE_CMDS or t == "wait":
            try:
                float(s.get("value"))
            except (TypeError, ValueError):
                return f"step {i + 1}: '{t}' needs a number"
        if t == "flip" and s.get("value") not in FLIP_DIRS:
            return f"step {i + 1}: flip needs l/r/f/b"
    return None

def exec_step(step):
    t = step.get("type")
    v = step.get("value")
    if t == "takeoff":
        tello.takeoff()
    elif t == "land":
        tello.land()
    elif t == "emergency":
        tello.emergency()          # cuts motors immediately — drone drops
    elif t in MOVE_CMDS:
        MOVE_CMDS[t](max(20, min(500, int(float(v)))))    # Tello range 20-500 cm
    elif t in ROTATE_CMDS:
        ROTATE_CMDS[t](max(1, min(360, int(float(v)))))   # 1-360 deg
    elif t == "flip":
        if v not in FLIP_DIRS:
            raise ValueError(f"bad flip dir '{v}'")
        tello.flip(v)
    elif t == "wait":
        time.sleep(float(v))
    else:
        raise ValueError(f"unknown step '{t}'")

def abort_mission(stop_cmd="land"):
    """Signal a running mission to stop and immediately fire a stop command to
    the drone WITHOUT waiting for a response — so it works even while the mission
    thread is blocked mid-move. Backs the emergency Land button."""
    mission_status["cancel"] = True
    try:
        tello.send_command_without_return(stop_cmd)   # 'land' (controlled) or 'emergency' (motor cut)
        log(f"!! Operator override: {stop_cmd.upper()}")
    except Exception as e:
        log(f"abort send failed: {e}")

def run_mission(steps):
    if not mission_lock.acquire(blocking=False):
        log("Mission already running.")
        return
    try:
        mission_status["running"] = True
        mission_status["cancel"] = False
        mission_status["log"] = []
        mission_status["current"] = -1
        log(f"Mission start — {len(steps)} step(s)")
        try:
            tello.set_speed(MISSION_SPEED)            # gentle, predictable moves
        except Exception:
            pass
        for i, step in enumerate(steps):
            if mission_status["cancel"]:
                log("Mission aborted by operator.")
                break
            mission_status["current"] = i
            log(f"[{i + 1}/{len(steps)}] {step_desc(step)}")
            exec_step(step)
            # settle after takeoff so the next move (the climb) is stable, not a lurch
            if step.get("type") == "takeoff":
                log(f"stabilizing {STABILIZE_AFTER_TAKEOFF}s...")
                time.sleep(STABILIZE_AFTER_TAKEOFF)
            if mission_status["cancel"]:
                log("Mission aborted by operator.")
                break
            time.sleep(0.3)
        else:
            log("Mission complete.")
    except Exception as e:
        log(f"Mission error: {e}")
        # don't fight an operator stop — only auto-land if no abort was requested
        if not mission_status["cancel"]:
            try:
                tello.land()
                log("Auto-land after error.")
            except Exception:
                pass
    finally:
        mission_status["current"] = -1
        mission_status["running"] = False
        mission_status["cancel"] = False
        mission_lock.release()

def gen_frames():
    while True:
        frame = frame_reader.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/mission/start', methods=['POST'])
def start_mission():
    if mission_status["running"]:
        return jsonify({"status": "busy", "msg": "mission already running"}), 409
    data = request.get_json(silent=True) or {}
    steps = data.get("steps", [])
    err = validate_steps(steps)
    if err:
        return jsonify({"status": "error", "msg": err}), 400
    threading.Thread(target=run_mission, args=(steps,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/mission/abort', methods=['POST'])
def abort_route():
    if not mission_status["running"]:
        return jsonify({"status": "idle", "msg": "no mission running"})
    abort_mission("land")
    return jsonify({"status": "aborting"})

@app.route('/control/<cmd>', methods=['POST'])
def control(cmd):
    if cmd not in ALL_TYPES:
        return jsonify({"status": "error", "msg": f"unknown cmd '{cmd}'"}), 400
    # Safety override: Land / Emergency may interrupt a running mission at ANY time.
    if cmd in ("land", "emergency"):
        if mission_status["running"]:
            abort_mission(cmd)
            return jsonify({"status": "aborting", "msg": f"{cmd} sent — mission aborting"})
        try:
            exec_step({"type": cmd})                # not in a mission: normal blocking land
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "msg": str(e)}), 500
    # Every other manual command is blocked while a mission is running.
    if mission_status["running"]:
        return jsonify({"status": "busy", "msg": "mission running — only Land/Emergency allowed"}), 409
    v = request.args.get("value")
    step = {"type": cmd}
    if cmd in MOVE_CMDS:
        step["value"] = v if v is not None else 30
    elif cmd in ROTATE_CMDS:
        step["value"] = v if v is not None else 45
    elif cmd == "flip":
        step["value"] = v or "f"
    try:
        exec_step(step)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/mission/status')
def mission_status_route():
    return jsonify({
        "running": mission_status["running"],
        "log": mission_status["log"],
        "current": mission_status["current"],
    })

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Tello Dashboard</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { background:#0d0d0d; color:#e0e0e0; font-family:'Segoe UI',sans-serif; padding:20px; }

        h2 { font-size:18px; font-weight:500; letter-spacing:0.05em; color:#aaa; margin-bottom:16px; text-transform:uppercase; }

        .grid {
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
            gap:10px;
            margin-bottom:20px;
        }
        .card {
            background:#161616;
            border:1px solid #222;
            border-radius:10px;
            padding:14px 16px;
        }
        .card .label { font-size:11px; color:#555; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.08em; }
        .card .value { font-size:24px; font-weight:600; color:#fff; }
        .card .unit  { font-size:12px; color:#555; margin-left:3px; }

        .layout { display:grid; grid-template-columns:1fr 380px; gap:16px; align-items:start; }

        img { width:100%; border-radius:10px; border:1px solid #222; display:block; }

        .mission-panel {
            background:#161616;
            border:1px solid #222;
            border-radius:10px;
            padding:18px;
            display:flex;
            flex-direction:column;
            gap:14px;
        }
        .mission-panel h3 { font-size:13px; color:#666; text-transform:uppercase; letter-spacing:0.08em; }

        .mission-steps {
            display:flex;
            flex-direction:column;
            gap:8px;
        }
        .step {
            display:flex;
            align-items:center;
            gap:10px;
            font-size:13px;
            color:#888;
        }
        .step-icon {
            width:28px; height:28px;
            border-radius:50%;
            background:#1e1e1e;
            border:1px solid #2a2a2a;
            display:flex; align-items:center; justify-content:center;
            font-size:14px;
            flex-shrink:0;
        }

        .btn-mission {
            background:#2563eb;
            color:#fff;
            border:none;
            border-radius:8px;
            padding:12px;
            font-size:14px;
            font-weight:600;
            cursor:pointer;
            transition:background 0.2s;
            width:100%;
        }
        .btn-mission:hover  { background:#1d4ed8; }
        .btn-mission:disabled { background:#1e3a6e; color:#555; cursor:not-allowed; }

        /* always-on safety controls */
        .btn-emergency {
            margin-top:10px; width:100%; padding:14px;
            background:#dc2626; color:#fff; border:none; border-radius:8px;
            font-size:15px; font-weight:700; letter-spacing:0.04em; cursor:pointer;
            transition:background .15s;
        }
        .btn-emergency:hover  { background:#b91c1c; }
        .btn-emergency:active { background:#991b1b; }
        .btn-cut {
            margin-top:6px; width:100%; padding:8px;
            background:#3a0d0d; color:#f87171; border:1px solid #7f1d1d;
            border-radius:8px; font-size:12px; font-weight:600; cursor:pointer;
        }
        .btn-cut:hover { background:#511414; }

        .log-box {
            background:#0d0d0d;
            border:1px solid #1a1a1a;
            border-radius:8px;
            padding:10px 12px;
            font-size:12px;
            font-family:monospace;
            color:#4ade80;
            min-height:80px;
            max-height:140px;
            overflow-y:auto;
            line-height:1.7;
        }

        .status-dot {
            display:inline-block;
            width:8px; height:8px;
            border-radius:50%;
            background:#22c55e;
            margin-right:6px;
            vertical-align:middle;
        }
        .status-dot.running { background:#f59e0b; animation:pulse 1s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

        /* side column: stacked panels */
        .side-col { display:flex; flex-direction:column; gap:16px; }
        .panel { background:#161616; border:1px solid #222; border-radius:10px; padding:16px; }
        .panel h3 { font-size:13px; color:#666; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px; }

        /* generic small button */
        .btn { border:none; border-radius:6px; padding:8px 10px; font-size:13px; font-weight:600; cursor:pointer; color:#fff; background:#333; transition:background .15s; }
        .btn:hover { background:#444; }
        .btn-add { background:#374151; }
        .btn-add:hover { background:#4b5563; }

        /* manual control pad */
        .pad { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
        .pad .btn { padding:11px 0; background:#1e1e1e; }
        .pad .btn:hover { background:#2a2a2a; }
        .pad .btn.takeoff { background:#166534; }
        .pad .btn.takeoff:hover { background:#15803d; }
        .pad .btn.land { background:#7f1d1d; }
        .pad .btn.land:hover { background:#991b1b; }
        .pad .spacer { visibility:hidden; }
        .manual-val { display:flex; align-items:center; gap:8px; margin-top:10px; font-size:12px; color:#666; }
        .manual-val input { width:60px; background:#0d0d0d; border:1px solid #2a2a2a; color:#e0e0e0; border-radius:6px; padding:6px; }
        .kbd-hint { font-size:11px; color:#444; margin-top:10px; line-height:1.6; }

        /* mission builder */
        .builder-row { display:flex; gap:6px; margin-bottom:10px; }
        .builder-row select, .builder-row input {
            background:#0d0d0d; border:1px solid #2a2a2a; color:#e0e0e0;
            border-radius:6px; padding:8px; font-size:13px;
        }
        .builder-row select { flex:1; min-width:0; }
        .builder-row input { width:64px; }

        .steps-list { display:flex; flex-direction:column; gap:6px; margin-bottom:12px; max-height:240px; overflow-y:auto; }
        .step-item {
            display:flex; align-items:center; gap:8px;
            background:#0d0d0d; border:1px solid #1f1f1f; border-radius:6px;
            padding:7px 10px; font-size:13px; color:#ccc;
        }
        .step-item.active { border-color:#f59e0b; color:#fff; background:#1a1407; }
        .step-item .num { color:#555; font-size:11px; width:18px; flex-shrink:0; }
        .step-item .desc { flex:1; }
        .step-item .ico-btn { background:none; border:none; color:#666; cursor:pointer; font-size:13px; padding:2px 4px; }
        .step-item .ico-btn:hover { color:#fff; }
        .empty-hint { color:#444; font-size:12px; padding:8px 0; }
    </style>
</head>
<body>
    <h2>&#9679; Tello Live Dashboard</h2>

    <div class="grid">
        <div class="card"><div class="label">Battery</div><div class="value" id="battery">--<span class="unit">%</span></div></div>
        <div class="card"><div class="label">Height</div><div class="value" id="height">--<span class="unit">cm</span></div></div>
        <div class="card"><div class="label">Temp</div><div class="value" id="temp">--<span class="unit">°C</span></div></div>
        <div class="card"><div class="label">Pitch</div><div class="value" id="pitch">--<span class="unit">°</span></div></div>
        <div class="card"><div class="label">Roll</div><div class="value" id="roll">--<span class="unit">°</span></div></div>
        <div class="card"><div class="label">Yaw</div><div class="value" id="yaw">--<span class="unit">°</span></div></div>
        <div class="card"><div class="label">Flight Time</div><div class="value" id="flight_time">--<span class="unit">s</span></div></div>
        <div class="card"><div class="label">Barometer</div><div class="value" id="barometer">--<span class="unit">cm</span></div></div>
    </div>

    <div class="layout">
        <img src="/video">

        <div class="side-col">

            <div class="panel">
                <h3>Manual Control</h3>
                <div class="pad">
                    <button class="btn takeoff" onclick="ctrl('takeoff')">🛫 Takeoff</button>
                    <button class="btn" onclick="ctrl('up')">⬆ Up</button>
                    <button class="btn land" onclick="ctrl('land')">🛬 Land</button>

                    <button class="btn" onclick="ctrl('ccw')">⟲ CCW</button>
                    <button class="btn" onclick="ctrl('forward')">▲ Fwd</button>
                    <button class="btn" onclick="ctrl('cw')">⟳ CW</button>

                    <button class="btn" onclick="ctrl('left')">◀ Left</button>
                    <button class="btn" onclick="ctrl('back')">▼ Back</button>
                    <button class="btn" onclick="ctrl('right')">▶ Right</button>

                    <button class="btn" onclick="ctrl('flip','f')">⤿ Flip</button>
                    <button class="btn" onclick="ctrl('down')">⬇ Down</button>
                    <div class="spacer"></div>
                </div>
                <div class="manual-val">
                    move <input type="number" id="manualDist" value="30" min="20" max="500"> cm
                    · turn <input type="number" id="manualAng" value="45" min="1" max="360"> °
                </div>
                <div class="kbd-hint">keys: T takeoff · G land · W/S fwd/back · A/D left/right · R/F up/down · Q/E rotate</div>
            </div>

            <div class="panel">
                <h3>Mission Builder</h3>
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
                    <button class="btn btn-add" onclick="addStep()">+ Add</button>
                </div>

                <div class="steps-list" id="stepsList"></div>

                <button class="btn-mission" id="missionBtn" onclick="runMission()">▶ Run Mission</button>
                <div style="display:flex;gap:8px;margin-top:8px;">
                    <button class="btn" style="flex:1" onclick="loadSquare()">▢ Square (air)</button>
                    <button class="btn" style="flex:1" onclick="clearSteps()">✕ Clear</button>
                </div>
                <button class="btn-emergency" onclick="emergencyLand()">🛬 EMERGENCY LAND</button>
                <button class="btn-cut" onclick="emergencyCut()">⛔ Cut motors (drone drops)</button>

                <div style="margin-top:12px;">
                    <div style="font-size:11px;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.08em;">
                        <span class="status-dot" id="statusDot"></span>
                        <span id="statusText">Idle</span>
                    </div>
                    <div class="log-box" id="logBox">Waiting...</div>
                </div>
            </div>

        </div>
    </div>

    <script>
        async function updateStats() {
            const res = await fetch('/stats');
            const d = await res.json();
            document.getElementById('battery').innerHTML    = d.battery    + '<span class="unit">%</span>';
            document.getElementById('height').innerHTML     = d.height     + '<span class="unit">cm</span>';
            document.getElementById('temp').innerHTML       = d.temp       + '<span class="unit">°C</span>';
            document.getElementById('pitch').innerHTML      = d.pitch      + '<span class="unit">°</span>';
            document.getElementById('roll').innerHTML       = d.roll       + '<span class="unit">°</span>';
            document.getElementById('yaw').innerHTML        = d.yaw        + '<span class="unit">°</span>';
            document.getElementById('flight_time').innerHTML = d.flight_time + '<span class="unit">s</span>';
            document.getElementById('barometer').innerHTML  = d.barometer  + '<span class="unit">cm</span>';
        }

        // ---- manual control ----
        const MOVE = ['up','down','forward','back','left','right'];
        const ROT  = ['cw','ccw'];

        function ctrl(cmd, val) {
            let url = '/control/' + cmd;
            if (val !== undefined)        url += '?value=' + encodeURIComponent(val);
            else if (MOVE.includes(cmd))  url += '?value=' + document.getElementById('manualDist').value;
            else if (ROT.includes(cmd))   url += '?value=' + document.getElementById('manualAng').value;
            fetch(url, { method: 'POST' }).catch(() => {});
        }

        const KEYMAP = { t:'takeoff', g:'land', w:'forward', s:'back',
                         a:'left', d:'right', r:'up', f:'down', q:'ccw', e:'cw' };
        document.addEventListener('keydown', (ev) => {
            const tag = ev.target.tagName;
            if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
            const cmd = KEYMAP[ev.key.toLowerCase()];
            if (cmd) { ev.preventDefault(); ctrl(cmd); }
        });

        // ---- mission builder ----
        let steps = [];
        const TYPE_META = {
            takeoff:{v:false}, land:{v:false},
            up:{v:true,def:50}, down:{v:true,def:50},
            forward:{v:true,def:50}, back:{v:true,def:50},
            left:{v:true,def:50}, right:{v:true,def:50},
            cw:{v:true,def:90}, ccw:{v:true,def:90},
            flip:{v:true,def:'f'}, wait:{v:true,def:2},
        };
        const LBL = { takeoff:'Takeoff',land:'Land',up:'Up',down:'Down',
            forward:'Forward',back:'Back',left:'Left',right:'Right',
            cw:'Rotate CW',ccw:'Rotate CCW',flip:'Flip',wait:'Wait' };

        function onTypeChange() {
            const t = document.getElementById('cmdType').value;
            const inp = document.getElementById('cmdVal');
            const m = TYPE_META[t];
            if (m.v) { inp.style.visibility = 'visible'; inp.value = m.def; }
            else     { inp.style.visibility = 'hidden';  inp.value = ''; }
        }

        function descOf(s) {
            const n = LBL[s.type] || s.type;
            if (s.type === 'takeoff' || s.type === 'land') return n;
            if (s.type === 'flip') return n + ' ' + s.value;
            if (s.type === 'wait') return n + ' ' + s.value + 's';
            const unit = ROT.includes(s.type) ? '°' : 'cm';
            return n + ' ' + s.value + unit;
        }

        function renderSteps(current) {
            const box = document.getElementById('stepsList');
            if (!steps.length) {
                box.innerHTML = '<div class="empty-hint">No steps. Add commands above.</div>';
                return;
            }
            box.innerHTML = steps.map((s, i) =>
                '<div class="step-item ' + (i === current ? 'active' : '') + '">' +
                    '<span class="num">' + (i + 1) + '</span>' +
                    '<span class="desc">' + descOf(s) + '</span>' +
                    '<button class="ico-btn" onclick="moveStep(' + i + ',-1)">▲</button>' +
                    '<button class="ico-btn" onclick="moveStep(' + i + ',1)">▼</button>' +
                    '<button class="ico-btn" onclick="removeStep(' + i + ')">✕</button>' +
                '</div>').join('');
        }

        function addStep() {
            const t = document.getElementById('cmdType').value;
            const m = TYPE_META[t];
            const step = { type: t };
            if (m.v) {
                const val = document.getElementById('cmdVal').value.trim();
                if (t === 'flip') {
                    if (!['l','r','f','b'].includes(val)) { alert('flip needs l/r/f/b'); return; }
                    step.value = val;
                } else {
                    if (val === '' || isNaN(val)) { alert('enter a number'); return; }
                    step.value = Number(val);
                }
            }
            steps.push(step);
            renderSteps(-1);
        }

        function removeStep(i) { steps.splice(i, 1); renderSteps(-1); }
        function moveStep(i, d) {
            const j = i + d;
            if (j < 0 || j >= steps.length) return;
            [steps[i], steps[j]] = [steps[j], steps[i]];
            renderSteps(-1);
        }
        function clearSteps() { steps = []; renderSteps(-1); }

        // Auto mission: takeoff → climb 200cm → trace a 200cm square (turning left) → descend → land
        function loadSquare() {
            steps = [ {type:'takeoff'}, {type:'up', value:200} ];
            for (let i = 0; i < 4; i++) {
                steps.push({type:'forward', value:200});
                if (i < 3) steps.push({type:'ccw', value:90});   // turn left at the first 3 corners
            }
            steps.push({type:'down', value:200});
            steps.push({type:'land'});
            renderSteps(-1);
        }

        async function runMission() {
            if (!steps.length) { alert('add steps first'); return; }
            const res = await fetch('/mission/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps })
            });
            if (!res.ok) {
                const d = await res.json().catch(() => ({}));
                alert(d.msg || 'mission error');
            }
        }

        async function pollMission() {
            const res = await fetch('/mission/status');
            const d = await res.json();
            const btn = document.getElementById('missionBtn');
            const dot = document.getElementById('statusDot');
            const txt = document.getElementById('statusText');
            const log = document.getElementById('logBox');

            btn.disabled = d.running;
            btn.textContent = d.running ? '⏳ Running...' : '▶ Run Mission';
            dot.className = 'status-dot' + (d.running ? ' running' : '');
            txt.textContent = d.running ? 'Running' : 'Idle';

            renderSteps(d.running ? d.current : -1);

            if (d.log && d.log.length) {
                log.innerHTML = d.log.map(l => '<div>' + l + '</div>').join('');
                log.scrollTop = log.scrollHeight;
            }
        }

        onTypeChange();
        renderSteps(-1);
        setInterval(updateStats, 1000);
        setInterval(pollMission, 800);
        updateStats();
        pollMission();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print('Dashboard at: http://10.197.206.2:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)