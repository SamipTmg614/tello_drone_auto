from djitellopy import Tello
from flask import Flask, Response, jsonify
import cv2
import threading
import time

app = Flask(__name__)

tello = Tello()
tello.connect()
tello.streamon()
frame_reader = tello.get_frame_read()

# Mission state
mission_status = {"running": False, "log": []}

def log(msg):
    print(msg)
    mission_status["log"].append(msg)
    if len(mission_status["log"]) > 20:
        mission_status["log"].pop(0)

def run_mission():
    if mission_status["running"]:
        log("Mission already running.")
        return
    mission_status["running"] = True
    mission_status["log"] = []
    try:
        log("Takeoff...")
        tello.takeoff()
        time.sleep(3)

        log("Ascending to 500 cm (5 m)...")
        tello.move_up(500)
        time.sleep(4)

        log("Moving forward 500 cm (5 m)...")
        tello.move_forward(500)
        time.sleep(4)

        log("Landing...")
        tello.land()
        time.sleep(3)

        log("Mission complete.")
    except Exception as e:
        log(f"Mission error: {e}")
        try:
            tello.land()
        except:
            pass
    finally:
        mission_status["running"] = False

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
    t = threading.Thread(target=run_mission, daemon=True)
    t.start()
    return jsonify({"status": "started"})

@app.route('/mission/status')
def mission_status_route():
    return jsonify({
        "running": mission_status["running"],
        "log": mission_status["log"]
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

        .layout { display:grid; grid-template-columns:1fr 340px; gap:16px; }

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

        <div class="mission-panel">
            <h3>Auto Mission</h3>

            <div class="mission-steps">
                <div class="step"><div class="step-icon">🛫</div> Takeoff</div>
                <div class="step"><div class="step-icon">⬆️</div> Ascend 5 m</div>
                <div class="step"><div class="step-icon">➡️</div> Forward 5 m</div>
                <div class="step"><div class="step-icon">🛬</div> Land</div>
            </div>

            <button class="btn-mission" id="missionBtn" onclick="startMission()">▶ Run Mission</button>

            <div>
                <div style="font-size:11px;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.08em;">
                    <span class="status-dot" id="statusDot"></span>
                    <span id="statusText">Idle</span>
                </div>
                <div class="log-box" id="logBox">Waiting...</div>
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

        async function pollMission() {
            const res = await fetch('/mission/status');
            const d = await res.json();
            const btn  = document.getElementById('missionBtn');
            const dot  = document.getElementById('statusDot');
            const txt  = document.getElementById('statusText');
            const log  = document.getElementById('logBox');

            btn.disabled = d.running;
            btn.textContent = d.running ? '⏳ Running...' : '▶ Run Mission';
            dot.className  = 'status-dot' + (d.running ? ' running' : '');
            txt.textContent = d.running ? 'Running' : 'Idle';

            if (d.log.length) {
                log.innerHTML = d.log.map(l => '<div>' + l + '</div>').join('');
                log.scrollTop = log.scrollHeight;
            }
        }

        async function startMission() {
            await fetch('/mission/start', { method: 'POST' });
        }

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