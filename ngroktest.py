from djitellopy import Tello
from flask import Flask, render_template_string
from flask_sock import Sock
from pyngrok import ngrok
import cv2
import base64
import time
import json

app = Flask(__name__)
sock = Sock(app)

tello = Tello()
tello.connect()
tello.streamon()
frame_reader = tello.get_frame_read()

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>Tello Live</title>
            <style>
                body { margin:0; background:#000; display:flex; flex-direction:column; align-items:center; }
                img { width:100%; max-width:960px; }
                #stats { color:#fff; font-family:sans-serif; font-size:13px; padding:8px; display:flex; gap:16px; flex-wrap:wrap; }
                #stats span { background:#1e1e1e; padding:4px 10px; border-radius:6px; }
            </style>
        </head>
        <body>
            <div id="stats">
                <span>Battery: <b id="battery">--</b>%</span>
                <span>Height: <b id="height">--</b>cm</span>
                <span>Temp: <b id="temp">--</b>°C</span>
                <span>Yaw: <b id="yaw">--</b>°</span>
            </div>
            <img id="feed">
            <script>
                const ws = new WebSocket(`wss://${location.host}/stream`);
                ws.onmessage = (msg) => {
                    const data = JSON.parse(msg.data);
                    if (data.frame) {
                        document.getElementById('feed').src = 'data:image/jpeg;base64,' + data.frame;
                    }
                    if (data.stats) {
                        document.getElementById('battery').textContent = data.stats.battery;
                        document.getElementById('height').textContent = data.stats.height;
                        document.getElementById('temp').textContent = data.stats.temp;
                        document.getElementById('yaw').textContent = data.stats.yaw;
                    }
                };
            </script>
        </body>
        </html>
    ''')

@sock.route('/stream')
def stream(ws):
    frame_count = 0
    while True:
        frame = frame_reader.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        encoded = base64.b64encode(buffer).decode('utf-8')

        stats = None
        if frame_count % 10 == 0:
            stats = {
                'battery':  tello.get_battery(),
                'height':   tello.get_height(),
                'temp':     tello.get_temperature(),
                'yaw':      tello.get_yaw(),
            }

        ws.send(json.dumps({'frame': encoded, 'stats': stats}))
        frame_count += 1
        time.sleep(0.033)

@app.route('/command/<cmd>')
def command(cmd):
    commands = {
        'takeoff':  tello.takeoff,
        'land':     tello.land,
        'up':       lambda: tello.move_up(30),
        'down':     lambda: tello.move_down(30),
        'forward':  lambda: tello.move_forward(30),
        'back':     lambda: tello.move_back(30),
        'left':     lambda: tello.move_left(30),
        'right':    lambda: tello.move_right(30),
    }
    if cmd in commands:
        commands[cmd]()
        return 'ok'
    return 'unknown command', 400

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f'Public URL: {public_url}')
    app.run(host='0.0.0.0', port=5000, threaded=True)