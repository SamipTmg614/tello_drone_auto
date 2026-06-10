import socket
import threading

TELLO_IP   = '192.168.10.1'
CMD_PORT   = 8889
STATE_PORT = 8890

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', CMD_PORT))

def send(cmd):
    sock.sendto(cmd.encode(), (TELLO_IP, CMD_PORT))
    response, _ = sock.recvfrom(1024)
    print(f'>> {cmd}  →  {response.decode()}')

def listen_state():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', STATE_PORT))
    while True:
        data, _ = s.recvfrom(1024)
        print('state:', data.decode())

# Start state listener in background
threading.Thread(target=listen_state, daemon=True).start()

send('command')       # enter SDK mode — must be first command
send('streamon')
import cv2
cap = cv2.VideoCapture('udp://0.0.0.0:11111')
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Tello', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break