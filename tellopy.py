from djitellopy import Tello
import cv2
import threading
import time

tello = Tello()
tello.connect()
tello.streamon()

frame_reader = tello.get_frame_read()

def fly():
    tello.takeoff()
    tello.move_up(50)
    # add slight right nudge to counteract left drift
    tello.send_rc_control(10, 0, 0, 0)  # push right slightly
    time.sleep(0.5)
    tello.send_rc_control(0, 0, 0, 0)  # stop
    tello.move_forward(10)
    tello.move_back(10)
    tello.land()
    
threading.Thread(target=fly, daemon=True).start()

while True:
    frame = frame_reader.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Tello', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()