import serial
import time

lightDevice = serial.Serial("COM5", 9600)
while True:
    for i in range(90):
        lightDevice.write(bytes([i]))
        lightDevice.flush()
        time.sleep(0.1)