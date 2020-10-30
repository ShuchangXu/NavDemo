import os
import time
import math
import threading
import socket
import struct
import numpy as np
import serial
from quaternion import quaternion
from experiment import DemoTask

class Optitrack:
    def __init__(self):
        super().__init__()
        self.OptitrackData = [0, 0, 0]

class ClientThread(threading.Thread):
    def __init__(self, serial, opti, backpack, handle):
        super().__init__()
        self.serial = serial
        self.opti = opti
        self.backpack = backpack
        self.handle = handle
        self.c = 180

    def run(self):
        # self.task.calculateInitiate()  # 任务初始化
        while True:
            try:
                print("1")
                self.c = self.c + 1
                if self.c > 270 : self.c = 90
                commandDirection = self.c
                vibrationIntensity = 0
                print("2")
                sendData = str(commandDirection).zfill(3) + str(vibrationIntensity).zfill(3) +'x'  # 三位是角度，三位是振动强度
                print("3")
                self.handle.write(sendData.encode())  # 发送数据
                print("4")
                self.handle.flush()
                print("5")
                time.sleep(0.02)

            except:
                print("Client thread error")
                break


class OptiTrackDevice:
    def __init__(self):
        super().__init__()
        self.HOST, self.PORT = '', 8080  # 域名和端口号
        self.optiSocket = socket.socket()

    def showHostInfo(self):
        hostname = socket.gethostname()
        result = socket.getaddrinfo(hostname, None, 0, socket.SOCK_STREAM)
        hostip = [x[4][0] for x in result]
        print('host ip:', hostip)
        print('host port:', self.PORT)

    def connect(self, ip, port):
        self.showHostInfo()
        self.optiSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.optiSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 120)
        self.optiSocket.bind((ip, port))
        self.optiSocket.listen(0)
        connection, address = self.optiSocket.accept()
        self.connection = connection


# class HandleDevice:
#     def __init__(self):
#         super().__init__()
#         self.HOST, self.PORT = '', 8080  # 域名和端口号
#         self.clientSocket = socket.socket()

#     def showHostInfo(self):
#         hostname = socket.gethostname()
#         result = socket.getaddrinfo(hostname, None, 0, socket.SOCK_STREAM)
#         hostip = [x[4][0] for x in result]
#         print('host ip:', hostip)
#         print('host port:', self.PORT)

#     def connect(self):
#         self.showHostInfo()
#         # self.optiSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         # self.optiSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 120)
#         # self.optiSocket.bind(('169.254.109.99', 10010))

#         print("Please connect client...")
#         self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.clientSocket.bind((self.HOST, self.PORT))
#         self.clientSocket.listen(5)
#         self.clientSocket, addr = self.clientSocket.accept()
#         print("Client is connected!")


if __name__ == "__main__":
    # backpack vibration
    optiData = Optitrack()

    # lightDevice = serial.Serial("/dev/ttyUSB1", 9600)
    lightDevice = ''
    device = ''

    handleDevice = serial.Serial("/dev/tty.usbserial-01E97196", 9600)

    clientTread = ClientThread(lightDevice, optiData, device, handleDevice)  # 创建任务，任务输出

    clientTread.start()

