import os
import time
import math
import threading
import socket
import struct
import serial
import numpy as np
from quaternion import quaternion
from experiment import DemoTask

x_axis = quaternion(0,1,0,0)
y_axis = quaternion(0,0,1,0)
z_axis = quaternion(0,0,0,1)

class Optitrack:
    def __init__(self):
        super().__init__()
        self.OptitrackData = [0, 0, 0]


class OptitrackThread(threading.Thread):
    def __init__(self, s, opti, connect):
        super().__init__()
        self.s = s
        self.opti = opti
        self.connect = connect

    def setOptitrackData(self, position, angle):
        self.opti.OptitrackData[0] = math.floor(position[0]) # x mm
        self.opti.OptitrackData[1] = math.floor(position[1])  # z mm
        self.opti.OptitrackData[2] = angle  # degree
        self.opti.OptitrackData[2] = self.opti.OptitrackData[2] % 360  # [0, 360)
        if self.opti.OptitrackData[2] > 180: self.opti.OptitrackData[2] -= 360  # (-180, 180]

    def run(self):
        while True:
            try:
                data = self.connect.recv(56)
                if len(data) > 0:
                    data_list = struct.unpack('ddddddd', data)
                    # [x, y, z, r0, r1, r2, r3]
                    position = np.array([data_list[0], data_list[2]]) * 1000
                    # 位置修正？平移、旋转
                    rotation = quaternion(data_list[-1], data_list[-4], data_list[-3], data_list[-2])
                    new_vec = ((rotation * y_axis) * (rotation.conjugate())).to_list()[1:]
                    vec1 = np.array([new_vec[0], new_vec[2]])
                    vec2 = np.array([0, 1])
                    angle_to_vec2 = math.atan2(vec2[0], vec2[1]) - math.atan2(vec1[0], vec1[1])
                    angle = math.floor(angle_to_vec2 / math.pi * 180) - 180
                    if angle <= -180: 
                        angle += 360  # (-180,180]

                    self.setOptitrackData(position, angle)
            except:
                print("Optitrack thread error")
                break


class ClientThread(threading.Thread):
    def __init__(self, serial, opti):
        super().__init__()
        self.serial = serial
        self.opti = opti

        self.pathName = ''
        self.logUserTaskDir = ''
        self.createTime = ''

        self.loadTask()

    def loadTask(self):
        # 创建被试文件夹
        logUserDir = 'experiment/demoUser'
        if os.path.exists(logUserDir):
            print("Load user dir:", logUserDir)
        else:
            os.mkdir(logUserDir)
            print("Create user dir:", logUserDir)
        
        # 创建被试实验文件夹
        self.pathName = 'demo'
        self.createTime = str(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()))
        self.logUserTaskDir = 'experiment/demoUser/'+ self.pathName  
        self.task = DemoTask(self.pathName, self.logUserTaskDir, self.createTime)
        print("Create Task", self.pathName, "at", self.createTime)

    def run(self):
        self.task.calculateInitiate()  # 任务初始化
        while True:
            try:
                commandDirection = 180
                vibrationIntensity = 0

                commandDirection, vibrationIntensity = self.task.calculate(self.opti.OptitrackData)
                if commandDirection == 0 and vibrationIntensity < 0:  # 任务结束
                    pass
                
                commandDirection += 180
                commandDirection = int(( commandDirection + 2 ) / 4)

                print("user (", self.opti.OptitrackData[0], ", ", self.opti.OptitrackData[1], "), a = ", self.opti.OptitrackData[2], "cmd = ", str(commandDirection))
                self.serial.write(bytes([commandDirection]))  # 发送数据
                self.serial.flush()
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


if __name__ == "__main__":
    otDevice = OptiTrackDevice()
    otDevice.connect('127.0.0.1', 10091)

    optiData = Optitrack()

    optitrackThread = OptitrackThread(otDevice.optiSocket, optiData, otDevice.connection)

    lightDevice = serial.Serial("/dev/ttyUSB0", 9600)
    clientTread = ClientThread(lightDevice, optiData)  # 创建任务，任务输出
    clientTread.start()
    time.sleep(1)  # 等待任务初始化

    optitrackThread.start()
    optitrackThread.join()
    clientTread.join()