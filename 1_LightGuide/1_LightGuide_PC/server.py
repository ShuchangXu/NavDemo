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
from Octavib import octaVib
from Handle import HandleDevice

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
                    new_vec = ((rotation * z_axis) * (rotation.conjugate())).to_list()[1:]
                    vec1 = np.array([new_vec[0], new_vec[2]])
                    vec2 = np.array([0, 1])
                    angle_to_vec2 = math.atan2(vec2[0], vec2[1]) - math.atan2(vec1[0], vec1[1])
                    angle = math.floor(angle_to_vec2 / math.pi * 180) - 180
                    if angle < 0: 
                        angle += 360  # [0,360)
                    angle = angle - 180 # [-180, 180)

                    self.setOptitrackData(position, angle)
            except:
                print("Optitrack thread error")
                break


class ClientThread(threading.Thread):
    def __init__(self, opti, hatSerial, backpack, handle):
        super().__init__()
        self.opti = opti
        self.hatSerial = hatSerial
        self.backpack = backpack
        self.handle = handle

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

                commandDirection, vibrationIntensity, mylogStr = self.task.calculate(self.opti.OptitrackData)
                if commandDirection == 0 and vibrationIntensity < 0:  # 任务结束
                    pass
                
                if self.backpack != None:
                    self.backpack.angle(commandDirection * 10)
                    print("user (", self.opti.OptitrackData[0], ", ", self.opti.OptitrackData[1], "), a = ", self.opti.OptitrackData[2], mylogStr, ", bag = ", str(commandDirection))

                if self.hatSerial != None:
                    lightDir = commandDirection + 180
                    lightDir = int(( lightDir + 2 ) / 4)
                    sendData = str(lightDir).zfill(3) +'x'
                    self.hatSerial.write(sendData.encode())  # 发送数据
                    self.hatSerial.flush()
                    print("user (", self.opti.OptitrackData[0], ", ", self.opti.OptitrackData[1], "), a = ", self.opti.OptitrackData[2], mylogStr, ", light = ", str(lightDir))
                
                if self.handle != None:
                    # data = self.handle.clientSocket.recv(3)
                    # sendData = str(commandDirection).zfill(3) + str(vibrationIntensity).zfill(3)
                    # self.handle.clientSocket.send(sendData.encode())
                    handleDir = commandDirection + 180 + 7  # 7-367
                    handleDir = int(handleDir) % 360

                    handleIntensity = int(vibrationIntensity)
                    sendData = str(handleDir).zfill(3) + str(handleIntensity).zfill(3) +'x'
                    self.handle.write(sendData.encode())
                    self.handle.flush()
                    print("user (", self.opti.OptitrackData[0], ", ", self.opti.OptitrackData[1], "), a = ", self.opti.OptitrackData[2], mylogStr, ", handle = ", str(handleDir))


                time.sleep(0.03)

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
    optiData = Optitrack()

    deviceType = input("\nInput device type:")

    backpackDevice = None
    lightDevice = None
    handleDevice = None

    if deviceType == "l":
        # light Guide
        lightDevice = serial.Serial("/dev/ttyUSB0", 9600)
    elif deviceType == "b":
        # backpack vibration
        backpackDevice = octaVib("/dev/ttyUSB0")
        backpackDevice.set_start(2000,15)
        backpackDevice.set_stop(2000,160)
        backpackDevice.set_angle_mapping(20,20,600,999)
        backpackDevice.set_distance_mapping(1500,20,700)
        backpackDevice.set_gamma(2.0)
    elif deviceType == "h":
        # handle Device
        handleDevice = serial.Serial("/dev/ttyUSB0", 9600)
        # handleDevice = HandleDevice()
        # handleDevice.connect()

    clientTread = ClientThread(optiData, lightDevice, backpackDevice, handleDevice)

    otDevice = OptiTrackDevice()
    otDevice.connect('127.0.0.1', 10092)
    optitrackThread = OptitrackThread(otDevice.optiSocket, optiData, otDevice.connection)

    clientTread.start()
    time.sleep(1)  # 等待任务初始化

    optitrackThread.start()
    optitrackThread.join()
    clientTread.join()