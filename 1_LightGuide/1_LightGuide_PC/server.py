import socket
import serial
import struct
from quaternion import quaternion
import multiprocessing
import os
from experiment import *

x_axis = quaternion(0,1,0,0)
y_axis = quaternion(0,0,1,0)
z_axis = quaternion(0,0,0,1)


class Optitrack:
    def __init__(self):
        super().__init__()
        self.OptitrackData = [0, 0, 0]
        self.OptitrackOffset = [0, 0, 0]
        self.IsCalibrated = False
        self.IsFinished = False
        self.path = ''


class OptitrackThread(threading.Thread):
    def __init__(self, s, opti):
        super().__init__()
        self.s = s
        self.opti = opti
        self.calibrationTemp = deque(maxlen=10)
        self.calibrationProgress = 0

    def setOptitrackData(self, position, angle):
        self.opti.OptitrackData[0] = math.floor(position[0]) - self.opti.OptitrackOffset[0]  # x mm
        self.opti.OptitrackData[1] = -math.floor(position[1]) - self.opti.OptitrackOffset[1]  # z mm
        self.opti.OptitrackData[2] = angle - self.opti.OptitrackOffset[2]  # degree
        self.opti.OptitrackData[2] = self.opti.OptitrackData[2] % 360  # [0, 360)
        if self.opti.OptitrackData[2] > 180: self.opti.OptitrackData[2] -= 360  # (-180, 180]

    def run(self):
        if not self.opti.IsCalibrated:
            print("Start calibration, press CTRL to continue...")
            playsound('verbal/12.mp3')  # 开始校准，保持站定

        while True:
            try:
                data, _ = self.s.recvfrom(120)  # 从optitrack接收数据，80Hz
                if len(data) > 0:
                    data_list = struct.unpack('>Iddddddd', data)
                    # [id, x, y, z, r0, r1, r2, r3]
                    position = np.array([data_list[1], data_list[3]]) * 1000
                    # 位置修正？平移、旋转
                    rotation = quaternion(data_list[-1], data_list[-4], data_list[-3], data_list[-2])
                    new_vec = ((rotation * y_axis) * (rotation.conjugate())).to_list()[1:]
                    vec1 = np.array([new_vec[0], new_vec[2]])
                    vec2 = np.array([0, 1])
                    angle_to_vec2 = math.atan2(vec2[0], vec2[1]) - math.atan2(vec1[0], vec1[1])
                    angle = math.floor(angle_to_vec2 / math.pi * 180) - 180
                    if angle <= -180: 
                        angle += 360  # (-180,180]
                    height = int(data_list[2] * 1000)

                    self.setOptitrackData(position, angle)

                    if not self.opti.IsCalibrated:  # 如果未校准，先校准
                        print("\r(x, z, theta)", self.opti.OptitrackData, end=' ')
                        self.calibrationProgress += 1
                        self.calibrationTemp.append(self.opti.OptitrackData)
                        if keyboard.is_pressed('ctrl'):  # or self.calibrationProgress >= 120 * 8:  # 按→完成校准，或5s之后
                            temp = np.mean(np.array(self.calibrationTemp).T, axis=1)
                            self.opti.OptitrackOffset[0] = int(temp[0])
                            self.opti.OptitrackOffset[1] = int(temp[1])
                            self.opti.OptitrackOffset[2] = int(temp[2])
                            self.setOptitrackData(position, angle)
                            if self.opti.path == 'S_r3.2':
                                self.opti.OptitrackOffset[2] -= 45
                            print('')
                            print("Set offset:", self.opti.OptitrackOffset)
                            playsound('verbal/13.mp3')  # 校准完成
                            self.opti.IsCalibrated = True

                    else:  # 如果已校准，则再按↓结束进程
                        # print(self.opti.OptitrackData)
                        if keyboard.is_pressed('down'): # or height < 800:
                            print("Task kill!")
                            playsound('verbal/16.mp3')  # 实验结束
                            self.opti.IsFinished = True
                            break
                        elif self.opti.IsFinished:
                            playsound('verbal/16.mp3')  # 实验结束
                            self.opti.IsFinished = True
                            break

            except:
                print("Optitrack thread error")
                break


class ClientThread(threading.Thread):
    def __init__(self, serial, opti):
        super().__init__()
        self.serial = serial
        self.opti = opti

        self.taskIndex = ''
        self.task = None

        self.pathName = ''
        self.userName = ''
        self.logUserTaskDir = ''
        self.createTime = ''

        self.verbalOn = False
        self.loadTask()

    def loadTask(self):
        self.userName = input("\nInput user name:")  # 输入被试姓名
        logUserDir = 'experiment/'+self.userName  # 创建被试路径
        if os.path.exists(logUserDir):
            print("Load user dir:", logUserDir)
        else:
            os.mkdir(logUserDir)
            print("Create user dir:", logUserDir)
        self.taskIndex = str(input("\nInput task index:"))
        self.createTime = str(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()))
        print("Create task", self.taskIndex, "at", self.createTime)

        if self.taskIndex == '1':
            self.logUserTaskDir = 'experiment/' + self.userName + '/task1'  # 创建被试实验路径
            if not os.path.exists(self.logUserTaskDir): os.mkdir(self.logUserTaskDir)
            self.task = Task1(self.pathName, self.logUserTaskDir, self.createTime, 0)
        if self.taskIndex == '2':
            self.logUserTaskDir = 'experiment/' + self.userName + '/task2'  # 创建被试实验路径
            if not os.path.exists(self.logUserTaskDir): os.mkdir(self.logUserTaskDir)
            self.task = Task2(self.pathName, self.logUserTaskDir, self.createTime, 0)
        elif self.taskIndex == '3':
            self.logUserTaskDir = 'experiment/'+self.userName+'/task3'  # 创建被试实验路径
            if not os.path.exists(self.logUserTaskDir): os.mkdir(self.logUserTaskDir)
            print("\n"
                  "#  Straight : (I)line_l9\n"
                  "# Right turn: (Rs)R_45 (Rr)R_90 (Rg)R_135\n"
                  "# Left  turn: (Ls)L_45 (Lr)L_90 (Lg)L_135\n"
                  "#   Zigzag  : (S)S_r3.2\n"
                  "#  Training : (T)learn"
                  "# NOTE: Path code is the code in ()")
            pathCode = str(input("Enter path code:")).lower()
            if pathCode == "i":
                self.pathName = "line_l9"
            elif pathCode == "rs":
                self.pathName = "L_45"
            elif pathCode == "rr":
                self.pathName = "L_90"
            elif pathCode == "rg":
                self.pathName = "L_135"
            elif pathCode == "ls":
                self.pathName = "L_-45"
            elif pathCode == "lr":
                self.pathName = "L_-90"
            elif pathCode == "lg":
                self.pathName = "L_-135"
            elif pathCode == "s":
                self.pathName = "S_r3"
            elif pathCode == "t":
                self.pathName = "learn"
            else:
                print("Invalid path code!")
            self.opti.path = self.pathName
            self.verbalOn = 0 #int(input("Verbal instruction 0/1:"))
            self.task = Task3(self.pathName, self.logUserTaskDir, self.createTime, self.verbalOn)

    def run(self):
        self.task.calculateInitiate()  # 任务初始化
        while True:
            try:
                commandDirection = 180
                vibrationIntensity = 0

                if self.opti.IsCalibrated:  # 校准后启用cal，否则发送0
                    self.task.begin()  # 等待设置校准，任务开始
                    commandDirection, vibrationIntensity = self.task.calculate(self.opti.OptitrackData)
                    if commandDirection == 0 and vibrationIntensity < 0:  # 任务结束
                        self.opti.IsFinished = True
                    if commandDirection == 1000 and vibrationIntensity == 0:  # 任务暂停
                        commandDirection = 220
                    commandDirection += 180


                if self.opti.IsFinished:
                    # print("send data " + str(45))
                    self.serial.write(bytes([45]))
                    self.serial.flush()
                    head_process = multiprocessing.Process(target=show,
                                                           args=(self.taskIndex, self.pathName, self.logUserTaskDir, self.createTime, self.verbalOn))
                    head_process.start()
                    break

                commandDirection = int(( commandDirection + 2 ) / 4)

                # print("send data " + str(commandDirection))
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

    def connect(self):
        self.showHostInfo()
        self.optiSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.optiSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 120)
        self.optiSocket.bind(('169.254.49.108', 10010))


if __name__ == "__main__":
    otDevice = OptiTrackDevice()
    otDevice.connect()

    optitrack = Optitrack()
    optitrackTread = OptitrackThread(otDevice.optiSocket, optitrack)

    lightDevice = serial.Serial("COM5", 9600)
    clientTread = ClientThread(lightDevice, optitrack)  # 创建任务，任务输出
    clientTread.start()
    time.sleep(1)  # 等待任务初始化
    print("\nTrial begin...")
    time.sleep(0.5)  # 等待任务初始化

    optitrackTread.start()
    optitrackTread.join()
    clientTread.join()

