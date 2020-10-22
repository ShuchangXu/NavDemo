import socket
import struct
from quaternion import quaternion
import threading
import keyboard
import winsound
import multiprocessing
from demo import *

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



class OptitrackThread(threading.Thread):
    def __init__(self, s, opti):
        super().__init__()
        self.s = s
        self.opti = opti
        self.calibrationTemp = deque(maxlen=10)
        self.calibrationProgress = 0
        self.logNameStr = 'log/trajRecord.txt'
        self.logfile = open(self.logNameStr, 'w')

    def run(self):
        while True:
            try:
                data, _ = self.s.recvfrom(120)  # 从optitrack接收数据，120Hz
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
                    if angle <= -180: angle += 360  # (-180,180]
                    height = int(data_list[2] * 1000)

                    self.opti.OptitrackData[0] = -math.floor(position[0]) - self.opti.OptitrackOffset[0]  # x mm
                    self.opti.OptitrackData[1] = math.floor(position[1]) - self.opti.OptitrackOffset[1]  # z mm
                    self.opti.OptitrackData[2] = angle - self.opti.OptitrackOffset[2]  # degree
                    self.opti.OptitrackData[2] = self.opti.OptitrackData[2] % 360  # [0, 360)
                    if self.opti.OptitrackData[2] > 180: self.opti.OptitrackData[2] -= 360  # (-180, 180]

                    if self.opti.IsCalibrated == False:  # 如果未校准，先校准
                        print("(x, z, theta)", self.opti.OptitrackData)
                        self.calibrationProgress += 1
                        self.calibrationTemp.append(self.opti.OptitrackData)
                        if keyboard.is_pressed('right'): #or self.calibrationProgress >= 120 * 10:  # 按→完成校准，或5s之后
                            winsound.Beep(600, 500)
                            temp = np.mean(np.array(self.calibrationTemp).T, axis=1)
                            self.opti.OptitrackOffset[0] = int(temp[0])
                            self.opti.OptitrackOffset[1] = int(temp[1])
                            self.opti.OptitrackOffset[2] = int(temp[2])
                            self.opti.OptitrackData[0] = -math.floor(position[0]) - self.opti.OptitrackOffset[0]  # x mm
                            self.opti.OptitrackData[1] = math.floor(position[1]) - self.opti.OptitrackOffset[1]  # z mm
                            self.opti.OptitrackData[2] = angle - self.opti.OptitrackOffset[2]  # degree
                            self.opti.OptitrackData[2] = self.opti.OptitrackData[2] % 360  # [0, 360)
                            if self.opti.OptitrackData[2] > 180: self.opti.OptitrackData[2] -= 360  # (-180, 180]
                            self.opti.IsCalibrated = True
                            print("Set offset:", self.opti.OptitrackOffset)
                    else:  # 如果已校准，则再按↓结束进程
                        # 记录
                        current_point = np.array([self.opti.OptitrackData[0], self.opti.OptitrackData[1]])
                        log_str = ''
                        log_str += str(current_point[0]) + ',' + str(current_point[1])
                        # log_str += str(time.time()) + ','
                        # log_str += ','.join(list(map(str,data)))
                        log_str += '\n'
                        self.logfile.write(log_str)
                        self.logfile.flush()

                        # print(self.opti.OptitrackData)
                        if keyboard.is_pressed('down') or height < 800 or self.opti.IsFinished:
                            winsound.Beep(600, 250)
                            winsound.Beep(600, 250)
                            head_process = multiprocessing.Process(target=show_trajmode)
                            head_process.start()
                            print("Trial finished!")
                            self.opti.IsFinished = True
                            break
            except:
                print("Optitrack thread error")
                break


class ConnectDevice:
    def __init__(self):
        super().__init__()
        self.HOST, self.PORT = '', 8080  # 域名和端口号
        self.optiSocket = socket.socket()
        self.clientSocket = socket.socket()

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
        self.optiSocket.bind(('169.254.109.99', 10010))

        winsound.Beep(600, 500)

if __name__ == "__main__":
    connectDevice = ConnectDevice()
    connectDevice.connect()

    taskIndex = '3'  # 任务序号
    optitrack = Optitrack()
    optitrackTread = OptitrackThread(connectDevice.optiSocket, optitrack)
    optitrackTread.start()

