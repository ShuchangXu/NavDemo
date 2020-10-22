import math
import numpy as np
import time
import random
import keyboard
import winsound
import threading
from playsound import playsound
from collections import deque
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator


def task1demo(OptitrackData):
    targetDirection = 0  # [-90, 90]
    commandDirection = targetDirection
    # print(commandDirection)
    return commandDirection


class Task1:
    def __init__(self, _, logUserTaskDir, createTime, __=0):
        super().__init__()
        self.isBegun = False
        self.pause = False

        self.createTime = createTime
        self.logFileDir = logUserTaskDir + '/' + self.createTime + '.csv'
        self.rawLogFileDir = logUserTaskDir + '/Raw' + self.createTime + '.csv'
        self.logfile = None
        self.rawLogFile = None

        a1 = np.arange(-165, 0, 15)
        a2 = np.arange(15, 181, 15)

        # a1 = np.arange(-150, 0, 30)
        # a2 = np.arange(30, 181, 30)
        self.orientationSerial = np.hstack((a1, a2))
        self.orientationSerialProgress = 0

        self.startDirection = 0
        self.finishDirection = 0

        self.startTime = 0.0
        self.finishTime = 0.0

    def calculateInitiate(self):
        # 生成序列
        random.seed(time.time())
        random.shuffle(self.orientationSerial)
        print("Shuffle orientation serial:", self.orientationSerial)
        print("log file dir:", self.logFileDir)
        print("raw log file dir:", self.rawLogFileDir)
        self.logfile = open(self.logFileDir, 'w')  # 如果已经存在，则覆盖之前的
        self.rawLogFile = open(self.rawLogFileDir, 'w')
        print("openOK")

    def begin(self):
        if not self.isBegun:
            self.isBegun = True
            print("\nStart...")
            playsound('verbal/15.mp3')  # 开始实验
            time.sleep(0.5)
            print("1/23 Orienting", self.orientationSerial[self.orientationSerialProgress], end=', ')
            winsound.Beep(600, 500)  # 定位开始
            self.startTime = time.time()

    def calculate(self, OptitrackData):
        vibrationIntensity = 0

        current_point = np.array([OptitrackData[0], OptitrackData[1]])
        currentDirection = OptitrackData[2]  # (-180,+180]
        targetDirection = self.startDirection + self.orientationSerial[self.orientationSerialProgress]

        commandDirection = targetDirection - currentDirection
        commandDirection = commandDirection % 360  # [0, 360)
        if commandDirection > 180: commandDirection -= 360  # (-180, 180]

        # 记录
        raw_log_str = ''  # 时刻，x，z
        raw_log_str += str(time.time()) + ','
        raw_log_str += str(current_point[0]) + ',' 
        raw_log_str += str(current_point[1]) + ',' 
        raw_log_str += str(currentDirection) + ','
        raw_log_str += str(commandDirection) + ','
        raw_log_str += str(self.orientationSerialProgress + 1) + '\n'
        self.rawLogFile.write(raw_log_str)
        self.rawLogFile.flush()

        if self.pause:
            duration = self.finishTime - self.startTime
            log_str = ''  # 目标角度，初始角度，结束角度，用时s
            log_str += str(self.orientationSerial[self.orientationSerialProgress]) + ','
            log_str += str(self.startDirection) + ',' 
            log_str += str(self.finishDirection) + ','
            log_str += str(duration) + ','
            log_str += str(self.startTime) + ','
            log_str += str(self.finishTime)
            log_str += '\n'
            self.logfile.write(log_str)
            self.logfile.flush()

            self.orientationSerialProgress += 1
            if self.orientationSerialProgress >= len(self.orientationSerial):
                print('All oriented!')
                return 0, -1  # 结束

            time.sleep(3)  # 继续
            print(str(self.orientationSerialProgress + 1) + "/23 Orienting",
                  self.orientationSerial[self.orientationSerialProgress], end=', ')
            winsound.Beep(600, 500)  # 定位开始
            self.pause = False
            self.startDirection = currentDirection  # 开始采样
            self.startTime = time.time()

        if keyboard.is_pressed('right'):
            self.finishTime = time.time()
            self.finishDirection = currentDirection  # 结束采样
            print(self.orientationSerial[self.orientationSerialProgress], "oriented!")
            winsound.Beep(600, 250)  # 定位结束
            winsound.Beep(600, 250)
            self.pause = True
            return 1000, 0  # 复位

        return commandDirection, vibrationIntensity

    def analysis(self):
        print("\nShow result...")
        target = []
        real = []
        with open(self.logFileDir, 'r') as f:
            for line in f:
                strs = line.strip('\n').split(',')
                target.append(int(strs[0]))
                real.append(int(strs[2]) - int(strs[1]))
        r = 1.0
        rs = np.array([r] * len(real))

        target = np.array(target) / 180 * math.pi
        real = np.array(real) / 180 * math.pi
        ax = plt.subplot(111, projection='polar')
        ax.plot(target, rs, '+')
        ax.plot(real, rs, '.')
        plt.show()

class Task2:
    def __init__(self, _, logUserTaskDir, createTime, __=0):
        super().__init__()
        self.isBegun = False
        self.pause = False

        self.createTime = createTime
        self.logFileDir = logUserTaskDir + '/' + self.createTime + '.csv'
        self.rawLogFileDir = logUserTaskDir + '/Raw' + self.createTime + '.csv'
        self.stdLogFileDir = logUserTaskDir + '/Std' + self.createTime + '.csv'
        self.logfile = None
        self.rawLogFile = None
        self.stdLogFile = None

        a1 = np.arange(-120, 0, 30)
        a2 = np.arange(30, 120, 30)

        self.orientationSerial = np.hstack((a1, a2))
        self.orientationSerialProgress = 0

        self.startDirection = 0
        self.finishDirection = 0

        self.startPosition = np.array([0, 0])

        self.startTime = 0.0
        self.finishTime = 0.0

    def calculateInitiate(self):
        # 生成序列
        random.seed(time.time())
        random.shuffle(self.orientationSerial)
        print("Shuffle orientation serial:", self.orientationSerial)
        print("log file dir:", self.logFileDir)

        self.logfile = open(self.logFileDir, 'w')  # 如果已经存在，则覆盖之前的
        self.rawLogFile = open(self.rawLogFileDir, 'w')
        self.stdLogFile = open(self.stdLogFileDir, 'w')

        print("openOK")

    def begin(self):
        if not self.isBegun:
            self.isBegun = True
            print("\nStart...")
            playsound('verbal/15.mp3')  # 开始实验
            time.sleep(0.5)
            print("1/8 Orienting", self.orientationSerial[self.orientationSerialProgress], end=', ')
            winsound.Beep(600, 500)  # 定位开始
            self.startTime = time.time()

    def calculate(self, OptitrackData):
        vibrationIntensity = 0

        current_point = np.array([OptitrackData[0], OptitrackData[1]])
        currentDirection = OptitrackData[2]  # (-180,+180]
        targetDirection = self.startDirection + self.orientationSerial[self.orientationSerialProgress]

        commandDirection = targetDirection - currentDirection
        commandDirection = commandDirection % 360  # [0, 360)
        if commandDirection > 180: commandDirection -= 360  # (-180, 180]

        radD = math.radians(self.startDirection)
        cosD = math.cos(radD)
        sinD = math.sin(radD)
        std_position = np.array([[cosD, sinD], [-sinD, cosD]]).dot(current_point - self.startPosition)
        std_orientation = currentDirection - self.startDirection

        # 记录
        raw_log_str = ''  # 时刻，x，z
        raw_log_str += str(time.time()) + ','
        raw_log_str += str(current_point[0]) + ',' 
        raw_log_str += str(current_point[1]) + ',' 
        raw_log_str += str(currentDirection) + ','
        raw_log_str += str(commandDirection) + ','
        raw_log_str += str(self.orientationSerialProgress + 1) + '\n'
        self.rawLogFile.write(raw_log_str)
        self.rawLogFile.flush()

        std_log_str = ''  # 时刻，x，z
        std_log_str += str(time.time()) + ','
        std_log_str += str(std_position[0]) + ',' 
        std_log_str += str(std_position[1]) + ',' 
        std_log_str += str(std_orientation) + ','
        std_log_str += str(commandDirection) + ','
        std_log_str += str(self.orientationSerialProgress + 1) + '\n'
        self.stdLogFile.write(std_log_str)
        self.stdLogFile.flush()

        if self.pause:
            duration = self.finishTime - self.startTime
            log_str = ''  # 目标角度，初始角度，结束角度，用时s
            log_str += str(self.orientationSerial[self.orientationSerialProgress]) + ','
            log_str += str(self.startDirection) + ',' 
            log_str += str(self.finishDirection) + ','
            log_str += str(duration) + ','
            log_str += str(self.startTime) + ','
            log_str += str(self.finishTime)
            log_str += '\n'
            self.logfile.write(log_str)
            self.logfile.flush()

            self.orientationSerialProgress += 1
            if self.orientationSerialProgress >= len(self.orientationSerial):
                print('All oriented!')
                return 0, -1  # 结束

            while not keyboard.is_pressed('space'):
                pass
            print(str(self.orientationSerialProgress + 1) + "/8 Orienting",
                  self.orientationSerial[self.orientationSerialProgress], end=', ')
            winsound.Beep(600, 500)  # 定位开始
            self.pause = False
            self.startDirection = currentDirection  # 开始采样
            self.startPosition = current_point
            self.startTime = time.time()

        if keyboard.is_pressed('right'):
            self.finishTime = time.time()
            self.finishDirection = currentDirection  # 结束采样
            print(self.orientationSerial[self.orientationSerialProgress], "oriented!")
            winsound.Beep(600, 250)  # 定位结束
            winsound.Beep(600, 250)
            self.pause = True
            return 1000, 0  # 复位

        return commandDirection, vibrationIntensity

    def analysis(self):
        print("\nShow trajectory...")
        trial_x = []
        trial_z = []
        with open(self.stdLogFile, 'r') as f:
            for line in f:
                strs = line.strip('\n').split(',')
                trial_x.append(int(strs[1]))
                trial_z.append(int(strs[2]))
                # print(strs[0], strs[1])

        plt.scatter(trial_x, trial_z, color='red', s=0.5, zorder=10)
        plt.axis('equal')
        plt.grid(True)
        x_major_locator = MultipleLocator(100)
        y_major_locator = MultipleLocator(100)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.show()


class VerbalThread(threading.Thread):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def run(self):
        verbalDir = 'verbal/' + str(self.index) + '.mp3'
        if self.index != 0:
            playsound(verbalDir)
            time.sleep(1)


class Filter:
    def __init__(self, tc):
        super().__init__()
        self.wc = 0
        self.filterTempLength = 0
        self.filterTemp = None
        if tc > 0: self.filterInitiate(tc)
        self.x_last = 0
        self.x_continue = 0
        self.max_dangerFactor = 0.9  # 和危险系数有关

    def filterInitiate(self, tc):
        self.wc = 2 * math.pi / float(tc)  # 截止频率fc，截至周期tc
        self.filterTempLength = int(3 * 50 / self.wc) + 1  # N~23tc, T~0.5tc
        self.filterTemp = deque(maxlen=self.filterTempLength)
        for i in range(self.filterTempLength):
            self.filterTemp.append(0)

    def filter(self, x, df):  # x=(-180, 180]
        if self.filterTempLength == 0: return x
        dx = x - self.x_last
        if dx < -340:
            dx += 360
        elif dx > 340:
            dx -= 360
        self.x_continue += dx
        # print(x, self.x_continue)
        self.filterTemp.append(self.x_continue)

        df = min(df, self.max_dangerFactor)
        wc = self.wc / (1 - df)

        filterLength = int(3 * 50 / wc) + 1
        result = 0.0
        for i in range(filterLength):
            result += self.filterTemp[self.filterTempLength - 1 - i] * math.exp(-wc * 0.02 * i) * 3 / filterLength

        self.x_last = x
        return int(result)


class Task3:
    def __init__(self, pathName, logUserTaskDir, creatTime, verbalOn=0):
        super().__init__()
        self.isBegun = False

        self.createTime = creatTime
        self.pathName = pathName
        self.pathFileDir = 'path/' + self.pathName + '.npy'
        self.logFileDir = ''
        if verbalOn: self.logFileDir = logUserTaskDir + '/A ' + self.pathName + ' ' + self.createTime + '.csv'
        else: self.logFileDir = logUserTaskDir + '/' + self.pathName + ' ' + self.createTime + '.csv'

        self.path = None
        self.logfile = None

        self.last_index = 0
        self.load_distance = 20
        self.ahead_distance = 40  # gd

        self.filter1 = Filter(5)  # tc_targetDirection s
        self.filter2 = Filter(0)  # tc_currentDirection s

        self.verbalOn = verbalOn
        self.verbalSerial = []  # [position_index, verbal_index]
        self.verbalProgress = 0
        self.verbalThread = VerbalThread(0)

    def calculateInitiate(self):  # 显示轨迹图时不执行
        self.path = np.load(self.pathFileDir)
        self.logfile = open(self.logFileDir, 'w')  # 如果已经存在，则覆盖之前的

        if self.pathName == 'line_l9':
            self.verbalSerial.append([30, 1])
        elif self.pathName == 'L_45':
            self.verbalSerial.append([30, 7])
            self.verbalSerial.append([500, 3])
            self.verbalSerial.append([734, 4])
        elif self.pathName == 'L_-45':
            self.verbalSerial.append([30, 2])
            self.verbalSerial.append([500, 3])
            self.verbalSerial.append([734, 4])
        elif self.pathName == 'L_90':
            self.verbalSerial.append([30, 8])
            self.verbalSerial.append([500, 3])
            self.verbalSerial.append([656, 4])
        elif self.pathName == 'L_-90':
            self.verbalSerial.append([30, 5])
            self.verbalSerial.append([500, 3])
            self.verbalSerial.append([656, 4])
        elif self.pathName == 'L_135':
            self.verbalSerial.append([30, 9])
            self.verbalSerial.append([500, 21])
            self.verbalSerial.append([577, 4])
        elif self.pathName == 'L_-135':
            self.verbalSerial.append([30, 6])
            self.verbalSerial.append([500, 21])
            self.verbalSerial.append([577, 4])
        elif self.pathName == 'S_r3.2':
            self.verbalSerial.append([30, 10])
            self.verbalSerial.append([502, 11])
        elif self.pathName == 'learn':  # 修改
            self.verbalSerial.append([30, 18])
            self.verbalSerial.append([700, 19])
            self.verbalSerial.append([934, 20])
            # self.verbalSerial.append([1033, 11])
            self.verbalSerial.append([1424, 10])
        print("Load path:", self.pathFileDir)  # 加载完成

    def playverbal(self, nearest_index):
        # 播报的不再播报， 可以跳过   not self.verbalThread.isAlive() and
        if self.verbalProgress < len(self.verbalSerial):
            temp = self.verbalProgress  # 从progress开始遍历，temp为当前遍历的序数
            for vs in self.verbalSerial[self.verbalProgress:]:
                if 0 <= (vs[0] - nearest_index) <= 30:  # 前后10cm以内，增加10cm以补偿程序播放音频的延迟
                    self.verbalProgress = temp + 1
                    self.verbalThread = VerbalThread(vs[1])
                    self.verbalThread.start()
                    break
                temp += 1

    def begin(self):
        if not self.isBegun:
            print("\nStart walking...")
            playsound('verbal/14.mp3')  # 开始行走
            self.isBegun = True

    def calculate(self, OptitrackData):
        vibrationIntensity = 0

        current_point = np.array([OptitrackData[0], OptitrackData[1]])
        currentDirection = OptitrackData[2]  # (-180,+180]
        # currentDirection = self.filter2.filter(currentDirection)  # 2.2-低通滤波用户方位

        # nearest point 获取最近点
        nearest_distance = np.min(np.linalg.norm(self.path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(self.path[:] - current_point, axis=1))
        last_index = len(self.path) - 1

        # 播报语音，注释掉即关闭语音
        if self.verbalOn: self.playverbal(nearest_index)

        # 判断结束
        if abs(nearest_index - last_index) <= 3 and nearest_distance <= 300:  # 终点附近， 30㎝半径圆之内
            print("Arrive terminal!")
            playsound('verbal/17.mp3')  # 抵达终点
            return 0, -1  # 结束

        # 宽度
        safe_distance = 300  # 半路宽
        dangerFactor = float(nearest_distance) / float(safe_distance)
        # print(dangerFactor)
        # 1-超出半路宽震动
        if nearest_distance >= safe_distance:
            vibrationIntensity = 100  # pattern: ...100100...

        # target point - 获取最近点向前一定距离、终点之前的目标点
        try:
            target_point = self.path[nearest_index + self.ahead_distance]  # ahead_distance不均匀
        except:
            target_point = self.path[-1]
        target_vector = (target_point - current_point)

        # target direction
        targetDirection = int(math.atan2(target_vector[1], target_vector[0]) / math.pi * 180)
        targetDirection = 90 - targetDirection
        if targetDirection > 180: targetDirection -= 360  # (-180,+180]
        self.filter1.filter(targetDirection, 0)  # 2.2-低通滤波物理方位

        # command diion = selrection
        commandDirection = targetDirection - currentDirection
        commandDirection = commandDirection % 360  # [0, 360)
        if commandDirection > 180: commandDirection -= 360  # (-180, 180]
        # print(commandDirection)
        commandDirection = int(float(commandDirection) * min((dangerFactor * 1.5 + 0.1), 1))  # 2.1-距离乘以角度非线性
        commandDirection = commandDirection % 360  # [0, 360)
        if commandDirection > 180: commandDirection -= 360  # (-180, 180]

        # debug_info = ''
        # debug_info += "pos = (" + str(current_point[0]) + ", " + str(current_point[1]) + "), "
        # debug_info += "dir = " + str(currentDirection) +", "
        # debug_info += "goal_pos = (" + str(target_point[0]) + ", " + str(target_point[1]) + "), "
        # debug_info += "diff = (" + str(target_vector[0]) + ", " + str(target_vector[1]) + "), "
        # debug_info += "goal_dir = " + str(targetDirection) +", "
        # debug_info += "cmd = " + str(commandDirection) +"\n"

        # print(debug_info)
        # self.logfile.write(debug_info)
        # self.logfile.flush()

        # 记录
        log_str = ''  # 时刻，x，z
        log_str += str(time.time()) + ','
        log_str += str(current_point[0]) + ','
        log_str += str(current_point[1]) + ','
        log_str += str(currentDirection) + ','
        log_str += str(commandDirection)
        log_str += '\n'
        self.logfile.write(log_str)
        self.logfile.flush()
        # print(log_str)
        # print(commandDirection)
        # print("current position:", current_point, "target point", target_point, "td", targetDirection)

        return commandDirection, vibrationIntensity

    def analysis(self):
        print("\nShow trajectory...")
        trial_x = []
        trial_z = []
        with open(self.logFileDir, 'r') as f:
            for line in f:
                strs = line.strip('\n').split(',')
                trial_x.append(int(strs[1]))
                trial_z.append(int(strs[2]))
                # print(strs[0], strs[1])

        path = np.load(self.pathFileDir)
        p = path.T
        plt.plot(p[0], p[1])
        plt.scatter(trial_x, trial_z, color='red', s=0.5, zorder=10)
        plt.axis('equal')
        plt.grid(True)
        x_major_locator = MultipleLocator(100)
        y_major_locator = MultipleLocator(100)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.show()


def show(index, pathName, logUserTaskDir, createTime, verbalOn):
    if index == '3':
        t = Task3(pathName, logUserTaskDir, createTime, verbalOn)
        t.analysis()
    elif index == '2':
        t = Task2(pathName, logUserTaskDir, createTime)
        t.analysis()
    print("Trial finish!")


def show_trajmode():
    trial_x = []
    trial_z = []
    with open('log/trajRecord.txt', 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_x.append(int(strs[0]))
            trial_z.append(int(strs[1]))
            # print(strs[0], strs[1])

    plt.plot(trial_x, trial_z)
    plt.axis('equal')
    plt.grid(True)
    x_major_locator = MultipleLocator(100)
    y_major_locator = MultipleLocator(100)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()


def task3order():
    target_angles = ["2:line_l9", "1:L_45", "1:L_90", "1:L_135", "3:L_-45", "3:L_-90", "3:L_-135", "2:S_r3.2"]
    random.seed(time.time())
    random.shuffle(target_angles)
    for i in target_angles:
        print(i)

if __name__ == "__main__":
    task3order()

