import math
import os
import numpy as np
import time
import random
import threading
from collections import deque
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

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


class DemoTask:
    def __init__(self, pathName, logUserTaskDir, createTime):
        super().__init__()

        if os.path.exists(logUserTaskDir):
            pass
        else:
            os.mkdir(logUserTaskDir)
            print("Create user task dir:", logUserTaskDir)

        self.createTime = createTime
        self.pathName = pathName
        self.pathFileDir = 'path/' + self.pathName + '.txt'
        self.logFileDir = logUserTaskDir + '/' + self.pathName + ' ' + self.createTime + '.csv'

        self.path = None
        self.logfile = None

        self.last_index = 0
        self.ahead_distance = 40  # gd

        self.filter1 = Filter(5)  # tc_targetDirection s
        self.filter2 = Filter(0)  # tc_currentDirection s

    def calculateInitiate(self):  # 显示轨迹图时不执行
        self.path = np.loadtxt(self.pathFileDir)
        self.logfile = open(self.logFileDir, 'w')  # 如果已经存在，则覆盖之前的
        print("Load path:", self.pathFileDir)  # 加载完成


    def calculate(self, OptitrackData):
        vibrationIntensity = 0

        current_point = np.array([OptitrackData[0], OptitrackData[1]])
        currentDirection = OptitrackData[2]  # (-180,+180]
        # currentDirection = self.filter2.filter(currentDirection)  # 2.2-低通滤波用户方位

        # nearest point 获取最近点
        nearest_distance = np.min(np.linalg.norm(self.path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(self.path[:] - current_point, axis=1))
        last_index = len(self.path) - 1

        # 判断结束
        if abs(nearest_index - last_index) <= 3 and nearest_distance <= 300:  # 终点附近， 30㎝半径圆之内
            print("Arrive terminal!")
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
        if targetDirection < -90: targetDirection += 360  # [-90,+270)

        targetDirection = targetDirection - 90

        mylogStr = " desVec (" 
        mylogStr += str(target_vector[0]) + "," + str(target_vector[1])
        mylogStr += "), desA = " + targetDirection

        self.filter1.filter(targetDirection, 0)  # 2.2-低通滤波物理方位

        # command direction
        commandDirection = targetDirection - currentDirection
        commandDirection = commandDirection % 360  # [0, 360)
        if commandDirection > 180: commandDirection -= 360  # (-180, 180]
        # print(commandDirection)
        commandDirection = int(float(commandDirection) * min((dangerFactor * 1.5 + 0.1), 1))  # 2.1-距离乘以角度非线性
        commandDirection = commandDirection % 360  # [0, 360)
        if commandDirection > 180: commandDirection -= 360  # (-180, 180]

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

        return commandDirection, vibrationIntensity, mylogStr