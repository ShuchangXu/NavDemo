import math
import numpy as np
from collections import deque
from matplotlib import pyplot as pl
from matplotlib.pyplot import MultipleLocator


def task1demo(OptitrackData):
    targetDirection = 0  # [-90, 90]
    commandDirection = targetDirection
    # print(commandDirection)
    return commandDirection


def task2demo(OptitrackData):
    targetDirection = 0
    currentDirection = OptitrackData[2]
    commandDirection = targetDirection - currentDirection
    if commandDirection > 180: commandDirection -= 360
    elif commandDirection <= -180: commandDirection += 360
    # print(commandDirection)
    return commandDirection

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
        if dx < -340: dx += 360
        elif dx > 340: dx -= 360
        self.x_continue += dx
        # print(x, self.x_continue)
        self.filterTemp.append(self.x_continue)

        df = min(df, self.max_dangerFactor)
        wc = self.wc / (1 - df)

        filterLength = int(3 * 50 / wc) + 1
        result = 0.0
        for i in range(filterLength):
            result += self.filterTemp[self.filterTempLength-1-i] * math.exp(-wc*0.02*i) * 3 / filterLength

        self.x_last = x
        return int(result)


class Task3:
    def __init__(self, pathName, logUserTaskDir):
        super().__init__()
        # 直线 line_l6 line_l10 line_l9
        # 右转弯 L_45 L_90 L_135
        # 左转弯 L_-45 L_-90 L_-135
        # S形曲线 S_r1 S_r3.2 S_r3.5
        # circle_r1
        # arc_r6
        '''
        print("直线 line_l9\n"
              "右转弯 L_45 L_90 L_135\n"
              "左转弯 L_-45 L_-90 L_-135\n"
              "S形曲线 S_r3.2")
              '''
        self.pathName = pathName
        self.pathNameStr = 'path/' + self.pathName + '.npy'
        self.logNameStr = 'log/' + self.pathName + '.txt'

        self.path = None
        self.logfile = None

        self.last_index = 0
        self.load_distance = 20
        self.ahead_distance = 40  # gd

        self.filter1 = Filter(5)  # tc_targetDirection s
        self.filter2 = Filter(0)  # tc_currentDirection s

        self.verbalSerial = []

        self.verbalProgress = 0

    def calculateInitiate(self):
        self.path = np.load(self.pathNameStr)
        self.logfile = open(self.logNameStr, 'w')
        # 记录文件
        # self.logfile_name = input('logfile name memo:')
        # self.logNameStr = 'log/'+pathName+'_'+logfile_name+'.txt'

    def calculate(self, OptitrackData):
        vibrationIntensity = 0

        current_point = np.array([OptitrackData[0], OptitrackData[1]])
        currentDirection = OptitrackData[2]  # (-180,+180]
        # currentDirection = self.filter2.filter(currentDirection)  # 2.2-低通滤波用户方位

        # nearest point 获取最近点
        nearest_distance = np.min(np.linalg.norm(self.path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(self.path[:] - current_point, axis=1))
        last_index = len(self.path)-1

        # 判断结束
        if abs(nearest_index - last_index) <= 3 and nearest_distance <= 150:  # 终点附近， 15㎝半径圆之内?
            return 0, -1  # 结束

        safe_distance = 300  # 半路宽
        dangerFactor = float(nearest_distance) / float(safe_distance)
        # print(dangerFactor)
        # 1-超出半路宽震动
        if nearest_distance >= safe_distance:
            vibrationIntensity = 200

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
        targetDirection = self.filter1.filter(targetDirection, 0)  # 2.2-低通滤波物理方位

        # command direction
        commandDirection = targetDirection - currentDirection
        commandDirection = commandDirection % 360  # [0, 360)
        if commandDirection > 180: commandDirection -= 360  # (-180, 180]
        # print(commandDirection)
        commandDirection = int(float(commandDirection) * (dangerFactor * 1.5 + 0.1))  # 2.1-距离乘以角度非线性
        commandDirection = commandDirection % 360  # [0, 360)
        if commandDirection > 180: commandDirection -= 360  # (-180, 180]


        # 1-角度阈值震动
        # if abs(commandDirection) >= 50:
        #   vibrationIntensity = 90

        log_str = ''
        log_str += str(current_point[0]) + ',' + str(current_point[1])
        # log_str += str(time.time()) + ','
        # log_str += ','.join(list(map(str,data)))
        log_str += '\n'
        self.logfile.write(log_str)
        self.logfile.flush()

        # print(log_str)
        #print(commandDirection)
        # print("current position:", current_point, "target point", target_point, "td", targetDirection)
        return commandDirection, vibrationIntensity

    def analysis(self):
        trial_x = []
        trial_z = []
        with open(self.logNameStr, 'r') as f:
            for line in f:
                strs = line.strip('\n').split(',')
                trial_x.append(int(strs[0]))
                trial_z.append(int(strs[1]))
                # print(strs[0], strs[1])

        path = np.load(self.pathNameStr)
        p = path.T
        pl.plot(p[0], p[1])
        pl.plot(trial_x, trial_z)
        pl.axis('equal')
        pl.grid(True)
        x_major_locator = MultipleLocator(100)
        y_major_locator = MultipleLocator(100)
        ax = pl.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        pl.show()


def show(index, pathName, logUserTaskDir):
    if index == '3':
        task = Task3(pathName, logUserTaskDir)
        task.analysis()
    print("Trial finish--------------------")


def show_trajmode():
    trial_x = []
    trial_z = []
    with open('log/trajRecord.txt', 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_x.append(int(strs[0]))
            trial_z.append(int(strs[1]))
            # print(strs[0], strs[1])

    pl.plot(trial_x, trial_z)
    pl.axis('equal')
    pl.grid(True)
    x_major_locator = MultipleLocator(100)
    y_major_locator = MultipleLocator(100)
    ax = pl.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    pl.show()


if __name__ == "__main__":
    show_trajmode()

