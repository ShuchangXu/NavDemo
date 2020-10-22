import numpy as np
import math
from matplotlib import pyplot as pl
from matplotlib.pyplot import MultipleLocator


pathName = 'L_-45'
createTime = '2020-08-03 14.36.48'
createTime_A = '2020-08-03 15.09.58'
pathFileDir = 'path/' + pathName + '.npy'
logFileDir = 'experiment/lmq/task3' + '/' + pathName + ' ' + createTime + '.csv'
logFileDir_A = 'experiment/lmq/task3' + '/' + pathName + ' ' + createTime_A + '.csv'

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def analysis():
    print("\nShow trajectory...")
    trial_x = []
    trial_z = []
    trial_t = []
    with open(logFileDir_A, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(int(strs[1]))
            trial_z.append(int(strs[2]))
    path = np.load(pathFileDir)
    p = path.T
    pl.subplot(121)
    pl.plot(p[0], p[1], linewidth=0.5)

    velocity = []
    velocity_length = 4
    n = len(trial_x)
    for i in range(n):
        s = 0
        start_index = max([0, int(i-velocity_length/2)])
        finish_index = min([n-1, int(i+velocity_length/2)])
        for j in np.arange(start_index, finish_index, 1):  # 四段
            s += distance(trial_x[j], trial_z[j], trial_x[j+1], trial_z[j+1])
        v = float(s/(trial_t[finish_index] - trial_t[start_index]))
        #print(v)
        velocity.append(v)

    pl.scatter(trial_x, trial_z, s=1, c=velocity, cmap='plasma')

    pl.axis('equal')
    pl.grid(True)
    # ax = pl.gca()
    # x_miloc = MultipleLocator(100)
    # y_miloc = MultipleLocator(100)
    # ax.xaxis.set_minor_locator(x_miloc)
    # ax.yaxis.set_minor_locator(y_miloc)
    # x_maloc = MultipleLocator(1000)
    # y_maloc = MultipleLocator(1000)
    # ax.xaxis.set_major_locator(x_maloc)
    # ax.yaxis.set_major_locator(y_maloc)
    # ax.grid(which='minor', color='#EEEEEE')
    # ax.grid(which='major', color='#AAAAAA')

    # trial_x = []
    # trial_z = []
    # with open(logFileDir_A, 'r') as f:
    #     for line in f:
    #         strs = line.strip('\n').split(',')
    #         trial_x.append(int(strs[1]))
    #         trial_z.append(int(strs[2]))
    # path = np.load(pathFileDir)
    # p = path.T
    # pl.subplot(122)
    # pl.plot(p[0], p[1])
    # pl.plot(trial_x, trial_z)
    #
    # pl.axis('equal')
    # pl.grid(True)
    # # ax = pl.gca()
    # # x_miloc = MultipleLocator(100)
    # # y_miloc = MultipleLocator(100)
    # # ax.xaxis.set_minor_locator(x_miloc)
    # # ax.yaxis.set_minor_locator(y_miloc)
    # # x_maloc = MultipleLocator(1000)
    # # y_maloc = MultipleLocator(1000)
    # # ax.xaxis.set_major_locator(x_maloc)
    # # ax.yaxis.set_major_locator(y_maloc)
    # # ax.grid(which='minor', color='#EEEEEE')
    # # ax.grid(which='major', color='#AAAAAA')

    pl.show()


analysis()
