import sys
import struct
import math
import time
import numpy as np
import random
import os
from playsound import playsound
import matplotlib.pyplot as plt
import threading
import pandas as pd

import winsound
from scipy.stats import f_oneway


print("hi")
# playsound('verbal/12.mp3')
# winsound.Beep(600, 250)
# winsound.Beep(600, 250)

#print(math.exp(1))

#print((-1)%360)
# dedata = struct.unpack('>II', data)
# print(dedata)
# print(struct.calcsize('>II'))

#os.mkdir('experiment/abcd')

# t = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print(t)
#
# for i in range(10):
#     print('\r' + str(time.time()), end='')
#     time.sleep(1)
#
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# target_angles = [-120,-90,-60,-40,-30,-20,-10,10,20,30,40,60,90,120]
# random.seed(time.time())
# random.shuffle(target_angles)
# print(target_angles)

# a1 = np.arange(-165, 0, 15)
# a2 = np.arange(15, 181, 15)
# target_angles = np.hstack((a1, a2)) * math.pi / 180
# print(target_angles)
#
# rx = []
# ry = []
# r = 1.0
# for a in target_angles:
#     aa = (a + 90) * math.pi / 180
#     x = r * math.cos(aa)
#     y = r * math.sin(aa)
#     rx.append(x)
#     ry.append(y)
#
# rs = np.array([r]*len(target_angles))
# ax = pl.subplot(111, projection='polar')
# ax.scatter(target_angles, rs)
# pl.show()
#
# ax2 = pl.subplot(111,projection='polar')
# ax2.plot()
# ax2.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha =0.75)
#
# ax = pl.subplot(111, projection='polar')
# pl.grid(True)
# ax.scatter(target_angles, rs)
# pl.show()


# line_l9\n"
#                   "# Right turn: L_45 L_90 L_135\n"
#                   "# Left turn: L_-45 L_-90 L_-135\n"
#                   "# S curve: S_r3.2

# class Clock(threading.Thread):
#     def __init__(self, count):
#         super().__init__()
#         self.count = count
#         self.current = 0
#
#     def run(self):
#         playverbal = Playverbal(1)
#         while self.current < self.count:
#             self.current += 1
#             if self.current == 5:
#                 playverbal.start()
#             print(self.current, playverbal.isAlive())
#             time.sleep(0.5)
#
#
# class Playverbal(threading.Thread):
#     def __init__(self, index):
#         super().__init__()
#         self.verbalDir = 'verbal/' + str(index) + '.mp3'
#
#     def run(self):
#         playsound(self.verbalDir)
#
# clock = Clock(10)
# clock.start()

# name = 'yty'
# filePath = 'experiment/' + name + '/task3'
# fileList = os.listdir(filePath)
# logFileName = fileList[0]
# print(logFileName)
# strs = logFileName.split(' ')
# print(strs)

# name = 'yty'
# saveFidDir = 'experiment/' + name + '/analysis'
# os.mkdir(saveFidDir)
# name = 'lxue'
# logFilePath = 'E:/blind/data_analysis/successful_task/' + name + '/task3'
# logfileList = os.listdir(logFilePath)
# pathName_list = []
# time_list = []
# for logFileName in logfileList:
#     info = logFileName.split(' ')
#     if info[0] == 'A':
#         pathName = info[1]
#         time = info[3]
#     else:
#         pathName = info[0]
#         time = info[2]
#     pathName_list.append(pathName)
#     timeinfo = time.split('.')
#     hms = timeinfo[0]+timeinfo[1]+timeinfo[2]
#     hms = int(hms)
#     time_list.append(hms)
#
#
# order = pd.DataFrame(index=time_list)
# order['path'] = pathName_list
# order = order.sort_index(axis=0)
# print(order)


save_dir = 'E:/blind/data_analysis/task3/filt_0818/（20人）task3_data_collection filt_0818 2020-08-18 21.23.34.xlsx'
data = pd.read_excel(save_dir, sheet_name='mean_deivation', index_col=0)
# print(data)
path_list = data.index
result = pd.DataFrame(columns=('F', 'p'))
name_list = data.columns
sub_data = data
droped_name = 'all'
for i in range(20):
    sub_name_list = sub_data.columns
    if len(sub_name_list) == 1: break
    T_sub_list = []
    TA_sub_list = []
    for path in path_list:
        info = path.split(' ')
        if info[0] == 'A':
            T_sub_list.extend(sub_data.loc[path])
        else:
            TA_sub_list.extend(sub_data.loc[path])
    stat_sub, p_sub = f_oneway(T_sub_list, TA_sub_list)
    result.loc[droped_name] = [stat_sub, p_sub]

    temp_result = pd.DataFrame(columns=('F', 'p'))
    for temp_drop_name in sub_name_list: # 去掉某一个
        new_data = pd.DataFrame()
        for name in sub_name_list:
            if temp_drop_name == name: continue
            new_data[name] = sub_data[name]
        T_list = []
        TA_list = []
        for path in path_list:
            info = path.split(' ')
            if info[0] == 'A':
                TA_list.extend(new_data.loc[path])
            else:
                T_list.extend(new_data.loc[path])
        stat, p = f_oneway(T_list, TA_list)
        temp_result.loc[temp_drop_name] = [stat, p]
    temp_result = temp_result.sort_values(by='p')

    droped_name = temp_result.index[0] # 去掉列表第一个
    print(droped_name)
    sub_data = sub_data.drop(droped_name, 1)

print(result)

# sub_name_list = ['lmq', 'zlj', 'cj', 'yt', 'wb', 'yxy', 'yhr', 'lx', 'lzz', 'txl', 'xtq', 'cfq', 'zhy', 'wjx', 'hjs',
#           'wxm',  'lxue']
#
# sub_data = pd.DataFrame()
# for name in sub_name_list:
#     sub_data[name] = data[name]
# T_sub_list = []
# TA_sub_list = []
# for path in path_list:
#     info = path.split(' ')
#     if info[0] == 'A':
#         T_sub_list.extend(sub_data.loc[path])
#     else:
#         TA_sub_list.extend(sub_data.loc[path])
# stat_sub, p_sub = f_oneway(T_sub_list, TA_sub_list)
# print('num='+str(len(sub_name_list)), 'F='+str(stat_sub), 'p='+str(p_sub))
#
# for drop_name in sub_name_list:
#     new_data = pd.DataFrame()
#     for name in sub_name_list:
#         if drop_name == name: continue
#         new_data[name] = sub_data[name]
#     T_list = []
#     TA_list = []
#     for path in path_list:
#         info = path.split(' ')
#         if info[0] == 'A':
#             TA_list.extend(new_data.loc[path])
#         else:
#             T_list.extend(new_data.loc[path])
#     stat, p = f_oneway(T_list, TA_list)
#     result.loc[drop_name] = [stat, p]
# result = result.sort_values(by='p')
# print(result)
