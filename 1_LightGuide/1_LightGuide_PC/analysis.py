from server import *
import pandas as pd
from scipy import interpolate
from scipy.fftpack import fft
from scipy import signal
from matplotlib.collections import LineCollection


def FFT (Fs,data):
    L = len (data)                        # 信号长度
    N = int(np.power(2, np.ceil(np.log2(L))))    # 下一个最近二次幂
    FFT_y1 = np.abs(fft(data,N))/L*2      # N点FFT 变化,但处于信号长度
    Fre = np.arange(int(N/2))*Fs/N        # 频率坐标
    FFT_y1 = FFT_y1[range(int(N/2))]      # 取一半
    return Fre, FFT_y1


def interp(t, x, Fs):
    t_start = t[0]
    t_end = t[-1]
    t_sample = np.arange(t_start, t_end, 1/Fs)
    f_linear = interpolate.interp1d(t, x)
    x_sample = f_linear(t_sample)
    return t_sample, x_sample


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))


# 步速分布、轨迹可视化，散点图
def plot_single_trial(name, logFilePath, logFileName, show, saveFigDir = ''):
    info = logFileName.split(' ')
    if info[0] == 'A': pathName = info[1]
    else: pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath + '/' + logFileName

    trial_x = []
    trial_z = []
    trial_t = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(int(strs[1]))
            trial_z.append(int(strs[2]))
    path = np.load(pathFileDir)
    p = path.T
    pathOutline = np.load(pathOutlineFileDir)
    pOutline = pathOutline.T
    print("open: " + logFileDir)

    velocity = []  # cm/s
    velocity_length = 4
    n = len(trial_x)
    for i in range(n):  # -2 -1 0 1 2
        s = 0
        v = 0
        start_index = max([0, int(i-velocity_length/2)])
        finish_index = min([n-1, int(i+velocity_length/2)])
        if trial_t[finish_index] - trial_t[start_index] != 0:
            for j in np.arange(start_index, finish_index, 1):  # 四段
                s += distance(trial_x[j], trial_z[j], trial_x[j+1], trial_z[j+1])
            v = float(s/(trial_t[finish_index] - trial_t[start_index])/10)
        # print(finish_index-start_index, len(np.arange(start_index, finish_index, 1)), v)
        velocity.append(v)

        # current_point = np.array([trial_x[i], trial_z[i]])
        # nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))

    max_velocity = max(velocity)
    velocity = np.array(velocity)
    bins = np.arange(0, max_velocity + 10, 2)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(name + ' : ' + logFileName)

    plt.subplot(121)
    plt.hist(velocity, bins)
    plt.xlabel("velocity (cm/s)")
    plt.ylabel("frequency")

    plt.subplot(122, facecolor='#111111')
    plt.axis('equal')
    ax = plt.gca()
    # 粗格子
    x_miloc = MultipleLocator(100)
    y_miloc = MultipleLocator(100)
    ax.xaxis.set_minor_locator(x_miloc)
    ax.yaxis.set_minor_locator(y_miloc)
    ax.grid(which='major', color='#555555')
    # 细格子
    x_maloc = MultipleLocator(1000)
    y_maloc = MultipleLocator(1000)
    ax.xaxis.set_major_locator(x_maloc)
    ax.yaxis.set_major_locator(y_maloc)
    ax.grid(which='minor', color='#555555', alpha=0.2)
    # 路径
    plt.plot(p[0], p[1], linewidth=1.5, color='#FFFFFF', zorder=5)
    plt.fill(pOutline[0], pOutline[1], color='#FFFFFF', zorder=3, alpha=0.2, lw=0)

    # 轨迹
    plt.scatter(trial_x, trial_z, s=1.5, c=velocity, cmap='jet', vmin=0, vmax=100, zorder=10, alpha=1)
    cb = plt.colorbar()
    cb.set_label("velocity (cm/s)")

    if show:
        plt.show()
    else:
        plt.savefig(saveFigDir)
        print("save:", saveFigDir)
        plt.close()
# 某一被试全部原始数据可视化
def plot_task3_original_single_trial(name, show):
    logFilePath = 'experiment/' + name + '/task3'
    logfileList = os.listdir(logFilePath)
    for logFileName in logfileList:
        saveFigPath = 'experiment/' + name + '/analysis'
        if not os.path.exists(saveFigPath):
            os.mkdir(saveFigPath)
        saveFigPath = 'experiment/' + name + '/analysis/single_trial'
        if not os.path.exists(saveFigPath):
            os.mkdir(saveFigPath)
        saveFigName = name + ' ' + logFileName + '.png'
        saveFigDir = saveFigPath + '/' + saveFigName
        plot_single_trial(name, logFilePath, logFileName, show, saveFigDir)



# d-t图，d-t频谱图，步速分布，轨迹可视化（折线图）
def plot_single_trial_sigment(name, logFilePath, logFileName, show, saveFigDir = ''):
    info = logFileName.split(' ')
    if info[0] == 'A': pathName = info[1]
    else: pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath + '/' + logFileName

    trial_x = []
    trial_z = []
    trial_t = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(float(strs[1]))
            trial_z.append(float(strs[2]))
    path = np.load(pathFileDir)
    p = path.T
    pathOutline = np.load(pathOutlineFileDir)
    pOutline = pathOutline.T
    print("open: " + logFileDir)

    # 计算偏差
    deivation = []  # cm
    deivation_sign = []
    n = len(trial_x)
    for i in range(n):
        x = trial_x[i]
        z = trial_z[i]
        current_point = np.array([x, z])
        nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
        deivation.append(nearest_distance / 10)

        tangent_direction = path[nearest_index + 1] - path[nearest_index]
        normal_direction = current_point - path[nearest_index]
        sign = np.sign(tangent_direction[0] * normal_direction[1] - tangent_direction[1] * normal_direction[0])
        deivation_sign.append(nearest_distance / 10 * sign)

    velocity = []  # cm/s
    velocity_length = 4
    n = len(trial_x)
    for i in range(n):  # -2 -1 0 1 2
        s = 0
        v = 0
        start_index = max([0, int(i-velocity_length/2)])
        finish_index = min([n-1, int(i+velocity_length/2)])
        if trial_t[finish_index] - trial_t[start_index] != 0:
            for j in np.arange(start_index, finish_index, 1):  # 四段
                s += distance(trial_x[j], trial_z[j], trial_x[j+1], trial_z[j+1])
            v = float(s/(trial_t[finish_index] - trial_t[start_index])/10)
        # print(finish_index-start_index, len(np.arange(start_index, finish_index, 1)), v)
        velocity.append(v)

        # current_point = np.array([trial_x[i], trial_z[i]])
        # nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))

    max_velocity = max(velocity)
    velocity = np.array(velocity)
    bins = np.arange(0, max_velocity + 10, 2)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(name + ' : ' + logFileName)

    # d-t图
    #t_sample, d_sample = interp(trial_t, deivation_sign, 50)
    t_sample = trial_t
    d_sample = deivation_sign
    # d_sample -= np.mean(d_sample)
    plt.subplot(321)
    plt.axhline(y=0, color='k', lw=0.5)
    plt.plot(t_sample, d_sample)
    #print(len(t_sample))

    # d-t频谱
    Fre, FFT_d = FFT(50, d_sample)
    plt.subplot(323)
    plt.plot(Fre, FFT_d, lw=1)
    plt.xscale('log')
    plt.yscale('log')
    
    # 速度分布图
    plt.subplot(325)
    plt.hist(velocity, bins)
    plt.xlabel("velocity (cm/s)")
    plt.ylabel("frequency")

    plt.subplot(122, facecolor='#111111')
    plt.axis('equal')
    ax = plt.gca()
    # 粗格子
    x_miloc = MultipleLocator(100)
    y_miloc = MultipleLocator(100)
    ax.xaxis.set_minor_locator(x_miloc)
    ax.yaxis.set_minor_locator(y_miloc)
    ax.grid(which='major', color='#555555')
    # 细格子
    x_maloc = MultipleLocator(1000)
    y_maloc = MultipleLocator(1000)
    ax.xaxis.set_major_locator(x_maloc)
    ax.yaxis.set_major_locator(y_maloc)
    ax.grid(which='minor', color='#555555', alpha=0.2)
    # 路径
    plt.plot(p[0], p[1], linewidth=1.5, color='#FFFFFF', zorder=5)
    plt.fill(pOutline[0], pOutline[1], color='#FFFFFF', zorder=3, alpha=0.2, lw=0)
    # 轨迹
    t = np.array(velocity)
    x = np.array(trial_x)
    y = np.array(trial_z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                        norm=plt.Normalize(0, 100))
    lc.set_array(t)
    lc.set_linewidth(2)
    lc.set_zorder(10)
    ax.add_collection(lc)
    cb = plt.colorbar(lc)
    cb.set_label("velocity (cm/s)")

    if show:
        plt.show()
    else:
        plt.savefig(saveFigDir)
        print("save:", saveFigDir)
        plt.close()
# 数据清洗
def clean_single_trial(name, logFilePath, logFileName, version):
    # name 姓名缩写
    # logFilePath 原始数据所在目录
    # logFileName 原始数据的文件名
    info = logFileName.split(' ')
    if info[0] == 'A': pathName = info[1]
    else: pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath + '/' + logFileName
    path = np.load(pathFileDir)
    print("load: " + pathFileDir)

    trial_x = []
    trial_z = []
    trial_t = []
    trial_o = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(int(strs[1]))
            trial_z.append(int(strs[2]))
            if len(strs) >= 4:
                trial_o.append(int(strs[3]))
            else:
                trial_o.append(0)
    print("open: " + logFileDir)

    saveFilePath = 'E:/blind/data_analysis/task3/' + version
    if not os.path.exists(saveFilePath):
        os.mkdir(saveFilePath)
    saveFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
    if not os.path.exists(saveFilePath):
        os.mkdir(saveFilePath)
    saveFileName = logFileName
    saveFigDir = saveFilePath + '/' + saveFileName
    saveFile = open(saveFigDir, 'w')  # 如果已经存在，则覆盖之前的
    save_x = []
    save_z = []
    save_t = []
    save_o = []
    print("create:", saveFigDir)

    n = len(trial_x)
    last_x = np.nan
    last_z = np.nan
    last_t = np.nan
    initial_t = trial_t[0]
    for i in range(n):
        if trial_t[i] != last_t and (trial_x[i] != last_x or trial_z[i] != last_z):  # 除去时间重复和地点重复
            new_t = trial_t[i] - initial_t
            x = trial_x[i]
            z = trial_z[i]
            current_point = np.array([x, z])
            nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
            nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
            last_index = len(path) - 1
            if abs(nearest_index - last_index) <= 3: break  # 去除终点的徘徊

            save_t.append(new_t)
            save_x.append(trial_x[i])
            save_z.append(trial_z[i])
            save_o.append(trial_o[i])

            last_t = trial_t[i]
            last_x = trial_x[i]
            last_z = trial_z[i]

    #
    # # 计算速度
    # velocity = []  # cm/s
    # velocity_length = 4
    # n = len(save_x)
    # for i in range(n):  # -2 -1 0 1 2
    #     s = 0
    #     v = 0
    #     start_index = max([0, int(i-velocity_length/2)])
    #     finish_index = min([n-1, int(i+velocity_length/2)])
    #     for j in np.arange(start_index, finish_index, 1):  # 四段
    #         s += distance(save_x[j], save_z[j], save_x[j + 1], save_z[j + 1])
    #     v = float(s / (save_t[finish_index] - save_t[start_index]) / 10)  # cm/s
    #     # print(finish_index-start_index, len(np.arange(start_index, finish_index, 1)), v)
    #     velocity.append(v)

    # plt.plot(save_t, deivation, '.')
    # plt.show()

    save_n = len(save_x)
    for i in range(save_n):
        log_str = ''  # 时刻，x，z
        log_str += str(save_t[i]) + ','
        log_str += str(save_x[i]) + ',' + str(save_z[i]) + ',' + str(save_o[i])
        log_str += '\n'
        saveFile.write(log_str)
        saveFile.flush()
    print("save:", saveFigDir)
# 某一被试全部清洗数据可视化
def clean_and_plot_task3_successful_single_trial(name, show, version):
    logFilePath = 'E:/blind/data_analysis/successful_task/' + name + '/task3'
    logfileList = os.listdir(logFilePath)
    for logFileName in logfileList:
        # clean_single_trial(name, logFilePath, logFileName, version)
        newlogFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
        newlogFileName = logFileName
        saveFigPath = newlogFilePath
        saveFigName = logFileName + '.png'
        saveFigDir = saveFigPath + '/' + saveFigName
        plot_single_trial_sigment(name, newlogFilePath, newlogFileName, show, saveFigDir)


# 插值采样
def interp_single_trial(name, logFilePath, logFileName, version):
    # name 姓名缩写
    # logFilePath 原始数据所在目录
    # logFileName 原始数据的文件名
    info = logFileName.split(' ')
    if info[0] == 'A': pathName = info[1]
    else: pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath + '/' + logFileName
    path = np.load(pathFileDir)
    print("load: " + pathFileDir)

    trial_x = []
    trial_z = []
    trial_t = []
    trial_o = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(int(strs[1]))
            trial_z.append(int(strs[2]))
            if len(strs) >= 4:
                trial_o.append(int(strs[3]))
            else:
                trial_o.append(0)
    print("open: " + logFileDir)

    saveFilePath = 'E:/blind/data_analysis/task3/' + version
    if not os.path.exists(saveFilePath):
        os.mkdir(saveFilePath)
    saveFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
    if not os.path.exists(saveFilePath):
        os.mkdir(saveFilePath)
    saveFileName = logFileName
    saveFigDir = saveFilePath + '/' + saveFileName
    saveFile = open(saveFigDir, 'w')  # 如果已经存在，则覆盖之前的
    save_x = []
    save_z = []
    save_t = []
    save_o = []
    print("create:", saveFigDir)

    # 插值采样
    Fs = 50
    sample_t, sample_x = interp(trial_t, trial_x, Fs)
    sample_t, sample_z = interp(trial_t, trial_z, Fs)
    save_t = sample_t
    save_x = sample_x
    save_z = sample_z

    save_n = len(save_x)
    for i in range(save_n):
        log_str = ''  # 时刻，x，z
        log_str += str(save_t[i]) + ','
        log_str += str(save_x[i]) + ',' + str(save_z[i]) + ',' + str(0)
        log_str += '\n'
        saveFile.write(log_str)
        saveFile.flush()
    print("save:", saveFigDir)
# 某一被预处理可视化
def interp_and_plot_task3_clean_single_trial(name, show, version):
    logFilePath = 'E:/blind/data_analysis/task3/clean_0812/' + name
    logfileList = os.listdir(logFilePath)
    for logFileName in logfileList:
        info = logFileName.split(' ')
        if info[0] == 'A': pathName = info[1]
        else: pathName = info[0]

        info = logFileName.split('.')
        if info[-1] == 'png': continue
        interp_single_trial(name, logFilePath, logFileName, version)
        newlogFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
        newlogFileName = logFileName
        saveFigPath = newlogFilePath
        saveFigName = logFileName + '.png'
        saveFigDir = saveFigPath + '/' + saveFigName
        plot_single_trial_sigment(name, newlogFilePath, newlogFileName, show, saveFigDir)


# 滤波
def filt_single_trial(name, logFilePath, logFileName, version):
    # name 姓名缩写
    # logFilePath 原始数据所在目录
    # logFileName 原始数据的文件名
    info = logFileName.split(' ')
    if info[0] == 'A': pathName = info[1]
    else: pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath + '/' + logFileName
    path = np.load(pathFileDir)
    print("load: " + pathFileDir)

    trial_x = []
    trial_z = []
    trial_t = []
    trial_o = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(float(strs[1]))
            trial_z.append(float(strs[2]))
            if len(strs) >= 4:
                trial_o.append(int(strs[3]))
            else:
                trial_o.append(0)
    print("open: " + logFileDir)

    saveFilePath = 'E:/blind/data_analysis/task3/' + version
    if not os.path.exists(saveFilePath):
        os.mkdir(saveFilePath)
    saveFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
    if not os.path.exists(saveFilePath):
        os.mkdir(saveFilePath)
    saveFileName = logFileName
    saveFigDir = saveFilePath + '/' + saveFileName
    saveFile = open(saveFigDir, 'w')  # 如果已经存在，则覆盖之前的
    save_x = []
    save_z = []
    save_t = []
    save_o = []
    print("create:", saveFigDir)

    deivation = []  # cm
    deivation_sign = []
    nearest_index_list = []
    tangent_direction_list = []
    n = len(trial_x)
    for i in range(n):
        x = trial_x[i]
        z = trial_z[i]
        current_point = np.array([x, z])
        nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
        deivation.append(nearest_distance / 10)

        tangent_direction = path[nearest_index + 1] - path[nearest_index]
        normal_direction = current_point - path[nearest_index]
        sign = np.sign(tangent_direction[0] * normal_direction[1] - tangent_direction[1] * normal_direction[0])
        deivation_sign.append(nearest_distance / 10 * sign)
        nearest_index_list.append(nearest_index)
        tangent_direction_list.append(tangent_direction)

    # 滤波
    fc = 0.3
    wn = 2 * fc / 50
    b, a = signal.butter(8, wn, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    d_filt = signal.filtfilt(b, a, deivation_sign)  # data为要过滤的信号
    new_x = []
    new_z = []
    for i in range(n):
        normal_direction = np.array([-tangent_direction_list[i][1], tangent_direction_list[i][0]])
        normal_direction = normal_direction / np.linalg.norm(normal_direction)
        new_point = path[nearest_index_list[i]] + normal_direction * d_filt[i] * 10
        new_x.append(new_point[0])
        new_z.append(new_point[1])

    save_x = new_x
    save_z = new_z
    save_t = trial_t

    save_n = len(save_x)
    for i in range(save_n):
        log_str = ''  # 时刻，x，z
        log_str += str(save_t[i]) + ','
        log_str += str(save_x[i]) + ',' + str(save_z[i]) + ',' + str(0)
        log_str += '\n'
        saveFile.write(log_str)
        saveFile.flush()
    print("save:", saveFigDir)
# 滤波前后对比图
def plot_compare_trial_sigment(name, logFilePath1, logFilePath2, logFileName, show, saveFigDir=''):
    info = logFileName.split(' ')
    if info[0] == 'A':
        pathName = info[1]
    else:
        pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath1 + '/' + logFileName

    trial_x = []
    trial_z = []
    trial_t = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(float(strs[1]))
            trial_z.append(float(strs[2]))
    path = np.load(pathFileDir)
    p = path.T
    pathOutline = np.load(pathOutlineFileDir)
    pOutline = pathOutline.T
    print("open: " + logFileDir)

    # 计算偏差
    deivation = []  # cm
    deivation_sign = []
    n = len(trial_x)
    for i in range(n):
        x = trial_x[i]
        z = trial_z[i]
        current_point = np.array([x, z])
        nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
        deivation.append(nearest_distance / 10)

        tangent_direction = path[nearest_index + 1] - path[nearest_index]
        normal_direction = current_point - path[nearest_index]
        sign = np.sign(tangent_direction[0] * normal_direction[1] - tangent_direction[1] * normal_direction[0])
        deivation_sign.append(nearest_distance / 10 * sign)

    velocity = []  # cm/s
    velocity_length = 4
    n = len(trial_x)
    for i in range(n):  # -2 -1 0 1 2
        s = 0
        v = 0
        start_index = max([0, int(i - velocity_length / 2)])
        finish_index = min([n - 1, int(i + velocity_length / 2)])
        if trial_t[finish_index] - trial_t[start_index] != 0:
            for j in np.arange(start_index, finish_index, 1):  # 四段
                s += distance(trial_x[j], trial_z[j], trial_x[j + 1], trial_z[j + 1])
            v = float(s / (trial_t[finish_index] - trial_t[start_index]) / 10)
        # print(finish_index-start_index, len(np.arange(start_index, finish_index, 1)), v)
        velocity.append(v)

        # current_point = np.array([trial_x[i], trial_z[i]])
        # nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))

    max_velocity = max(velocity)
    velocity = np.array(velocity)
    bins = np.arange(0, max_velocity + 10, 2)

    fig = plt.figure(figsize=(25, 10))
    fig.suptitle(name + ' : ' + logFileName)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.3)

    # d-t图
    t_sample, d_sample = interp(trial_t, deivation_sign, 50)
    # d_sample -= np.mean(d_sample)
    plt.subplot(341)
    plt.axhline(y=0, color='k', lw=0.5)
    plt.axhline(y=30, color='gray', lw=0.5)
    plt.axhline(y=-30, color='gray', lw=0.5)
    plt.plot(t_sample, d_sample)
    plt.xlabel('time (s)')
    plt.ylabel('deivation (cm)')
    plt.yticks((-30, 0, 30))
    plt.grid()
    # print(len(t_sample))

    # d-t频谱
    Fre, FFT_d = FFT(50, d_sample)
    plt.subplot(345)
    plt.axvline(x=0.3, color='k', lw=0.5)
    plt.plot(Fre, FFT_d, lw=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('step frequency (Hz)')
    plt.ylabel('amplitude')
    plt.grid()

    # 速度分布图
    plt.subplot(349)
    plt.hist(velocity, bins)
    plt.xlabel("velocity (cm/s)")
    plt.ylabel("frequency")

    plt.subplot(143, facecolor='#111111')
    plt.axis('equal')
    ax = plt.gca()
    # 粗格子
    x_miloc = MultipleLocator(100)
    y_miloc = MultipleLocator(100)
    ax.xaxis.set_minor_locator(x_miloc)
    ax.yaxis.set_minor_locator(y_miloc)
    ax.grid(which='major', color='#555555')
    # 细格子
    x_maloc = MultipleLocator(1000)
    y_maloc = MultipleLocator(1000)
    ax.xaxis.set_major_locator(x_maloc)
    ax.yaxis.set_major_locator(y_maloc)
    ax.grid(which='minor', color='#555555', alpha=0.2)
    # 路径
    plt.plot(p[0], p[1], linewidth=1.5, color='#FFFFFF', zorder=5)
    plt.fill(pOutline[0], pOutline[1], color='#FFFFFF', zorder=3, alpha=0.2, lw=0)
    # 轨迹
    t = np.array(velocity)
    x = np.array(trial_x)
    y = np.array(trial_z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                        norm=plt.Normalize(0, 100))
    lc.set_array(t)
    lc.set_linewidth(2)
    lc.set_zorder(10)
    ax.add_collection(lc)
    cb = plt.colorbar(lc)
    cb.set_label("velocity (cm/s)")

    # figure2
    logFileDir = logFilePath2 + '/' + logFileName

    trial_x = []
    trial_z = []
    trial_t = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(float(strs[1]))
            trial_z.append(float(strs[2]))
    print("open: " + logFileDir)

    # 计算偏差
    deivation = []  # cm
    deivation_sign = []
    n = len(trial_x)
    for i in range(n):
        x = trial_x[i]
        z = trial_z[i]
        current_point = np.array([x, z])
        nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
        deivation.append(nearest_distance / 10)

        tangent_direction = path[nearest_index + 1] - path[nearest_index]
        normal_direction = current_point - path[nearest_index]
        sign = np.sign(tangent_direction[0] * normal_direction[1] - tangent_direction[1] * normal_direction[0])
        deivation_sign.append(nearest_distance / 10 * sign)

    velocity = []  # cm/s
    velocity_length = 4
    n = len(trial_x)
    for i in range(n):  # -2 -1 0 1 2
        s = 0
        v = 0
        start_index = max([0, int(i - velocity_length / 2)])
        finish_index = min([n - 1, int(i + velocity_length / 2)])
        if trial_t[finish_index] - trial_t[start_index] != 0:
            for j in np.arange(start_index, finish_index, 1):  # 四段
                s += distance(trial_x[j], trial_z[j], trial_x[j + 1], trial_z[j + 1])
            v = float(s / (trial_t[finish_index] - trial_t[start_index]) / 10)
        # print(finish_index-start_index, len(np.arange(start_index, finish_index, 1)), v)
        velocity.append(v)

        # current_point = np.array([trial_x[i], trial_z[i]])
        # nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))

    max_velocity = max(velocity)
    velocity = np.array(velocity)
    bins = np.arange(0, max_velocity + 10, 2)

    # d-t图
    t_sample, d_sample = interp(trial_t, deivation_sign, 50)
    # d_sample -= np.mean(d_sample)
    plt.subplot(342)
    plt.axhline(y=0, color='k', lw=0.5)
    plt.plot(t_sample, d_sample)
    # print(len(t_sample))
    plt.xlabel('time (s)')
    plt.ylabel('deivation (cm)')
    plt.yticks((-30, 0, 30))
    plt.grid()

    # d-t频谱
    Fre, FFT_d = FFT(50, d_sample)
    plt.subplot(346)
    plt.axvline(x=0.3, color='k', lw=0.5)
    plt.plot(Fre, FFT_d, lw=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('step frequency (Hz)')
    plt.ylabel('amplitude')
    plt.grid()

    # 速度分布图
    plt.subplot(3, 4, 10)
    plt.hist(velocity, bins)
    plt.xlabel("velocity (cm/s)")
    plt.ylabel("frequency")

    plt.subplot(144, facecolor='#111111')
    plt.axis('equal')
    ax = plt.gca()
    # 粗格子
    x_miloc = MultipleLocator(100)
    y_miloc = MultipleLocator(100)
    ax.xaxis.set_minor_locator(x_miloc)
    ax.yaxis.set_minor_locator(y_miloc)
    ax.grid(which='major', color='#555555')
    # 细格子
    x_maloc = MultipleLocator(1000)
    y_maloc = MultipleLocator(1000)
    ax.xaxis.set_major_locator(x_maloc)
    ax.yaxis.set_major_locator(y_maloc)
    ax.grid(which='minor', color='#555555', alpha=0.2)
    # 路径
    plt.plot(p[0], p[1], linewidth=1.5, color='#FFFFFF', zorder=5)
    plt.fill(pOutline[0], pOutline[1], color='#FFFFFF', zorder=3, alpha=0.2, lw=0)
    # 轨迹
    t = np.array(velocity)
    x = np.array(trial_x)
    y = np.array(trial_z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                        norm=plt.Normalize(0, 100))
    lc.set_array(t)
    lc.set_linewidth(2)
    lc.set_zorder(10)
    ax.add_collection(lc)
    cb = plt.colorbar(lc)
    cb.set_label("velocity (cm/s)")


    if show:
        plt.show()
    else:
        plt.savefig(saveFigDir)
        print("save:", saveFigDir)
        plt.close()

def filt_and_plot_task3_interp_compare_trial(name, show, version):
    logFilePath = 'E:/blind/data_analysis/task3/interp_0816/' + name
    logfileList = os.listdir(logFilePath)
    for logFileName in logfileList:
        info = logFileName.split(' ')
        if info[0] == 'A':
            pathName = info[1]
        else:
            pathName = info[0]

        if pathName == 'learn': continue
        info = logFileName.split('.')
        if info[-1] == 'png': continue
        filt_single_trial(name, logFilePath, logFileName, version)
        newlogFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
        newlogFileName = logFileName
        saveFigPath = newlogFilePath
        saveFigName = logFileName + '.png'
        saveFigDir = saveFigPath + '/' + saveFigName
        plot_compare_trial_sigment(name, logFilePath, newlogFilePath, logFileName, show, saveFigDir)

def cut_turn_and_plot_task3_filt_trial(name, version):
    logFilePath = 'E:/blind/data_analysis/task3/filt_0818/' + name
    logfileList = os.listdir(logFilePath)
    for logFileName in logfileList:
        info = logFileName.split('.')
        if info[-1] == 'png': continue
        info = logFileName.split(' ')
        if info[0] == 'A':
            pathName = info[1]
        else:
            pathName = info[0]
        info = pathName.split('_')

        logFileDir = logFilePath + '/' + logFileName
        trial_x = []
        trial_z = []
        trial_t = []
        trial_o = []
        if info[0] == 'L':
            with open(logFileDir, 'r') as f:
                for line in f:
                    strs = line.strip('\n').split(',')
                    if float(strs[2]) > 3000:
                        trial_t.append(float(strs[0]))
                        trial_x.append(float(strs[1]))
                        trial_z.append(float(strs[2]))
                        if len(strs) >= 4:
                            trial_o.append(int(strs[3]))
                        else:
                            trial_o.append(0)
        else:
            with open(logFileDir, 'r') as f:
                for line in f:
                    strs = line.strip('\n').split(',')
                    trial_t.append(float(strs[0]))
                    trial_x.append(float(strs[1]))
                    trial_z.append(float(strs[2]))
                    if len(strs) >= 4:
                        trial_o.append(int(strs[3]))
                    else:
                        trial_o.append(0)
        print("open: " + logFileDir)

        saveFilePath = 'E:/blind/data_analysis/task3/' + version
        if not os.path.exists(saveFilePath):
            os.mkdir(saveFilePath)
        saveFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
        if not os.path.exists(saveFilePath):
            os.mkdir(saveFilePath)
        saveFileName = logFileName
        saveFigDir = saveFilePath + '/' + saveFileName
        saveFile = open(saveFigDir, 'w')  # 如果已经存在，则覆盖之前的
        save_x = trial_x
        save_z = trial_z
        save_t = trial_t
        save_o = trial_o
        print("create:", saveFigDir)

        save_n = len(save_x)
        for i in range(save_n):
            log_str = ''  # 时刻，x，z
            log_str += str(save_t[i]) + ','
            log_str += str(save_x[i]) + ',' + str(save_z[i]) + ',' + str(0)
            log_str += '\n'
            saveFile.write(log_str)
            saveFile.flush()
        print("save:", saveFigDir)

        newlogFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
        newlogFileName = logFileName
        saveFigPath = newlogFilePath
        saveFigName = logFileName + '.png'
        saveFigDir = saveFigPath + '/' + saveFigName
        plot_single_trial_sigment(name, newlogFilePath, newlogFileName, False, saveFigDir)


# 分析一次行走
def analysis_single_trial(name, logFilePath, logFileName):
    info = logFileName.split(' ')
    if info[0] == 'A': pathName = info[1]
    else: pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath + '/' + logFileName
    path = np.load(pathFileDir)
    print("load: " + pathFileDir)

    trial_x = []
    trial_z = []
    trial_t = []
    trial_o = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(float(strs[1]))
            trial_z.append(float(strs[2]))
            if len(strs) >= 4:
                trial_o.append(int(strs[3]))
            else:
                trial_o.append(0)
    print("open: " + logFileDir)

    # 分析出界
    out_state = False
    out_time = 0
    out_num = 0
    out_begin_time = 0
    out_end_time = 0

    # 计算偏差
    deivation = []  # cm
    n = len(trial_x)
    for i in range(n):
        x = trial_x[i]
        z = trial_z[i]
        current_point = np.array([x, z])
        nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
        deivation.append(nearest_distance / 10)

        if not out_state:  # 界内
            if nearest_distance > 300:  # 内->外
                out_state = True
                out_begin_time = trial_t[i]
                # print('begin', out_begin_time, x, z)
        else:  # 界外
            if nearest_distance < 300 or i == n-1:  # 外->内 或 结束
                out_state = False
                out_end_time = trial_t[i]
                out_time += out_end_time - out_begin_time
                # print('out', out_end_time, x, z)
                out_num += 1

    # 计算速度
    velocity = []  # cm/s
    velocity_length = 4
    n = len(trial_x)
    for i in range(n):  # -2 -1 0 1 2
        s = 0
        v = 0
        start_index = max([0, int(i - velocity_length / 2)])
        finish_index = min([n - 1, int(i + velocity_length / 2)])
        for j in np.arange(start_index, finish_index, 1):  # 四段
            s += distance(trial_x[j], trial_z[j], trial_x[j + 1], trial_z[j + 1])
        v = float(s / (trial_t[finish_index] - trial_t[start_index]) / 10)  # cm/s
        # print(finish_index-start_index, len(np.arange(start_index, finish_index, 1)), v)
        velocity.append(v)

    mean_deivation = np.mean(deivation)
    mean_velocity = np.mean(velocity)

    # 计算总时长
    total_time = trial_t[-1] - trial_t[0]
    out_proportion = out_time/total_time

    return mean_deivation, mean_velocity, total_time, out_num, out_proportion
# 分析一次转弯
def analysis_single_trial_turn(name, logFilePath, logFileName):
    info = logFileName.split(' ')
    if info[0] == 'A': pathName = info[1]
    else: pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath + '/' + logFileName

    path = np.load(pathFileDir)
    print("load: " + pathFileDir)

    trial_x = []
    trial_z = []
    trial_t = []
    trial_o = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            if int(strs[2]) > 3000:
                trial_t.append(float(strs[0]))
                trial_x.append(int(strs[1]))
                trial_z.append(int(strs[2]))
                if len(strs) >= 4:
                    trial_o.append(int(strs[3]))
                else:
                    trial_o.append(0)
    print("open: " + logFileDir)

    # 计算偏差
    deivation = []  # cm
    n = len(trial_x)
    for i in range(n):
        x = trial_x[i]
        z = trial_z[i]
        current_point = np.array([x, z])
        nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
        deivation.append(nearest_distance / 10)

    # 计算速度
    velocity = []  # cm/s
    velocity_length = 4
    n = len(trial_x)
    for i in range(n):  # -2 -1 0 1 2
        s = 0
        v = 0
        start_index = max([0, int(i - velocity_length / 2)])
        finish_index = min([n - 1, int(i + velocity_length / 2)])
        for j in np.arange(start_index, finish_index, 1):  # 四段
            s += distance(trial_x[j], trial_z[j], trial_x[j + 1], trial_z[j + 1])
        v = float(s / (trial_t[finish_index] - trial_t[start_index]) / 10)  # cm/s
        # print(finish_index-start_index, len(np.arange(start_index, finish_index, 1)), v)
        velocity.append(v)

    mean_deivation = np.mean(deivation)
    mean_velocity = np.mean(velocity)

    return mean_deivation, mean_velocity
# 汇总实验二的分析数据
def analysis_task3_collection(names, version, memo=''):
    path_name_index_s = ['line_l9', 'S_r3.2', 'L_45', 'L_90', 'L_135', 'L_-45', 'L_-90', 'L_-135',
                      'A line_l9', 'A S_r3.2', 'A L_45', 'A L_90', 'A L_135', 'A L_-45', 'A L_-90', 'A L_-135']
    # path_name_index_s = ['L_45', 'L_90', 'L_135', 'L_-45', 'L_-90', 'L_-135',
    #                    'A L_45', 'A L_90', 'A L_135', 'A L_-45', 'A L_-90', 'A L_-135']

    mean_deivation_df = pd.DataFrame(index=path_name_index_s, columns=names)
    mean_velocity_df = pd.DataFrame(index=path_name_index_s, columns=names)
    total_time_df = pd.DataFrame(index=path_name_index_s, columns=names)
    out_proportion_df = pd.DataFrame(index=path_name_index_s, columns=names)
    out_num_df = pd.DataFrame(index=path_name_index_s, columns=names)

    for name in names:  # 某一个被试
        logFilePath = 'E:/blind/data_analysis/task3/' + version + '/' + name
        logfileList = os.listdir(logFilePath)
        for logFileName in logfileList:
            strs = logFileName.split('.')
            if strs[-1] == 'csv':
                info = logFileName.split(' ')
                if info[0] == 'A':
                    path_name_index = info[0] + ' ' + info[1]
                    path_name = info[1]
                else:
                    path_name_index = info[0]
                    path_name = info[0]
                if path_name != 'learn':
                    deivation, velocity, total_time, out_num, out_proportion = analysis_single_trial(name, logFilePath, logFileName)
                    mean_deivation_df.loc[path_name_index, name] = deivation
                    mean_velocity_df.loc[path_name_index, name] = velocity
                    total_time_df.loc[path_name_index, name] = total_time
                    out_proportion_df.loc[path_name_index, name] = out_proportion
                    out_num_df.loc[path_name_index, name] = out_num

    save_time = str(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()))
    save_name = 'task3_data_collection ' + version + ' ' + save_time + memo + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    with pd.ExcelWriter(save_dir) as writer:
        mean_deivation_df.to_excel(writer, sheet_name='mean_deivation')
        mean_velocity_df.to_excel(writer, sheet_name='mean_velocity')
        total_time_df.to_excel(writer, sheet_name='total_time')
        out_num_df.to_excel(writer, sheet_name='out_num')
        out_proportion_df.to_excel(writer, sheet_name='out_proportion')

    print('save:', save_dir)


def plot_task3_d2v_by_path(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    mean_deivation_df = pd.read_excel(save_dir, sheet_name='mean_deivation', index_col=0)
    mean_velocity_df = pd.read_excel(save_dir, sheet_name='mean_velocity', index_col=0)

    # path_list = ['line_l9', 'S_r3.2', 'L_45', 'L_90', 'L_135', 'L_-45', 'L_-90', 'L_-135',
    #                    'A line_l9', 'A S_r3.2', 'A L_45', 'A L_90', 'A L_135', 'A L_-45', 'A L_-90', 'A L_-135']
    path_list = ['line_l9']
    path_list = ['line_l9', 'S_r3.2', 'A line_l9', 'A S_r3.2']
    path_list = ['L_45', 'L_-45', 'A L_45', 'A L_-45']
    path_list = ['L_90', 'L_-90', 'A L_90', 'A L_-90']
    path_list = ['L_135', 'L_-135', 'A L_135', 'A L_-135']
    colorList = ['b', 'g', 'r', 'c', 'm', 'y']

    fig = plt.figure()
    for i in range(len(path_list)):
        path = path_list[i]
        deivation_all_names = []
        velocity_all_names = []
        for name in name_list:
            deivation_all_names.append(round(mean_deivation_df.loc[path, name], 1))
            velocity_all_names.append(round(mean_velocity_df.loc[path, name], 1))
        plt.plot(velocity_all_names, deivation_all_names, '.', label=path, c=colorList[i])
        mean_deivation = np.mean(deivation_all_names)
        plt.plot([min(velocity_all_names), max(velocity_all_names)], [mean_deivation, mean_deivation], c=colorList[i])
    plt.xlabel('mean velocity (cm/s)')
    plt.ylabel('mean deivation (cm)')
    plt.legend()
    plt.show()

def plot_task3_d2v_by_name(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    mean_deivation_df = pd.read_excel(save_dir, sheet_name='mean_deivation', index_col=0)
    mean_velocity_df = pd.read_excel(save_dir, sheet_name='mean_velocity', index_col=0)

    path_list = ['line_l9', 'S_r3.2', 'L_45', 'L_90', 'L_135', 'L_-45', 'L_-90', 'L_-135',
                 'A line_l9', 'A S_r3.2', 'A L_45', 'A L_90', 'A L_135', 'A L_-45', 'A L_-90', 'A L_-135']
    name_list = ['lmq', 'wb', 'zhy', 'xtq']

    fig = plt.figure()
    for name in name_list:
        deivation_all_paths = []
        velocity_all_paths = []
        for path in path_list:
            deivation_all_paths.append(round(mean_deivation_df.loc[path, name], 1))
            velocity_all_paths.append(round(mean_velocity_df.loc[path, name], 1))
            x0 = round(mean_velocity_df.loc[path, name], 1)
            y0 = round(mean_deivation_df.loc[path, name], 1)
            #plt.text(x0, y0, path, fontsize=7)
        plt.plot(velocity_all_paths, deivation_all_paths, '.', label=name)

    plt.xlabel('mean velocity (cm/s)')
    plt.ylabel('mean deivation (cm)')
    plt.legend()
    plt.show()

def plot_task3_d2v_by_verbal_path(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    mean_deivation_df = pd.read_excel(save_dir, sheet_name='mean_deivation', index_col=0)
    mean_velocity_df = pd.read_excel(save_dir, sheet_name='mean_velocity', index_col=0)

    # path_list = ['line_l9', 'S_r3.2', 'L_45', 'L_90', 'L_135', 'L_-45', 'L_-90', 'L_-135',
    #                    'A line_l9', 'A S_r3.2', 'A L_45', 'A L_90', 'A L_135', 'A L_-45', 'A L_-90', 'A L_-135']
    # nonverbal_path_list = ['L_45', 'L_-45']
    # nonverbal_path_list = ['L_90', 'L_-90']
    nonverbal_path_list = ['L_135', 'L_-135']
    fig = plt.figure()
    deivation_all_names = []
    velocity_all_names = []
    for i in range(len(nonverbal_path_list)):
        path = nonverbal_path_list[i]
        for name in name_list:
            deivation_all_names.append(round(mean_deivation_df.loc[path, name], 1))
            velocity_all_names.append(round(mean_velocity_df.loc[path, name], 1))
    plt.plot(velocity_all_names, deivation_all_names, '.', label=nonverbal_path_list, c='blue')
    mean_deivation = np.mean(deivation_all_names)
    plt.plot([min(velocity_all_names), max(velocity_all_names)], [mean_deivation, mean_deivation], c='b')

    verbal_path_list = []
    for str in nonverbal_path_list:
        verbal_path_list.append('A ' + str)
    deivation_all_names = []
    velocity_all_names = []
    for i in range(len(verbal_path_list)):
        path = verbal_path_list[i]
        for name in name_list:
            deivation_all_names.append(round(mean_deivation_df.loc[path, name], 1))
            velocity_all_names.append(round(mean_velocity_df.loc[path, name], 1))
    plt.plot(velocity_all_names, deivation_all_names, '.', label=verbal_path_list, c='r')
    mean_deivation = np.mean(deivation_all_names)
    plt.plot([min(velocity_all_names), max(velocity_all_names)], [mean_deivation, mean_deivation], c='r')

    plt.xlabel('mean velocity (cm/s)')
    plt.ylabel('mean deivation (cm)')
    plt.legend()
    plt.show()

def plot_task3_d2v_by_turn(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    mean_deivation_df = pd.read_excel(save_dir, sheet_name='mean_deivation', index_col=0)
    mean_velocity_df = pd.read_excel(save_dir, sheet_name='mean_velocity', index_col=0)

    # path_list = ['line_l9', 'S_r3.2', 'L_45', 'L_90', 'L_135', 'L_-45', 'L_-90', 'L_-135',
    #                    'A line_l9', 'A S_r3.2', 'A L_45', 'A L_90', 'A L_135', 'A L_-45', 'A L_-90', 'A L_-135']

    theta = [45, 90, 135]
    colorlist = ['r', 'g', 'b']
    fig = plt.figure()
    for i in range(len(theta)):
        path = 'L_' + str(theta[i])
        deivation_all_names = []
        velocity_all_names = []
        for name in name_list:
            deivation_all_names.append(round(mean_deivation_df.loc[path, name], 1))
            velocity_all_names.append(round(mean_velocity_df.loc[path, name], 1))
        path = 'L_-' + str(theta[i])
        for name in name_list:
            deivation_all_names.append(round(mean_deivation_df.loc[path, name], 1))
            velocity_all_names.append(round(mean_velocity_df.loc[path, name], 1))
        path = 'A L_' + str(theta[i])
        for name in name_list:
            deivation_all_names.append(round(mean_deivation_df.loc[path, name], 1))
            velocity_all_names.append(round(mean_velocity_df.loc[path, name], 1))
        path = 'A L_-' + str(theta[i])
        for name in name_list:
            deivation_all_names.append(round(mean_deivation_df.loc[path, name], 1))
            velocity_all_names.append(round(mean_velocity_df.loc[path, name], 1))

        mean_deivation = np.mean(deivation_all_names)
        plt.plot(velocity_all_names, deivation_all_names, '.', label='(A)'+'L_(-)'+str(theta[i]), c=colorlist[i])
        plt.plot([min(velocity_all_names), max(velocity_all_names)], [mean_deivation, mean_deivation], c=colorlist[i])


    plt.xlabel('mean velocity (cm/s)')
    plt.ylabel('mean deivation (cm)')
    plt.legend()
    plt.show()

def plot_task3_time_by_verbal_path(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    total_time_df = pd.read_excel(save_dir, sheet_name='total_time', index_col=0)
    out_proportion_df = pd.read_excel(save_dir, sheet_name='out_proportion', index_col=0)

    nonverbal_path_list = ['line_l9', 'S_r3.2', 'L_45', 'L_-45', 'L_90', 'L_-90', 'L_135', 'L_-135']
    verbal_path_list = ['A line_l9', 'A S_r3.2', 'A L_45', 'A L_-45', 'A L_90', 'A L_-90', 'A L_135', 'A L_-135']

    fig = plt.figure()
    mean_total_time = []
    mean_out_proportion = []
    for i in range(len(nonverbal_path_list)):
        path = nonverbal_path_list[i]
        total_time_all_names = []
        out_proportion_all_names = []
        for name in name_list:
            total_time_all_names.append(round(total_time_df.loc[path, name], 1))
            out_proportion_all_names.append(round(out_proportion_df.loc[path, name], 1))
        mean_total_time.append(np.mean(total_time_all_names))
        mean_out_proportion.append(np.mean(out_proportion_all_names))
    plt.bar(np.arange(8), mean_total_time, width=0.3, label='T')

    mean_total_time = []
    mean_out_proportion = []
    for i in range(len(verbal_path_list)):
        path = verbal_path_list[i]
        total_time_all_names = []
        out_proportion_all_names = []
        for name in name_list:
            total_time_all_names.append(round(total_time_df.loc[path, name], 1))
            out_proportion_all_names.append(round(out_proportion_df.loc[path, name], 1))
        mean_total_time.append(np.mean(total_time_all_names))
        mean_out_proportion.append(np.mean(out_proportion_all_names))
    plt.bar(np.arange(8)+0.3, mean_total_time, width=0.3, label='T+A')

    plt.xticks(range(8), nonverbal_path_list)
    plt.xlabel('path')
    plt.ylabel('total time (s)')
    plt.legend()
    plt.show()

def plot_task3_outpro_by_verbal_path(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    total_time_df = pd.read_excel(save_dir, sheet_name='total_time', index_col=0)
    out_proportion_df = pd.read_excel(save_dir, sheet_name='out_proportion', index_col=0)

    nonverbal_path_list = ['line_l9', 'S_r3.2', 'L_45', 'L_-45', 'L_90', 'L_-90', 'L_135', 'L_-135']
    verbal_path_list = ['A line_l9', 'A S_r3.2', 'A L_45', 'A L_-45', 'A L_90', 'A L_-90', 'A L_135', 'A L_-135']

    fig = plt.figure()
    mean_total_time = []
    mean_out_proportion = []
    for i in range(len(nonverbal_path_list)):
        path = nonverbal_path_list[i]
        total_time_all_names = []
        out_proportion_all_names = []
        for name in name_list:
            total_time_all_names.append(round(total_time_df.loc[path, name], 1))
            out_proportion_all_names.append(round(out_proportion_df.loc[path, name], 1))
        mean_total_time.append(np.mean(total_time_all_names))
        mean_out_proportion.append(np.mean(out_proportion_all_names))
    plt.bar(np.arange(8), mean_out_proportion, width=0.3, label='T')

    mean_total_time = []
    mean_out_proportion = []
    for i in range(len(verbal_path_list)):
        path = verbal_path_list[i]
        total_time_all_names = []
        out_proportion_all_names = []
        for name in name_list:
            total_time_all_names.append(round(total_time_df.loc[path, name], 1))
            out_proportion_all_names.append(round(out_proportion_df.loc[path, name], 1))
        mean_total_time.append(np.mean(total_time_all_names))
        mean_out_proportion.append(np.mean(out_proportion_all_names))
    plt.bar(np.arange(8)+0.3, mean_out_proportion, width=0.3, label='T+A')

    plt.xticks(range(8), nonverbal_path_list)
    plt.ylim([0, 0.15])
    plt.xlabel('path')
    plt.ylabel('out proportion')
    plt.legend()
    plt.show()

def plot_task3_learning_curve_deivation_by_name(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    mean_deivation_df = pd.read_excel(save_dir, sheet_name='mean_deivation', index_col=0)
    mean_velocity_df = pd.read_excel(save_dir, sheet_name='mean_velocity', index_col=0)

    colorList = ['b', 'g', 'r', 'c', 'm', 'y']
    # name_list = ['lmq', 'zlj', 'cj', 'yt']
    for i in range(len(name_list)):
        name = name_list[i]
        logFilePath = 'E:/blind/data_analysis/successful_task/' + name + '/task3'
        logfileList = os.listdir(logFilePath)
        pathName_list = []
        time_list = []
        for logFileName in logfileList:
            info = logFileName.split(' ')
            if info[0] == 'A':
                pathName = 'A ' + info[1]
                time = info[3]
            else:
                pathName = info[0]
                time = info[2]
            if pathName == 'learn' or pathName == 'A learn': continue
            pathName_list.append(pathName)
            timeinfo = time.split('.')
            hms = timeinfo[0] + timeinfo[1] + timeinfo[2]
            hms = int(hms)
            time_list.append(hms)
        order = pd.DataFrame(index=time_list)
        order['path'] = pathName_list
        order = order.sort_index(axis=0)
        path_order = order['path']

        order_list = np.arange(1, 17, 1)
        path_order_list = []
        deivation_order_list = []
        mean_deivation_order_list = []
        for path in path_order:
            path_order_list.append(path)
            deivation_order_list.append(mean_deivation_df.loc[path, name])
            mean_deivation_order_list.append(np.mean(deivation_order_list))
        color = 'b'
        plt.figure(figsize=(12, 8))
        plt.plot(path_order_list, deivation_order_list, '.', c=color, label=name)
        plt.plot(path_order_list, mean_deivation_order_list, '--', c=color, label=name+'_mean')
        plt.legend()
        plt.title(name + ' deivation - learning curve')

        save_name = name + '.png'
        save_dir = 'E:/blind/data_analysis/task3/' + version + '/learning_curve/deivation_single_name/' + save_name
        plt.savefig(save_dir)
        print('save:', save_dir)
        # plt.show()
        plt.close()


    # plt.legend()
    # plt.show()
    # print(order)

def plot_task3_learning_curve_outpro_by_name(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    out_proportion_df = pd.read_excel(save_dir, sheet_name='out_proportion', index_col=0)

    colorList = ['b', 'g', 'r', 'c', 'm', 'y']

    name_list = ['lmq', 'zlj', 'cj', 'yt', 'lcy', 'wb', 'yxy', 'yhr', 'lx', 'lzz', 'txl', 'xtq', 'cfq', 'zhy', 'wjx', 'hjs',
             'jlm', 'wxm', 'zdd', 'lxue']
    # name_list = ['lmq', 'zlj', 'cj', 'yt']
    for i in range(len(name_list)):
        name = name_list[i]
        logFilePath = 'E:/blind/data_analysis/successful_task/' + name + '/task3'
        logfileList = os.listdir(logFilePath)
        pathName_list = []
        time_list = []
        for logFileName in logfileList:
            info = logFileName.split(' ')
            if info[0] == 'A':
                pathName = 'A ' + info[1]
                time = info[3]
            else:
                pathName = info[0]
                time = info[2]
            if pathName == 'learn' or pathName == 'A learn': continue
            pathName_list.append(pathName)
            timeinfo = time.split('.')
            hms = timeinfo[0] + timeinfo[1] + timeinfo[2]
            hms = int(hms)
            time_list.append(hms)
        order = pd.DataFrame(index=time_list)
        order['path'] = pathName_list
        order = order.sort_index(axis=0)
        path_order = order['path']

        order_list = np.arange(1, 17, 1)
        path_order_list = []
        out_proportion_order_list = []
        mean_out_proportion_order_list = []
        for path in path_order:
            path_order_list.append(path)
            out_proportion_order_list.append(out_proportion_df.loc[path, name])
            mean_out_proportion_order_list.append(np.mean(out_proportion_order_list))
        color = 'g'
        plt.figure(figsize=(12, 8))
        plt.plot(path_order_list, out_proportion_order_list, '.', c=color, label=name)
        plt.plot(path_order_list, mean_out_proportion_order_list, '--', c=color, label=name+'_mean')
        plt.legend()
        plt.title(name + ' out_proportion - learning curve')

        save_name = name + '.png'
        save_dir = 'E:/blind/data_analysis/task3/' + version + '/learning_curve/out_proportion_single_name/' + save_name
        plt.savefig(save_dir)
        print('save:', save_dir)
        # plt.show()
        plt.close()


    # plt.legend()
    # plt.show()
    # print(order)

def plot_task3_learning_curve_deivation_collection(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    mean_deivation_df = pd.read_excel(save_dir, sheet_name='mean_deivation', index_col=0)
    mean_velocity_df = pd.read_excel(save_dir, sheet_name='mean_velocity', index_col=0)
    out_proportion_df = pd.read_excel(save_dir, sheet_name='out_proportion', index_col=0)

    colorList = ['b', 'g', 'r', 'c', 'm', 'y']

    # name_list = ['lmq', 'zlj', 'cj', 'yt']
    plt.figure(figsize=(12, 8))
    deivation_order_df = pd.DataFrame()
    order_list = np.arange(1, 17, 1)

    for i in range(len(name_list)):
        name = name_list[i]
        logFilePath = 'E:/blind/data_analysis/successful_task/' + name + '/task3'
        logfileList = os.listdir(logFilePath)
        pathName_list = []
        time_list = []
        for logFileName in logfileList:
            info = logFileName.split(' ')
            if info[0] == 'A':
                pathName = 'A ' + info[1]
                time = info[3]
            else:
                pathName = info[0]
                time = info[2]
            if pathName == 'learn' or pathName == 'A learn': continue
            pathName_list.append(pathName)
            timeinfo = time.split('.')
            hms = timeinfo[0] + timeinfo[1] + timeinfo[2]
            hms = int(hms)
            time_list.append(hms)
        order = pd.DataFrame(index=time_list)
        order['path'] = pathName_list
        order = order.sort_index(axis=0)
        path_order = order['path']

        path_order_list = []
        deivation_order_list = []
        mean_deivation_order_list = []
        for path in path_order:
            path_order_list.append(path)
            deivation_order_list.append(mean_deivation_df.loc[path, name])
            mean_deivation_order_list.append(np.mean(deivation_order_list))
        deivation_order_df[name] = mean_deivation_order_list
        plt.plot(order_list, mean_deivation_order_list, '-', label=name, lw=0.5)

    mean_d = []
    for i in deivation_order_df.index:
        mean_d.append(np.mean(deivation_order_df.loc[i]))
    plt.plot(order_list, mean_d, label='mean')
    plt.legend()
    plt.title('deivation - learning curve')
    plt.show()

    # save_name = name + ' deivation' + '.png'
    # save_dir = 'E:/blind/data_analysis/task3/' + version + '/learning_curve/' + save_name
    # plt.savefig(save_dir)
    # print('save:', save_dir)
    # plt.close()


    # plt.legend()
    # plt.show()
    # print(order)

def plot_task3_learning_curve_outpro_collection(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    out_proportion_df = pd.read_excel(save_dir, sheet_name='out_proportion', index_col=0)

    colorList = ['b', 'g', 'r', 'c', 'm', 'y']

    # name_list = ['lmq', 'zlj', 'cj', 'yt']
    plt.figure(figsize=(12, 8))
    out_proportion_order_df = pd.DataFrame()
    order_list = np.arange(1, 17, 1)

    for i in range(len(name_list)):
        name = name_list[i]
        logFilePath = 'E:/blind/data_analysis/successful_task/' + name + '/task3'
        logfileList = os.listdir(logFilePath)
        pathName_list = []
        time_list = []
        for logFileName in logfileList:
            info = logFileName.split(' ')
            if info[0] == 'A':
                pathName = 'A ' + info[1]
                time = info[3]
            else:
                pathName = info[0]
                time = info[2]
            if pathName == 'learn' or pathName == 'A learn': continue
            pathName_list.append(pathName)
            timeinfo = time.split('.')
            hms = timeinfo[0] + timeinfo[1] + timeinfo[2]
            hms = int(hms)
            time_list.append(hms)
        order = pd.DataFrame(index=time_list)
        order['path'] = pathName_list
        order = order.sort_index(axis=0)
        path_order = order['path']

        path_order_list = []
        out_proportion_order_list = []
        mean_out_proportion_order_list = []
        for path in path_order:
            path_order_list.append(path)
            out_proportion_order_list.append(out_proportion_df.loc[path, name])
            mean_out_proportion_order_list.append(np.mean(out_proportion_order_list))
        out_proportion_order_df[name] = mean_out_proportion_order_list
        plt.plot(order_list, mean_out_proportion_order_list, '-', label=name, lw=0.5)

    mean_d = []
    for i in out_proportion_order_df.index:
        mean_d.append(np.mean(out_proportion_order_df.loc[i]))
    plt.plot(order_list, mean_d, label='mean')
    plt.legend()
    plt.title('out_proportion - learning curve')
    plt.show()

    # save_name = name + ' deivation' + '.png'
    # save_dir = 'E:/blind/data_analysis/task3/' + version + '/learning_curve/' + save_name
    # plt.savefig(save_dir)
    # print('save:', save_dir)
    # plt.close()


    # plt.legend()
    # plt.show()
    # print(order)

def plot_task3_learning_curve_velocity_collection(name_list, version, save_time):
    save_name = 'task3_data_collection ' + version + ' ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task3/' + version + '/' + save_name

    mean_velocity_df = pd.read_excel(save_dir, sheet_name='mean_velocity', index_col=0)

    colorList = ['b', 'g', 'r', 'c', 'm', 'y']
    # name_list = ['lmq', 'zlj', 'cj', 'yt']
    plt.figure(figsize=(12, 8))
    velocity_order_df = pd.DataFrame()
    order_list = np.arange(1, 17, 1)

    for i in range(len(name_list)):
        name = name_list[i]
        logFilePath = 'E:/blind/data_analysis/successful_task/' + name + '/task3'
        logfileList = os.listdir(logFilePath)
        pathName_list = []
        time_list = []
        for logFileName in logfileList:
            info = logFileName.split(' ')
            if info[0] == 'A':
                pathName = 'A ' + info[1]
                time = info[3]
            else:
                pathName = info[0]
                time = info[2]
            if pathName == 'learn' or pathName == 'A learn': continue
            pathName_list.append(pathName)
            timeinfo = time.split('.')
            hms = timeinfo[0] + timeinfo[1] + timeinfo[2]
            hms = int(hms)
            time_list.append(hms)
        order = pd.DataFrame(index=time_list)
        order['path'] = pathName_list
        order = order.sort_index(axis=0)
        path_order = order['path']

        path_order_list = []
        velocity_order_list = []
        mean_velocity_order_list = []
        for path in path_order:
            path_order_list.append(path)
            velocity_order_list.append(mean_velocity_df.loc[path, name])
            mean_velocity_order_list.append(np.mean(velocity_order_list))
        velocity_order_df[name] = mean_velocity_order_list
        plt.plot(order_list, mean_velocity_order_list, '-', label=name, lw=0.5)

    mean_d = []
    for i in velocity_order_df.index:
        mean_d.append(np.mean(velocity_order_df.loc[i]))
    plt.plot(order_list, mean_d, label='mean')
    plt.legend()
    plt.title('velocity - learning curve')
    plt.show()

    # save_name = name + ' deivation' + '.png'
    # save_dir = 'E:/blind/data_analysis/task3/' + version + '/learning_curve/' + save_name
    # plt.savefig(save_dir)
    # print('save:', save_dir)
    # plt.close()


    # plt.legend()
    # plt.show()
    # print(order)


def plot_deviation_distribution_task3(name_list):
    deivation = []  # cm

    for name in name_list:
        logFilePath = 'E:/blind/data_analysis/task3/filt_0818/' + name
        logfileList = os.listdir(logFilePath)

        for logFileName in logfileList:
            info = logFileName.split(' ')
            if info[0] == 'A':
                pathName = info[1]
            else:
                pathName = info[0]

            if pathName == 'learn': continue
            info = logFileName.split('.')
            if info[-1] == 'png': continue

            pathFileDir = 'path/' + pathName + '.npy'
            pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
            logFileDir = logFilePath + '/' + logFileName

            trial_x = []
            trial_z = []
            trial_t = []
            with open(logFileDir, 'r') as f:
                for line in f:
                    strs = line.strip('\n').split(',')
                    trial_t.append(float(strs[0]))
                    trial_x.append(float(strs[1]))
                    trial_z.append(float(strs[2]))
            path = np.load(pathFileDir)
            p = path.T
            pathOutline = np.load(pathOutlineFileDir)
            pOutline = pathOutline.T
            print("open: " + logFileDir)

            # 计算偏差

            deivation_sign = []
            n = len(trial_x)
            for i in range(n):
                x = trial_x[i]
                z = trial_z[i]
                current_point = np.array([x, z])
                nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
                nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
                deivation.append(nearest_distance / 10)

                tangent_direction = path[nearest_index + 1] - path[nearest_index]
                normal_direction = current_point - path[nearest_index]
                sign = np.sign(tangent_direction[0] * normal_direction[1] - tangent_direction[1] * normal_direction[0])
                deivation_sign.append(nearest_distance / 10 * sign)

    name = 'all'
    print('total data count:', len(deivation))
    max_deivation = max(deivation)
    bins = np.arange(0, max_deivation + 4, 0.4)
        # fig = plt.figure(figsize=(15, 10))
    plt.title(name + ' : ' + 'deviation distribution')

    plt.hist(deivation, bins, density=True)
    plt.xlabel("deviation (cm)")
    plt.ylabel("frequency density")

    saveFigPath = 'E:/blind/data_analysis/task3/filt_0818/deviation_distribution'
    saveFigName = name + '.png'
    saveFigDir = saveFigPath + '/' + saveFigName
    plt.savefig(saveFigDir)
    print('save:', saveFigDir)
    plt.close()


    rd = pd.DataFrame(np.array(deivation))

    save_dir = 'E:/blind/data_analysis/task3/filt_0818/deviation_distribution/deviation.csv'
    with pd.ExcelWriter(save_dir) as writer:
        rd.to_csv(writer)
    print('save:', save_dir)
    # Calculate all the desired values
    df = pd.DataFrame({'mean': rd.mean(), 'median': rd.median(),
                       '25%': rd.quantile(0.25), '50%': rd.quantile(0.5),
                       '75%': rd.quantile(0.75)})
    # And plot it
    df.plot()
    print(df)
        # plt.show()

def plot_deviation_distribution_from_file():
    save_dir = 'E:/blind/data_analysis/task3/filt_0818/deviation_distribution/deviation.xlsx'

    deivation = pd.read_excel(save_dir, index_col=0)
    print('open:', save_dir)

    df = pd.DataFrame({'mean': deivation.mean(), 'median': deivation.median(),
                       '50%': deivation.quantile(0.5),
                       '90%': deivation.quantile(0.90),
                       '93.7%': deivation.quantile(0.937),
                       '95%': deivation.quantile(0.95),
                       '99.9%': deivation.quantile(0.999)})
    print(df)

    deivation = deivation.iloc[:,0]
    name = 'all'
    print('total data count:', len(deivation))
    max_deivation = max(deivation)
    bins = np.arange(0, max_deivation + 4, 0.4)
        # fig = plt.figure(figsize=(15, 10))
    plt.title(name + ' : ' + 'deviation distribution on time')

    plt.hist(deivation, bins, density=True)
    plt.xlabel("deviation (cm)")
    plt.ylabel("frequency density")

    # saveFigPath = 'E:/blind/data_analysis/task3/filt_0818/deviation_distribution'
    # saveFigName = name + '.png'
    # saveFigDir = saveFigPath + '/' + saveFigName
    # plt.savefig(saveFigDir)
    # print('save:', saveFigDir)
    plt.show()




def analysis_task2(name):
    logFilePath = 'experiment/' + name + '/task2'
    logfileList = os.listdir(logFilePath)

    for logFileName in logfileList:
        logFileDir = logFilePath + '/' + logFileName
        print("open: " + logFileDir)

        target = []
        real = []
        with open(logFileDir, 'r') as f:
            for line in f:
                strs = line.strip('\n').split(',')
                target.append(int(strs[0]))
                real.append(int(strs[2]) - int(strs[1]))
        r = 1.0
        rs = np.array([r] * len(real))

        target = np.array(target) / 180 * math.pi
        real = np.array(real) / 180 * math.pi

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        ax.plot(target, rs, '+')
        ax.plot(real, rs, '.')
        plt.title(name + ' : ' + logFileName)

        saveFidDir = 'experiment/' + name + '/analysis'
        if not os.path.exists(saveFidDir):
            os.mkdir(saveFidDir)

        saveFidDir = 'experiment/' + name + '/analysis/task2'
        if not os.path.exists(saveFidDir):
            os.mkdir(saveFidDir)

        saveFigPath = saveFidDir + '/' + name + ' task2 ' + logFileName + '.png'
        plt.savefig(saveFigPath)
        print("save:", saveFigPath)
        #plt.show()
        plt.close()

def analysis_task2_collection(names):
    theta_target_s = np.hstack((np.arange(-165, 0, 15), np.arange(15, 181, 15)))
    theta_real_df = pd.DataFrame(index=theta_target_s)
    dtheta_df = pd.DataFrame(index=theta_target_s)
    time_df = pd.DataFrame(index=theta_target_s)

    for name in names:  # 某一个被试
        analysis_task2_collection_add(name, theta_real_df, dtheta_df, time_df)

    save_time = str(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()))
    save_name = 'task2_data_collection ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task2/' + save_name

    with pd.ExcelWriter(save_dir) as writer:
        theta_real_df.to_excel(writer, sheet_name='theta_real')
        dtheta_df.to_excel(writer, sheet_name='dtheta')
        time_df.to_excel(writer, sheet_name='time')

def analysis_task2_collection_add(name, theta_real_df, dtheta_df, time_df):
    logFilePath = 'E:/blind/data_analysis/successful_task/' + name + '/task2'
    logfileList = os.listdir(logFilePath)
    for logFileName in logfileList:
        logFileDir = logFilePath + '/' + logFileName
        print("open: " + logFileDir)
        theta_target_list = []  # theta_target
        theta_real_list = []  # theta_real
        dtheta_list = []  # delta_theta = theta_real - theta_target
        time_list = []  # time
        with open(logFileDir, 'r') as f:
            for line in f:
                strs = line.strip('\n').split(',')

                theta_target = int(strs[0])
                theta_target_list.append(theta_target)

                theta_real = int(strs[2]) - int(strs[1])
                theta_real = theta_real % 360
                if theta_real > 180: theta_real -= 360
                theta_real_list.append(theta_real)

                dtheta = theta_real - theta_target
                dtheta = dtheta % 360
                if dtheta > 180: dtheta -= 360
                dtheta_list.append(dtheta)

                time_list.append(round(float(strs[3]), 3))

        if len(time_list) == 23:  # 完整实验
            print("add:", logFileDir)
            temp_df = pd.DataFrame({'theta_real': theta_real_list,
                                    'dtheta': dtheta_list,
                                    'time': time_list},
                                   index=theta_target_list)
            theta_real_df[name] = pd.Series(temp_df['theta_real'], index=temp_df.index)
            dtheta_df[name] = pd.Series(temp_df['dtheta'], index=temp_df.index)
            time_df[name] = pd.Series(temp_df['time'], index=temp_df.index)

def plot_task2_collection(save_time):
    save_name = 'task2_data_collection ' + save_time + '.xlsx'
    save_dir = 'E:/blind/data_analysis/task2/' + save_name

    theta_real_df = pd.read_excel(save_dir, sheet_name='theta_real', index_col=0)
    name_list = ['lmq', 'zlj', 'cj', 'yt', 'lcy', 'wb', 'yxy', 'yhr', 'lx', 'lzz', 'txl', 'xtq', 'cfq', 'zhy', 'wjx', 'hjs',
             'jlm', 'wxm', 'zdd', 'lxue']

    theta_target_s = np.hstack((np.arange(-165, 0, 15), np.arange(15, 181, 15)))
    plt.figure(figsize=(10, 12))
    r = 1.0
    rs = np.array([r] * len(name_list))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 15), ('0', '15', '30', '45', '60', '75', '90',
                                              '105', '120', '135', '150', '165', '180',
                                              '-165', '-150', '-135', '-120', '-105',
                                              '-90', '-75', '-60', '-45', '-30', '-15'))

    for target_theta in theta_real_df.index:
        theta_real_all_name = theta_real_df.loc[target_theta, name_list]
        theta_real_s = theta_real_all_name / 180 * math.pi
        ax.plot(target_theta / 180 * math.pi, r, '+', c='b')
        ax.plot(theta_real_s, rs, '.', alpha=0.3, label=target_theta)

    #ax.legend(loc='right', bbox_to_anchor=(1.3, 0.5))
    plt.show()


def flat_task2_collection():
    save_dir = 'E:/blind/data_analysis/task2/' + '0825 角度-时间.xlsx'

    abs_dtheta_df = pd.read_excel(save_dir, sheet_name='data', index_col=0)
    print(abs_dtheta_df)

    flat_df = pd.DataFrame(columns=('index', 'data'))
    tag_list = []
    tag_number = np.arange(15, 166, 15)
    for i in tag_number:
        tag_list.append('N' + str(i))
    tag_list.reverse()
    tag_number = np.arange(15, 181, 15)
    for i in tag_number:
        tag_list.append('P' + str(i))
    tag_list = np.arange(0, 23, 1)
    tag_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    tag_list_length = len(tag_list)
    for i in range(tag_list_length):
        tag = tag_list[i]
        data_list = abs_dtheta_df.iloc[:, i]
        for data in data_list:
            new_data = pd.Series([tag, data], index=['index', 'data'])
            flat_df = flat_df.append(new_data, ignore_index=True)
    print(flat_df)

    save_dir = save_dir = 'E:/blind/data_analysis/task2/' + '0825 上下-时间 flat.xlsx'
    with pd.ExcelWriter(save_dir) as writer:
        flat_df.to_excel(writer, sheet_name='dtheta')




def analysis_single_trial2(name, logFilePath, logFileName):
    info = logFileName.split(' ')
    if info[0] == 'A': pathName = info[1]
    else: pathName = info[0]
    pathFileDir = 'path/' + pathName + '.npy'
    pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
    logFileDir = logFilePath + '/' + logFileName
    path = np.load(pathFileDir)
    p = path.T
    pathOutline = np.load(pathOutlineFileDir)
    pOutline = pathOutline.T
    print("load: " + pathFileDir)

    trial_x = []
    trial_z = []
    trial_t = []
    trial_o = []
    with open(logFileDir, 'r') as f:
        for line in f:
            strs = line.strip('\n').split(',')
            trial_t.append(float(strs[0]))
            trial_x.append(int(strs[1]))
            trial_z.append(int(strs[2]))
            if len(strs) >= 4:
                trial_o.append(int(strs[3]))
            else:
                trial_o.append(0)
    print("open: " + logFileDir)

    # 分析出界
    out_state = False
    out_time = 0
    out_num = 0
    out_begin_time = 0
    out_end_time = 0

    # 计算偏差
    deivation = []  # cm
    deivation_sign = []
    nearest_index_list = []
    normal_direction_list = []
    n = len(trial_x)
    for i in range(n):
        x = trial_x[i]
        z = trial_z[i]
        current_point = np.array([x, z])
        nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
        deivation.append(nearest_distance / 10)

        tangent_direction = path[nearest_index + 1] - path[nearest_index]
        normal_direction = current_point - path[nearest_index]
        sign = np.sign(tangent_direction[0] * normal_direction[1] - tangent_direction[1] * normal_direction[0])
        deivation_sign.append(nearest_distance / 10 * sign)
        nearest_index_list.append(nearest_index)
        normal_direction_list.append(normal_direction)

        if not out_state:  # 界内
            if nearest_distance > 300:  # 内->外
                out_state = True
                out_begin_time = trial_t[i]
                # print('begin', out_begin_time, x, z)
        else:  # 界外
            if nearest_distance < 300 or i == n-1:  # 外->内 或 结束
                out_state = False
                out_end_time = trial_t[i]
                out_time += out_end_time - out_begin_time
                # print('out', out_end_time, x, z)
                out_num += 1

    # d-t图
    t_sample, d_sample = interp(trial_t, deivation_sign, 50)
    # d_sample -= np.mean(d_sample)
    plt.subplot(221)
    plt.axhline(y=0, color='k', lw=0.5)
    plt.plot(t_sample, d_sample)
    #print(len(t_sample))

    # d-t频谱
    Fre, FFT_d = FFT(50, d_sample)
    plt.subplot(223)
    plt.plot(Fre, FFT_d, lw=1)
    plt.xscale('log')
    plt.yscale('log')

    wn = 2 * 0.5 / 50
    b, a = signal.butter(8, wn, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    d_filt = signal.filtfilt(b, a, d_sample)  # data为要过滤的信号

    plt.subplot(222)
    plt.axhline(y=0, color='k', lw=0.5)
    plt.plot(t_sample, d_filt)

    Fre, FFT_d = FFT(50, d_filt)
    plt.subplot(224)
    plt.plot(Fre, FFT_d, lw=1)
    plt.xscale('log')
    plt.yscale('log')

    plt.show()
    plt.close()

    new_x = []
    new_z = []
    for i in range(n):
        x = trial_x[i]
        z = trial_z[i]
        current_point = np.array([x, z])
        nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
        nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))

        new_point = path[nearest_index_list[i]] + normal_direction_list[i] * d_sample[i] / np.linalg.norm(normal_direction_list[i]) * 10
        new_x.append(new_point[0])
        new_z.append(new_point[1])

    # 计算速度
    velocity = []  # cm/s
    velocity_length = 4
    n = len(trial_x)
    for i in range(n):  # -2 -1 0 1 2
        s = 0
        v = 0
        start_index = max([0, int(i - velocity_length / 2)])
        finish_index = min([n - 1, int(i + velocity_length / 2)])
        for j in np.arange(start_index, finish_index, 1):  # 四段
            s += distance(trial_x[j], trial_z[j], trial_x[j + 1], trial_z[j + 1])
        v = float(s / (trial_t[finish_index] - trial_t[start_index]) / 10)  # cm/s
        # print(finish_index-start_index, len(np.arange(start_index, finish_index, 1)), v)
        velocity.append(v)

    mean_deivation = np.mean(deivation)
    mean_velocity = np.mean(velocity)

    # 计算总时长
    total_time = trial_t[-1] - trial_t[0]
    out_proportion = out_time/total_time

    # return mean_deivation, mean_velocity, total_time, out_num, out_proportion

    max_velocity = max(velocity)
    velocity = np.array(velocity)
    bins = np.arange(0, max_velocity + 10, 2)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(name + ' : ' + logFileName)

    plt.subplot(121)
    plt.hist(velocity, bins)
    plt.xlabel("velocity (cm/s)")
    plt.ylabel("frequency")

    plt.subplot(122, facecolor='#111111')
    plt.axis('equal')
    ax = plt.gca()
    # 粗格子
    x_miloc = MultipleLocator(100)
    y_miloc = MultipleLocator(100)
    ax.xaxis.set_minor_locator(x_miloc)
    ax.yaxis.set_minor_locator(y_miloc)
    ax.grid(which='major', color='#555555')
    # 细格子
    x_maloc = MultipleLocator(1000)
    y_maloc = MultipleLocator(1000)
    ax.xaxis.set_major_locator(x_maloc)
    ax.yaxis.set_major_locator(y_maloc)
    ax.grid(which='minor', color='#555555', alpha=0.2)
    # 路径
    plt.plot(p[0], p[1], linewidth=1.5, color='#FFFFFF', zorder=5)
    plt.fill(pOutline[0], pOutline[1], color='#FFFFFF', zorder=3, alpha=0.2, lw=0)
    # 轨迹
    t = np.array(velocity)
    x = np.array(trial_x)
    y = np.array(trial_z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                        norm=plt.Normalize(0, 100))
    lc.set_array(t)
    lc.set_linewidth(2)
    lc.set_zorder(10)
    ax.add_collection(lc)
    cb = plt.colorbar(lc)
    cb.set_label("velocity (cm/s)")

    plt.show()

def analysis_single_trial3(name):
    logFilePath = 'E:/blind/data_analysis/task3/clean_0812/' + name
    logfileList = os.listdir(logFilePath)
    global_t = []
    global_d = []

    t_last = 0
    d_last = 0
    for logFileName in logfileList:
        info = logFileName.split(' ')
        if info[0] == 'A': pathName = info[1]
        else: pathName = info[0]

        if pathName == 'learn': continue
        info = logFileName.split('.')
        if info[-1] == 'png': continue

        pathFileDir = 'path/' + pathName + '.npy'
        pathOutlineFileDir = 'path/path_outline/' + pathName + '.npy'
        logFileDir = logFilePath + '/' + logFileName
        path = np.load(pathFileDir)
        p = path.T
        pathOutline = np.load(pathOutlineFileDir)
        pOutline = pathOutline.T
        print("load: " + pathFileDir)

        trial_x = []
        trial_z = []
        trial_t = []
        trial_o = []
        with open(logFileDir, 'r') as f:
            for line in f:
                strs = line.strip('\n').split(',')
                trial_t.append(float(strs[0]))
                trial_x.append(int(strs[1]))
                trial_z.append(int(strs[2]))
                if len(strs) >= 4:
                    trial_o.append(int(strs[3]))
                else:
                    trial_o.append(0)
        print("open: " + logFileDir)

        # 分析出界
        out_state = False
        out_time = 0
        out_num = 0
        out_begin_time = 0
        out_end_time = 0

        # 计算偏差
        deivation = []  # cm
        deivation_sign = []
        n = len(trial_x)
        for i in range(n):
            x = trial_x[i]
            z = trial_z[i]
            current_point = np.array([x, z])
            nearest_distance = np.min(np.linalg.norm(path[:] - current_point, axis=1))
            nearest_index = np.argmin(np.linalg.norm(path[:] - current_point, axis=1))
            deivation.append(nearest_distance / 10)

            tangent_direction = path[nearest_index + 1] - path[nearest_index]
            normal_direction = current_point - path[nearest_index]
            sign = np.sign(tangent_direction[0] * normal_direction[1] - tangent_direction[1] * normal_direction[0])
            deivation_sign.append(nearest_distance / 10 * sign)

            if not out_state:  # 界内
                if nearest_distance > 300:  # 内->外
                    out_state = True
                    out_begin_time = trial_t[i]
                    # print('begin', out_begin_time, x, z)
            else:  # 界外
                if nearest_distance < 300 or i == n-1:  # 外->内 或 结束
                    out_state = False
                    out_end_time = trial_t[i]
                    out_time += out_end_time - out_begin_time
                    # print('out', out_end_time, x, z)
                    out_num += 1

        t_sample, d_sample = interp(trial_t, deivation_sign, 50)
        t_sample += t_last
        d_sample += d_last
        global_t.extend(t_sample)
        global_d.extend(d_sample)
        t_last = t_sample[-1]
        d_last = d_sample[-1]

    global_d -= np.mean(global_d)
    plt.subplot(211)
    plt.plot(global_t, global_d)
    plt.axhline(y=0, color='k')

    Fre, FFT_d = FFT(50, global_d)
    plt.subplot(212)
    plt.plot(Fre, FFT_d)
    plt.xscale('log')
    #plt.yscale('log')

    plt.show()

if __name__ == "__main__":
    names = ['lmq', 'zlj', 'cj', 'yt', 'lcy', 'wb', 'yxy', 'yhr', 'lx', 'lzz', 'txl', 'xtq', 'cfq', 'zhy', 'wjx', 'hjs',
             'jlm', 'wxm', 'zdd', 'lxue']

    names = ['lmq', 'zlj', 'cj', 'yt', 'lcy', 'wb', 'yxy', 'yhr', 'lx', 'lzz', 'txl', 'xtq', 'cfq', 'zhy', 'wjx', 'hjs',
             'wxm', 'lxue']
    # names = ['xtq', 'lzz']
    # for name in names:
    # #     # plot_task3_original_single_trial(name, False)
    # #     # clean_and_plot_task3_successful_single_trial(name, False, 'clean_0812')
    #     filt_and_plot_task3_interp_compare_trial(name, False, 'filt_0818')
    #     cut_turn_and_plot_task3_filt_trial(name, 'filt_cut_0818')

    # analysis_task3_collection(names, 'filt_0818')
    # analysis_task3_collection(names, 'filt_0818')
    #
    # version = 'filt_0818'
    # time = '2020-08-18 21.30.06'
    # plot_task3_d2v_by_path(names, 'interp_0816', '2020-08-16 21.55.58')
    # plot_task3_d2v_by_name(names, 'interp_0816', '2020-08-16 21.55.58')
    # plot_task3_d2v_by_verbal_path(names, 'interp_0816', '2020-08-16 21.55.58')
    # plot_task3_time_by_verbal_path(names, 'interp_0816', '2020-08-16 21.55.58')
    # plot_task3_outpro_by_verbal_path(names, 'interp_0816', '2020-08-16 21.55.58')

    # plot_task3_d2v_by_path(names, version, time)
    # plot_task3_d2v_by_name(names, version, time)
    # plot_task3_d2v_by_verbal_path(names, version, time)
    # plot_task3_time_by_verbal_path(names, version, time)
    # plot_task3_outpro_by_verbal_path(names, version, time)
    # plot_task3_learning_curve_velocity_collection(names, version, time)
    # plot_task3_learning_curve_deivation_collection(names, version, time)
    # plot_task3_learning_curve_outpro_collection(names, version, time)
    # plot_task3_learning_curve_deivation_by_name(names, version, time)

    # analysis_task2_collection(names)
    # plot_task2_collection('2020-08-15 20.38.01')


    # logFilePath = 'E:/blind/data_analysis/task3/clean_0812/' + name
    # logfileList = os.listdir(logFilePath)
    # # analysis_single_trial2(name, logFilePath, 'L_-90 2020-08-14 15.46.27.csv')
    # analysis_single_trial2(name, logFilePath, 'A S_r3.2 2020-08-14 15.19.01.csv')
    # # plot_single_trial_sigment(name, logFilePath, 'A S_r3.2 2020-08-14 15.19.01.csv', True)

    # analysis_single_trial3('zdd')
    # filt_and_plot_task3_interp_compare_trial('lxue', False, 'filt_0816')
    # flat_task2_collection()
    # plot_deviation_distribution_task3(names)
    plot_deviation_distribution_from_file()


