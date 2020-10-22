import numpy as np
import pylab as pl
import math

p_d = 10  # 1cm


def line(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    length = np.linalg.norm(p2-p1)  # 两点间的距离
    ds = np.linspace(0, length, int(length/p_d))
    points = np.matmul(ds.reshape((ds.shape[0],1)),(p2-p1).reshape((1,2)))/length+p1
    return points[1:]


def arc(p1,r,theta,gamma):  # gamma逆时针旋转角度，theta圆弧角度
    p1 = np.array(p1)
    length = r*theta  # 弧长
    ds = np.linspace(0,length,abs(int(length/p_d)))
    delta_theta = ds/r
    zs = r*np.sin(delta_theta)
    xs = r*(1-np.cos(delta_theta))
    points = np.vstack((xs,zs)).T
    rotation = np.array([[math.cos(gamma),-math.sin(gamma)],[math.sin(gamma),math.cos(gamma)]])
    new_points = []
    for p in points:
        new_points.append(np.matmul(rotation,p)+p1)
    return np.array(new_points)[1:]


'''
path = np.array([[0,0]])
path = np.vstack((path, line(path[-1], path[-1]+np.array([0, 3000]))))  # 直走3
path = np.vstack((path, arc(path[-1], 1000, math.pi, 0)))  # 向右转180度半径为1m的圆弧
'''

# 圆圈
def generateCirclePath(r):  # 半径r(m)的圆
    pathName = 'circle_r' + str(r)
    path = np.array([[0, 0]])
    path = np.vstack((path, arc(path[-1], r * 1000, 6, 0)))
    np.save('path/' + pathName, path)

    p = np.array(path).T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()


r = 1  # 半径 m
#generateCirclePath(r)

def generateArcPath(r, theta):  # 半径r(m)的圆
    pathName = 'arc_r' + str(r)
    path = np.array([[0, 0]])
    path = np.vstack((path, arc(path[-1], r * 1000, theta /180 * math.pi, 0)))
    np.save('path/' + pathName, path)

    p = np.array(path).T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()


r = 6  # 半径 m
theta = 50
# generateArcPath(r, theta)



# 直线
def generateLinePath(l):  # 长l(m)的直线
    pathName = 'line_l' + str(l)
    path = np.array([[0, 0]])
    path = np.vstack((path, line(path[-1], path[-1] + np.array([0, l*1000]))))  # 直走3
    np.save('path/' + pathName, path)

    p = path.T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()



l = 9  # m
#generateLinePath(l)



# S形曲线
def generateSPath(r):  # 圆弧半径r(m)
    pathName = 'S_r' + str(r)

    path = np.array([[0, 0]])
    print(len(path))
    path = np.vstack((path, arc(path[-1], r * 1000, math.pi / 2, 0)))
    print(len(path))
    path = np.vstack((path, arc(path[-1], r * 1000, -math.pi / 2, math.pi / 2)))
    # path = np.vstack((path, arc(path[-1], r * 1000, math.pi / 2, 0)))
    # path = np.vstack((path, arc(path[-1], r * 1000, -math.pi / 2, math.pi / 2)))

    gamma = math.pi/4  # 顺时针旋转45度
    rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    new_path = []
    for p in path:
        # print(p)
        new_path.append(np.matmul(rotation, p))

    for p in new_path:
        # print(p)
        p[0] = -p[0]

    new_path = np.array(new_path)
    np.save('path/' + pathName, new_path)
    p = new_path.T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()


# generateSPath(3.2)


# 8形曲线
def generate8Path(r):  # 圆弧半径r(m)
    pathName = '8_r' + str(r)

    path = np.array([[0, 0]])
    path = np.vstack((path, arc(path[-1], r * 1000, math.pi, 0)))
    path = np.vstack((path, arc(path[-1], r * 1000, -math.pi, 0)))
    path = np.vstack((path, arc(path[-1], r * 1000, -math.pi, math.pi)))
    path = np.vstack((path, arc(path[-1], r * 1000, math.pi, -math.pi)))

    gamma = math.pi/2  # 逆时针旋转90度
    rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    new_path = []
    for p in path:
        # print(p)
        new_path.append(np.matmul(rotation, p))

    new_path = np.array(new_path)
    np.save('path/' + pathName, new_path)
    p = new_path.T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()


r = 2  # m
#generate8Path(r)

def generateRightTurn():  # 圆弧半径r(m)
    l1 = 5  # m
    l2 = 2  # m
    r = 1  # m
    theta = 45  # degree

    pathName = 'L_' + str(theta)

    path = np.array([[0, 0]])
    #print(len(path))

    path = np.vstack((path, line(path[-1], path[-1] + np.array([0, l1*1000]))))  # 直走l1
    #print(len(path))

    path = np.vstack((path, arc(path[-1], r * 1000, (180 - theta) / 180 * math.pi, 0)))  # 转圆弧
    #print(len(path))

    p1 = np.array([0, l2*1000])
    gamma = (theta - 180) / 180 * math.pi  # 逆时针旋转90度
    rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    p1 = np.matmul(rotation, p1)
    path = np.vstack((path, line(path[-1], path[-1] + p1)))  # 继续直走l2
    #print(len(path))

    new_path = np.array(path)
    np.save('path/' + pathName, new_path)
    p = new_path.T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()

#generateRightTurn()


def generateLeftTurn():  # 圆弧半径r(m)
    l1 = 5  # m
    l2 = 2  # m
    r = 1  # m
    theta = 90  # degree

    pathName = 'L_-' + str(theta)

    path = np.array([[0, 0]])
    print(len(path))

    path = np.vstack((path, line(path[-1], path[-1] + np.array([0, l1*1000]))))  # 直走l1
    print(len(path))

    path = np.vstack((path, arc(path[-1], r * 1000, -(180 - theta) / 180 * math.pi, math.pi)))  # 转圆弧
    print(len(path))

    p1 = np.array([0, l2*1000])
    gamma = (-theta - 180) / 180 * math.pi  # 逆时针旋转90度
    rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    p1 = np.matmul(rotation, p1)
    path = np.vstack((path, line(path[-1], path[-1] + p1)))  # 继续直走l2
    print(len(path))

    new_path = np.array(path)
    np.save('path/' + pathName, new_path)
    p = new_path.T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()

#generateLeftTurn()


def generateLearnPath():  # 圆弧半径r(m)
    l1 = 7  # m
    l2 = 1  # m
    r = 1  # m
    theta = 45  # degree

    pathName = 'learn'

    path = np.array([[0, 0]])
    print(len(path))

    path = np.vstack((path, line(path[-1], path[-1] + np.array([0, l1*1000]))))  # 直走l1
    print(len(path))

    path = np.vstack((path, arc(path[-1], r * 1000, (180 - theta) / 180 * math.pi, 0)))  # 转圆弧
    print(len(path))

    p1 = np.array([0, l2*1000])
    gamma = (theta - 180) / 180 * math.pi  # 逆时针旋转90度
    rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    p1 = np.matmul(rotation, p1)

    path = np.vstack((path, line(path[-1], path[-1] + p1)))  # 继续直走l2
    print(len(path))

    r = 2.5  # m
    path = np.vstack((path, arc(path[-1], r * 1000, math.pi / 2, -3 / 4 * math.pi)))
    print(len(path))

    path = np.vstack((path, arc(path[-1], r * 1000, -math.pi / 2, -1 / 4 * math.pi)))
    print(len(path))

    new_path = np.array(path)
    np.save('path/' + pathName, new_path)
    p = new_path.T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()


#generateLearnPath()

def generateOutline():
    safe_distance = 300
    ## line_l9
    # l = 9
    # pathName = 'line_l' + str(l)
    # path = np.array([[-safe_distance, 0]])
    # path = np.vstack((path, line(path[-1], path[-1] + np.array([0, l*1000]))))  # 直走3
    # path = np.vstack((path, [safe_distance, l*1000]))
    # path = np.vstack((path, line(path[-1], path[-1] + np.array([0, -l*1000]))))  # 直走3

    ## L_
    # l1 = 5  # m
    # l2 = 2  # m
    # r = 1  # m
    # theta = 135  # degree
    # pathName = 'L_' + str(theta)
    # path = np.array([[-safe_distance, 0]])
    # path = np.vstack((path, line(path[-1], path[-1] + np.array([0, l1*1000]))))  # 直走l1
    # path = np.vstack((path, arc(path[-1], (r * 1000 + safe_distance), (180 - theta) / 180 * math.pi, 0)))  # 转圆弧
    # p1 = np.array([0, l2*1000])
    # gamma = (theta - 180) / 180 * math.pi  # 逆时针旋转90度
    # rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    # p1 = np.matmul(rotation, p1)
    # path = np.vstack((path, line(path[-1], path[-1] + p1)))  # 继续直走l2
    #
    # path2 = np.array([[safe_distance, 0]])
    # path2 = np.vstack((path2, line(path2[-1], path2[-1] + np.array([0, l1*1000]))))  # 直走l1
    # path2 = np.vstack((path2, arc(path2[-1], (r * 1000 - safe_distance), (180 - theta) / 180 * math.pi, 0)))  # 转圆弧
    # p1 = np.array([0, l2*1000])
    # gamma = (theta - 180) / 180 * math.pi  # 逆时针旋转90度
    # rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    # p1 = np.matmul(rotation, p1)
    # path2 = np.vstack((path2, line(path2[-1], path2[-1] + p1)))  # 继续直走l2
    # path2list = list(path2.tolist())
    # path2list.reverse()
    # path2 = np.array(path2list)
    # path = np.vstack((path, path2))

    ## L_-
    # l1 = 5  # m
    # l2 = 2  # m
    # r = 1  # m
    # theta = 45  # degree
    # pathName = 'L_-' + str(theta)
    # path = np.array([[-safe_distance, 0]])
    # path = np.vstack((path, line(path[-1], path[-1] + np.array([0, l1*1000]))))  # 直走l1
    # path = np.vstack((path, arc(path[-1], (r * 1000 - safe_distance), -(180 - theta) / 180 * math.pi, math.pi)))  # 转圆弧
    # p1 = np.array([0, l2*1000])
    # gamma = -(theta - 180) / 180 * math.pi  # 逆时针旋转90度
    # rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    # p1 = np.matmul(rotation, p1)
    # path = np.vstack((path, line(path[-1], path[-1] + p1)))  # 继续直走l2
    #
    # path2 = np.array([[safe_distance, 0]])
    # path2 = np.vstack((path2, line(path2[-1], path2[-1] + np.array([0, l1*1000]))))  # 直走l1
    # path2 = np.vstack((path2, arc(path2[-1], (r * 1000 + safe_distance), -(180 - theta) / 180 * math.pi, math.pi)))  # 转圆弧
    # p1 = np.array([0, l2*1000])
    # gamma = -(theta - 180) / 180 * math.pi  # 逆时针旋转90度
    # rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    # p1 = np.matmul(rotation, p1)
    # path2 = np.vstack((path2, line(path2[-1], path2[-1] + p1)))  # 继续直走l2
    # path2list = list(path2.tolist())
    # path2list.reverse()
    # path2 = np.array(path2list)
    # path = np.vstack((path, path2))

    ## S
    # r = 3.2
    # pathName = 'S_r' + str(r)
    # path = np.array([[safe_distance, 0]])
    # path = np.vstack((path, arc(path[-1], r * 1000 - safe_distance, math.pi / 2, 0)))
    # path = np.vstack((path, arc(path[-1], r * 1000 + safe_distance, -math.pi / 2, math.pi / 2)))
    # path2 = np.array([[-safe_distance, 0]])
    # path2 = np.vstack((path2, arc(path2[-1], r * 1000 + safe_distance, math.pi / 2, 0)))
    # path2 = np.vstack((path2, arc(path2[-1], r * 1000 - safe_distance, -math.pi / 2, math.pi / 2)))
    # path2list = list(path2.tolist())
    # path2list.reverse()
    # path2 = np.array(path2list)
    # path = np.vstack((path, path2))
    # gamma = math.pi/4  # 顺时针旋转45度
    # rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    # new_path = []
    # for p in path:
    #     # print(p)
    #     new_path.append(np.matmul(rotation, p))
    # for p in new_path:
    #     # print(p)
    #     p[0] = -p[0]
    # path = np.array(new_path)

    ## learn
    l1 = 7  # m
    l2 = 1  # m
    r = 1  # m
    theta = 45  # degree
    pathName = 'learn'
    path = np.array([[-safe_distance, 0]])
    path = np.vstack((path, line(path[-1], path[-1] + np.array([0, l1*1000]))))  # 直走l1
    path = np.vstack((path, arc(path[-1], r * 1000 + safe_distance, (180 - theta) / 180 * math.pi, 0)))  # 转圆弧
    p1 = np.array([0, l2*1000])
    gamma = (theta - 180) / 180 * math.pi  # 逆时针旋转90度
    rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    p1 = np.matmul(rotation, p1)
    path = np.vstack((path, line(path[-1], path[-1] + p1)))  # 继续直走l2
    r = 2.5  # m
    path = np.vstack((path, arc(path[-1], r * 1000 + safe_distance, math.pi / 2, -3 / 4 * math.pi)))
    path = np.vstack((path, arc(path[-1], r * 1000 - safe_distance, -math.pi / 2, -1 / 4 * math.pi)))

    r = 1
    path2 = np.array([[safe_distance, 0]])
    path2 = np.vstack((path2, line(path2[-1], path2[-1] + np.array([0, l1*1000]))))  # 直走l1
    path2 = np.vstack((path2, arc(path2[-1], r * 1000 - safe_distance, (180 - theta) / 180 * math.pi, 0)))  # 转圆弧
    p1 = np.array([0, l2*1000])
    gamma = (theta - 180) / 180 * math.pi  # 逆时针旋转90度
    rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    p1 = np.matmul(rotation, p1)
    path2 = np.vstack((path2, line(path2[-1], path2[-1] + p1)))  # 继续直走l2
    r = 2.5  # m
    path2 = np.vstack((path2, arc(path2[-1], r * 1000 - safe_distance, math.pi / 2, -3 / 4 * math.pi)))
    path2 = np.vstack((path2, arc(path2[-1], r * 1000 + safe_distance, -math.pi / 2, -1 / 4 * math.pi)))

    path2list = list(path2.tolist())
    path2list.reverse()
    path2 = np.array(path2list)
    path = np.vstack((path, path2))

    np.save('path/path_outline/' + pathName, path)
    p = path.T
    pl.plot(p[0], p[1])
    pl.axis('equal')
    pl.show()

generateOutline()