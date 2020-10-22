import socket
import struct
from quaternion import quaternion
import numpy as np
import math
import time

HOST, PORT ='', 8080  # 域名和端口号

def showHostInfo():
    hostname = socket.gethostname()
    result = socket.getaddrinfo(hostname, None, 0, socket.SOCK_STREAM)
    hostip = [x[4][0] for x in result]
    print('host ip:', hostip)
    # print('host port:', PORT)
showHostInfo()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

s.bind(('169.254.97.52', 10010))

x_axis = quaternion(0,1,0,0)
y_axis = quaternion(0,0,1,0)
z_axis = quaternion(0,0,0,1)


while True:
    data, addr = s.recvfrom(120)
    if len(data) > 0:
        data_list = struct.unpack('>Iddddddd', data)
        # The format of optitrack quaternion is (x,y,z,w)
        position = data_list[1:4]
        rotation = quaternion(data_list[-1],data_list[-4],data_list[-3],data_list[-2])
        new_vec = ((rotation*y_axis)*(rotation.conjugate())).to_list()[1:]
        # (x,z)
        vec1 = np.array([new_vec[0],new_vec[2]])
        vec2 = np.array([0,1])
        angle_to_vec2 = math.atan2(vec2[0],vec2[1]) - math.atan2(vec1[0],vec1[1])
        angle = math.floor(angle_to_vec2 / math.pi * 180) - 180
        if angle <= -180: angle += 360

        height = int(data_list[2] * 1000)
        print(angle)
    else:
        print('no data')
