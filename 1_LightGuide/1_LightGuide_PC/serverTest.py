import socket
import struct

HOST, PORT ='', 8080  # 域名和端口号

def showHostInfo():
    hostname = socket.gethostname()
    result = socket.getaddrinfo(hostname, None, 0, socket.SOCK_STREAM)
    hostip = [x[4][0] for x in result]
    print('host ip:', hostip)
    print('host port:', PORT)


showHostInfo()

BUFFER_SIZE = 6  # 缓冲区(缓存)
tcpServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ADDR = (HOST, PORT)
tcpServerSocket.bind(ADDR)
tcpServerSocket.listen(5)


count = 0

while True:
    # 连接
    tcpClientSocket, addr = tcpServerSocket.accept()
    #tcpClientSocket.send(str(count).encode())
    print("Connected!")

    while True:
        data = tcpClientSocket.recv(BUFFER_SIZE)  # 接收数据
        IMUDirection = int(data.decode())
        print("IMU: ", IMUDirection)
        # if 0 <= IMUDirection < 360:
        # CommandDirection = 66 + 180  # 数据处理
        # (0, 360]
        data1 = str(IMUDirection).zfill(3)
        data2 = str(IMUDirection).zfill(3)
        CommandDirection = data1+data2  # 直接将两个数据拼接成长度为6的字符串，各3个
        tcpClientSocket.send(str(CommandDirection).encode())  # 发送数据

    tcpClientSocket.close()

