import socket

class HandleDevice:
    def __init__(self):
        super().__init__()
        self.HOST, self.PORT = '', 8080  # 域名和端口号
        self.clientSocket = socket.socket()

    def connect(self):
        print("Please connect client...")
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.bind((self.HOST, self.PORT))
        self.clientSocket.listen(5)
        self.clientSocket, addr = self.clientSocket.accept()
        print("Client is connected!")