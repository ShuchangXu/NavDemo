import serial
import struct
import time

# Hradware interface

class octaVib:
    def __init__(self, port):
        self.port = serial.Serial(port,115200)
        
    def direct_vibration(self,channel,duration,amplitude):
        tx_array = bytearray(8)
        struct.pack_into('<BBiH',tx_array,0,int(0x01),int(channel),int(duration),int(amplitude))
        self.port.write(tx_array)
        self.port.flush()

    def NOP(self):
        tx_array = bytearray(8)
        struct.pack_into('<B',tx_array,0,int(0x02))
        self.port.write(tx_array)
        self.port.flush()
    
    def start(self):
        tx_array = bytearray(8)
        struct.pack_into('<B',tx_array,0,int(0x03))
        self.port.write(tx_array)
        self.port.flush()
    
    def stop(self):
        tx_array = bytearray(8)
        struct.pack_into('<B',tx_array,0,int(0x04))
        self.port.write(tx_array)
        self.port.flush()

    def angle(self,angle):
        tx_array = bytearray(8)
        struct.pack_into('<Bi',tx_array,0,int(0x05),int(angle))
        self.port.write(tx_array)
        self.port.flush()

    def distance(self,distance):
        tx_array = bytearray(8)
        struct.pack_into('<Bi',tx_array,0,int(0x06),int(distance))
        self.port.write(tx_array)
        self.port.flush()

    def set_start(self,Vp,duration):
        tx_array = bytearray(8)
        struct.pack_into('<BHi',tx_array,0,int(0x81),int(Vp),int(duration))
        self.port.write(tx_array)
        self.port.flush()
        time.sleep(0.05)

    def set_stop(self,Vp,duration):
        tx_array = bytearray(8)
        struct.pack_into('<BHi',tx_array,0,int(0x82),int(Vp),int(duration))
        self.port.write(tx_array)
        self.port.flush()
        time.sleep(0.05)

    def set_angle_mapping(self,angle_start,gamma_start,angle_end,gamma_end):
        tx_array = bytearray(8)
        struct.pack_into('<BHi',tx_array,0,int(0x84),int(gamma_start),int(angle_start))
        self.port.write(tx_array)
        self.port.flush()
        time.sleep(0.05)
        tx_array = bytearray(8)
        struct.pack_into('<BHi',tx_array,0,int(0x85),int(gamma_end),int(angle_end))
        self.port.write(tx_array)
        self.port.flush()
        time.sleep(0.05)

    def set_distance_mapping(self,distance_start,gamma_start,gamma_end):
        tx_array = bytearray(8)
        struct.pack_into('<BHi',tx_array,0,int(0x86),int(gamma_start),int(distance_start))
        self.port.write(tx_array)
        self.port.flush()
        time.sleep(0.05)
        tx_array = bytearray(8)
        struct.pack_into('<BH',tx_array,0,int(0x87),int(gamma_end))
        self.port.write(tx_array)
        self.port.flush()
        time.sleep(0.05)

    def set_gamma(self,gamma):
        tx_array = bytearray(8)
        struct.pack_into('<BBBBf',tx_array,0,int(0x88),int(0),int(0),int(0),float(gamma))
        self.port.write(tx_array)
        self.port.flush()
        time.sleep(0.5)
    
if __name__ == '__main__':
    device = octaVib("COM6")
    device.set_start(2000,15)
    device.set_stop(2000,160)
    device.set_angle_mapping(20,20,600,999)
    device.set_distance_mapping(1500,20,700)
    device.set_gamma(2.0)
    for i in range(200):
        device.angle(i*10-1000)
        time.sleep(0.03)
    device.NOP()