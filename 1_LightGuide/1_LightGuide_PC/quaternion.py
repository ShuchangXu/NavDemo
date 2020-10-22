import math

class quaternion():
    def __init__(self, r, i, j, k):
        self.r = r
        self.i = i
        self.j = j
        self.k = k

    def __mul__(self, B):
        r = self.r*B.r-self.i*B.i-self.j*B.j-self.k*B.k
        i = self.r*B.i+self.i*B.r+self.j*B.k-self.k*B.j
        j = self.r*B.j-self.i*B.k+self.j*B.r+self.k*B.i
        k = self.r*B.k+self.i*B.j-self.j*B.i+self.k*B.r
        return quaternion(r, i, j, k)
    
    def to_list(self):
        return [self.r,self.i,self.j,self.k]

    def conjugate(self):
        return quaternion(self.r,-self.i,-self.j,-self.k)

    def mod(self):
        return math.sqrt(self.r**2+self.i**2+self.j**2+self.k**2)

if __name__ == '__main__':
    r = [-0.022586537525057793, -0.765666663646698, -0.6217049360275269,0.16348497569561005]
    a = quaternion(0.0,0.707,0.0,-0.707)
    b = quaternion(0.707, 0.0, -0.707, 0.0)
    print((a*b).to_list())
