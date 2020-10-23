import numpy as np

# path = np.load('path/line_l6.npy')
path = np.loadtxt('path/test.txt')

np.set_printoptions(threshold=np.inf)
print(path)