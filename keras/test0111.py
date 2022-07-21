import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([6,7,8,9,10])

print(x.shape)# (5,)
print(np.tile(x,2))
print(np.tile(x,2).shape) #(10,)

