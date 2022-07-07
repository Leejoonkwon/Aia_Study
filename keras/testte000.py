# [실습]
import numpy as np
y =np.array( [0.51,0.3,0.2,0.7,15,16,17,18,19,20])

y =np.round(y,0)
print(y)

y = np.where(y>0.5,0,1)
print(y)
