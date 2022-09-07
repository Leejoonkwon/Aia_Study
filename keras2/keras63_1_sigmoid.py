import numpy as np
import matplotlib.pyplot as  plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))
sigmoid2 = lambda x:1/(1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)

print(x,len(x)) # 100

y = sigmoid(x) # 0~1 사이의 값에 수렴한다. 0 or 1 이 아니다!

plt.plot(x,y)
plt.grid()
plt.show()