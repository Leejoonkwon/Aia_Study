import numpy as  np
import matplotlib.pyplot as plt

def SELU(x, lambdaa = 1.0507, alpha = 1.6732):
    if x >= 0:
        return lambdaa * x
    else:
        return lambdaa * alpha * (np.exp(x) - 1)
# selu2 = lambda x : np.where(x>=0, x, 0.1*(np.exp(x)-1))

# x = np.arange(-5, 5, 0.1)
# y = selu(x)
series_in = [x for x in range(-5, 10)]
series_out = [SELU(x) for x in input]
plt.plot(series_in,series_out)
plt.grid()
plt.show()

# elu , selu , reaky relu
# 3_2 , 3_3 ,  3_4

