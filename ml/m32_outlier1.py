import numpy as np
aaa = np.array([2,3,-16,4,5,6,7,8,50,10])

def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
outliers_loc = outliers(aaa)
print("이상치의 위치 :",outliers_loc)
# 1사분위 :  2.5
# q2 :  5.5
# quartile_3 :  7.75
# iqr : 5.25
# 이상치의 위치 : (array([2, 8, 9], dtype=int64),)
import matplotlib.pyplot as  plt
plt.boxplot(aaa)
plt.show()


