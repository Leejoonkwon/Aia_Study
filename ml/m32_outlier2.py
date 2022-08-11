import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
# (2,13)
aaa = np.transpose(aaa) #(13,2)
# print(aaa)
# [[   -10    100]
#  [     2    200]
#  [     3    -30]
#  [     4    400]
#  [     5    500]
#  [     6    600]
#  [     7 -70000]
#  [     8    800]
#  [     9    900]
#  [    10   1000]
#  [    11    210]
#  [    12    420]
#  [    50    350]]

abc = aaa[:,0]
abc2 = aaa[:,1]

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
    
outliers_loc1 = outliers(abc)
print("이상치의 위치 :",outliers_loc1)
# 이상치의 위치 : (array([ 0, 12], dtype=int64),)
outliers_loc2 = outliers(abc2)
print("이상치의 위치 :",outliers_loc2)
# 이상치의 위치 : (array([6], dtype=int64),

# 1사분위 :  4.0
# q2 :  7.0
# 3사분위 :  10.0
# iqr : 6.0
# 이상치의 위치 : (array([ 0, 12], dtype=int64),)
# 1사분위 :  200.0
# q2 :  400.0
# 3사분위 :  600.0
# iqr : 400.0
# 이상치의 위치 : (array([6], dtype=int64),)
