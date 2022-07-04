# [과제]
# 3가지 원핫인코딩 방식을 비교할 것
#
#1. pandas의 get_dummies

#2. tensoflow의 to_categorical

#3. sklearn의 OneHotEncoder

# 미세한 차이를 정리하시오



######################################################
# 1.pandas 에서는 
# import pandas as pd
# import numpy as np
# # dum_col=pd.get_dummies(df1['colabc'],prefix="col")
# # dum_col 
# fruit = pd.DataFrame({'name':['apple', 'banana', 'cherry', 'durian', np.nan],
#                       'color':['red', 'yellow', 'red', 'green', np.nan]}) 
# print(fruit.shape) # (5, 2)
# #      name   color
# # 0   apple     red
# # 1  banana  yellow
# # 2  cherry     red
# # 3  durian   green
# # 4     NaN     NaN


# print(pd.get_dummies(fruit)) #[5 rows x 7 columns]
# #    name_apple  name_banana  ...  color_red  color_yellow
# # 0           1            0  ...          1             0
# # 1           0            1  ...          0             1
# # 2           0            0  ...          1             0
# # 3           0            0  ...          0             0
# # 4           0            0  ...          0             0

# 정수가 아닌 문자열 데이터를 코드 상에서 따로 열을 지정하여 더미를 만들어 원핫인코딩을 할 수 있다.
# 단점:pandas.get_dummies는 train 데이터의 특성을 학습하지 않기 때문에 train 데이터에만 있고 
# test 데이터에는 없는 카테고리를 test 데이터에서 원핫인코딩 된 칼럼으로 바꿔주지 않는다.
######################################################

# 2.keras에서는

# >> from tensorflow.keras.utils import to_categorical

# >> to_categorical(data)
# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y) 
# 가지고 있는 y 의 정수가 아닌 값을 y의 값으로 치환해준다.
# 해당 방법은 무조건 0부터 시작해서 순차적으로 카테고리를 작성한다.ex) unique가 [1,3,4,5] 의 4개 특성이 인코딩 되어 컬럼도 5개가
# 되어야 하지만 0부터 시작해서 순차적으로 가기 때문에 [0,1,2,3,4,5]로 총 6개의 컬럼이 생겨나므로 일반적인 상황에 사용하지 않는다.
######################################################
 

# 3.sklearn에서는


# from sklearn.preprocessing import OneHotEncoder
# y = np.array(y).reshape(-1,1)
# print(y) # (581012, 8)

# ohe = OneHotEncoder()
# ohe.fit(y)

# y_class = ohe.transform(y).toarray()

# print(y_class.shape) # (581012, 7)
# sklearn에서는 원핫인코딩은 데이터가 문자로 되어 있다면 바로 실시할 수 없다.라벨인코딩 등 문자를 정수로 가공한 후
# 원핫인코딩을 진행할 수 있다.kesras에 to_categorical과 다르게 reshape를 통해 shape을 바꾸고 진행해야 한다.

####################결론##################
# 3개의 모델을 모두 비교해보았지만 pandas에 get_dummies의 활용이 다른 기능보다 유용할 것으로 보인다.