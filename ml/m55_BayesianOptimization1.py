param_bounds ={'x1': (-1, 5),
               'x2': (0, 4)}
def y_function(x1,x2):
    return -x1 **2 - (x2 - 2) **2 + 10

from pickletools import optimize
from bayes_opt import BayesianOptimization
# 앞으로 y 펑션에 모델 돌리고 난 점수의 계산식을 넣는다.
# pbounds 는 파라미터를 딕셔너리 형태로 넣는다.
optimizer = BayesianOptimization(f=y_function,
                                 pbounds=param_bounds,
                                 random_state=1234)
optimizer.maximize(init_points=2, # 초기 탐색치(초기값)
                   n_iter=20, # 반복횟수 
                   )

print(optimizer.max)
# {'target': 9.999835124236093, 
#  'params': {'x1': 0.007890722975053967, 
#             'x2': 1.9898702292800778}}




