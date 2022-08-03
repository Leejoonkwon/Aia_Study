from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.9,shuffle=True,random_state=100)
# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) 442행 10열

# print(datasets.feature_names)
# print(datasets.DESCR)
#2. 모델구성
model = LinearSVC()
#3. 컴파일,훈련

model.fit(x_train,y_train)

#4. 평가,예측
results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
#ex)분류는 ACC 회귀는 R2스코어
print("results :",results)
# loss : 2155.687744140625
# r2스코어 : 0.6430334416083464

#######ML 사용시
# results : 0.022222222222222223

