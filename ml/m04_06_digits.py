from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import r2_score,accuracy_score
#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.95,shuffle=True, random_state=12 ) 
#2. 모델 구성
from sklearn.utils import all_estimators

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='Regressor')

# print('allAlgorithms :',allAlgorithms)
print('모델의 갯수 :',len(allAlgorithms)) #모델의 갯수 : 41

for (name,algorithms) in allAlgorithms:
    try: # for문을 실행하는 와중에 예외 (error)가 발생하면 무시하고 진행 <예외처리>
        model = algorithms()
        model.fit(x_train,y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(name,'의 정답률 :',acc)
    except:
        #continue
        print(name,'은 안나온 놈!!!')
# LogisticRegressionCV 의 정답률 : 0.9666666666666667
# MLPClassifier 의 정답률 : 0.9666666666666667
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 : 0.9
# NearestCentroid 의 정답률 : 0.9
# NuSVC 의 정답률 : 0.9777777777777777
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 : 0.9555555555555556
# Perceptron 의 정답률 : 0.9444444444444444
#    QuadraticDiscriminantAnalysis 의 정답률 : 0.8888888888888888
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 : 0.9777777777777777
# RidgeClassifier 의 정답률 : 0.9444444444444444    
# RidgeClassifierCV 의 정답률 : 0.9444444444444444  
# SGDClassifier 의 정답률 : 0.9777777777777777
# SVC 의 정답률 : 1.0
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!        