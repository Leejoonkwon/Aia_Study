#2. 모델 구성
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['font.family']='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus']=False
import time

#1. 데이터
datasets = load_boston()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.75,shuffle=True,random_state=100)
#2. 모델구성
from sklearn.svm import LinearSVC,SVC
#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #공부하자 
from sklearn.ensemble import RandomForestClassifier #공부하자 
from sklearn.linear_model import LinearRegression 

def models(model):
    if model == 'knn':
        mod = KNeighborsClassifier()
    elif model == 'svc':
        mod = SVC()
    elif model == 'tree':
        mod =  DecisionTreeClassifier()
    elif model == 'forest':
        mod =  RandomForestClassifier()
    elif model == 'linear':
        mod =  LinearRegression ()    
    elif model == 'linearSVC':
        mod =  LinearSVC ()       
    return mod
model_list = ['knn', 'svc',  'tree', 'forest','linear','linearSVC']
cnt = 0
empty_list = [] #empty list for progress bar in tqdm library
for model in (model_list):
    empty_list.append(model) # fill empty_list to fill progress bar
    #classifier
    clf = models(model)
    #Training
    clf.fit(x_train, y_train) 
    #Predict
    result = clf.score(x_test,y_test)
    pred = clf.predict(x_test) 
    print('{}-{}'.format(model,result))

# knn-0.44371014889060933
# svr-0.2060097280934967 
# tree-0.7598265665684787
# forest-0.8817271234480091