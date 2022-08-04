from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.75,shuffle=True,random_state=100)
#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron ,LogisticRegression 
#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #공부하자 
from sklearn.ensemble import RandomForestClassifier #공부하자 
from tqdm import tqdm

def models(model):
    if model == 'knn':
        mod = KNeighborsClassifier()
    elif model == 'svc':
        mod = SVC()
    elif model == 'tree':
        mod =  DecisionTreeClassifier()
    elif model == 'forest':
        mod =  RandomForestClassifier()
    return mod
model_list = ['knn', 'svc',  'tree', 'forest']
empty_list = [] #empty list for progress bar in tqdm library
for model in tqdm(model_list, desc = 'Models are training and predicting ... '):
    empty_list.append(model) # fill empty_list to fill progress bar
    #classifier
    clf = models(model)
    #Training
    clf.fit(x_train, y_train) 
    #Predict
    result = clf.score(x_test,y_test)
    pred = clf.predict(x_test) 
    print('{}-{}'.format(model,result))
    
    
# knn-0.009009009009009009
# svc-0.0
# tree-0.009009009009009009
# forest-0.0    
