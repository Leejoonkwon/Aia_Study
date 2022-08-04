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
from tqdm import tqdm
from sklearn.svm import LinearSVC,SVC,SVR
from sklearn.linear_model import Perceptron ,LogisticRegression 
#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
<<<<<<< HEAD
=======
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #공부하자 
from sklearn.ensemble import RandomForestClassifier #공부하자 
from sklearn.linear_model import LinearRegression 
>>>>>>> 867878a6ab04aaa05ae01cf47d4eb319482b719e

#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor #공부하자 
from sklearn.ensemble import RandomForestRegressor #공부하자 
from sklearn.linear_model import LogisticRegression 
def models(model):
    if model == 'knn':
        mod = KNeighborsRegressor()
    elif model == 'svr':
        mod = SVR()
    elif model == 'tree':
        mod =  DecisionTreeRegressor()
    elif model == 'forest':
<<<<<<< HEAD
        mod =  RandomForestRegressor()
    return mod
model_list = ['knn', 'svr',  'tree', 'forest']
=======
        mod =  RandomForestClassifier()
    elif model == 'linear':
        mod =  LinearRegression ()    
    elif model == 'linearSVC':
        mod =  LinearSVC ()       
    return mod
model_list = ['knn', 'svc',  'tree', 'forest','linear','linearSVC']
cnt = 0
>>>>>>> 867878a6ab04aaa05ae01cf47d4eb319482b719e
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

<<<<<<< HEAD
# knn-0.36975907952835774
# svr-0.19021720314000012
# tree--0.07117271384360624
# forest-0.44679527655794804
=======
    
    
# knn-0.009009009009009009
# svc-0.0
# tree-0.009009009009009009
# forest-0.0
# linear-0.5282621411648274
# linearSVC-0.009009009009009009
>>>>>>> 867878a6ab04aaa05ae01cf47d4eb319482b719e
