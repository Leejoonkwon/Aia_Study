import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
path = 'D:\study_data\_data\_csv\kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

print(train_set) 
print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']
for col in cols:
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
print(x) #(891, 9)
y = train_set['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,shuffle=True ,random_state=1234)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(x_train.shape,y_train.shape)   # (712, 9) (712,)
print(x_test.shape,y_test.shape)     # (179, 9) (179,)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

#2. 모델 
# model = nn.Sequential(
#     nn.Linear(30,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,1),
#     nn.Sigmoid()
# ).to(DEVICE)
class Model(nn.Module):
    def __init__(self,input_dim,output_dim):
        # super().__init__() # cannot assign module before Module.__init__() call  super없이 실행할 경우 error
        super(Model,self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
    
model = Model(x_train.shape[1],y_train.shape[1]).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis,y_train)
    
    loss.backward()
    optimizer.step()
    return loss.item()
EPOCHS = 1000
for epoch  in range(1, EPOCHS+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    print('epoch : {}, loss : {:.8f}'.format(epoch,loss))

#4. 평가, 예측
print("========== 평가, 예측========")
def evaluate(model,criterion,x_test,y_test):
    model.eval()
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis,y_test)
    return loss.item()

loss = evaluate(model, criterion,x_test,y_test)
print('loss : ',loss)
y_predict = (model(x_test) >= 0.5).float()
print(y_predict[:10])
score = (y_predict == y_test).float().mean()
print('Accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score

# acc = accuracy_score(y_predict,y_test) #  GPU 상태라서  error 임
# print('ACC : ',acc)

acc = accuracy_score(y_predict.cpu(),y_test.cpu())
print('ACC : ',acc)

# Accuracy : 0.9883
# ACC :  0.9883040935672515

# Accuracy : 0.9942
# ACC :  0.9941520467836257








