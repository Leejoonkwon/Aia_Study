# 이진분류가 아닌 다중분류는 loss가 crossentropyloss와 y를 LongTensor로 타입 교체한다.
# 모델의 아웃풋 노드를 클래스에 맞게 입력!
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import pandas as pd
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
# DEVICE = torch.device(['cuda:0','cuda:1'] if USE_CUDA else 'cpu')
# gpu 다중 사용시 list로 참조
print('torch : ',torch.__version__,'사용DEVICE : ',DEVICE)

#1. 데이터
path = 'D:\study_data\_data\_csv\_ddarung/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)


###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['count'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1459, 9)

y = train_set['count']
x = x.values
y = y.values
from sklearn.model_selection import train_test_split
   
x_train,x_test,y_train,y_test \
= train_test_split(x,y,train_size=0.7,
  shuffle=True,random_state=1234)
   
x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.FloatTensor(y_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)

x_test = torch.FloatTensor(x_test)
# y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.FloatTensor(y_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

# sklearn에 있는 scaler 쓸 시 gpu 모드로 못함!
# print(x_train.size())
print(x_train.shape,y_train.shape) 
# torch.Size([1021, 9]) torch.Size([1021, 1])

#2. 모델 
model = nn.Sequential(
    nn.Linear(9,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.Linear(16,1),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
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

# y_predict = torch.argmax(model(x_test),axis=1)
y_predict = model(x_test)

                            
# print('result : ',y_predict.detach().cpu().numpy())
from sklearn.metrics import r2_score
r2 = r2_score(y_test.detach().cpu().numpy(),
              y_predict.detach().cpu().numpy()
              )
print('R2 : ',r2)

# ========== 평가, 예측========
# loss :  0.28000548481941223
# R2 :  0.7564272695202631


