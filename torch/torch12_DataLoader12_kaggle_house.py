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
path = 'D:\study_data\_data\_csv\kaggle_house/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)

print(train_set)

print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']
from sklearn.preprocessing import LabelEncoder
for col in cols:
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


print(test_set)
print(train_set.shape) #(1460,76)
print(test_set.shape) #(1459, 75) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']


x = x.values
y = y.values
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
    train_test_split(x,y,train_size=0.7,shuffle=True,
                     random_state=123)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
from torch.utils.data import TensorDataset,DataLoader
train_set = TensorDataset(x_train,y_train)
test_set = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_set,batch_size=40,shuffle=True)
test_loader = DataLoader(test_set,batch_size=40)
           
#2. 모델
class Model(nn.Module):
    def __init__(self,input_dim,out_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim,64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    
    def forward(self,input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        # x = self.softmax(x)
        return x
model = Model(x_train.shape[1],1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,loader):
    total_loss = 0
    for x_batch,y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis,y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model,criterion,optimizer,train_loader)
    if epoch % 10 == 0 :
        print('epoch : {}, loss : {:.8f}'.format(epoch,loss))

#4. 평가,예측

def evaluate(model,criterion,loader):
    model.eval()
    total_loss = 0
    for x_batch,y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis,y_batch)
            total_loss += loss.item()
    return loss.item()

loss = evaluate(model,criterion,test_loader)
print('==============점수============')
print('loss :', loss)
from sklearn.metrics import accuracy_score,r2_score
y_predict = model(x_test)
         
r2 = r2_score(y_predict.detach().cpu().numpy(),
              y_test.detach().cpu().numpy())
print('R2 : ',r2)

# ==============점수============
# loss : 0.5514456629753113
# acc : 93.33%


