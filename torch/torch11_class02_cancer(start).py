from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
# DEVICE = torch.device(['cuda:0','cuda:1'] if USE_CUDA else 'cpu')
# gpu 다중 사용시 list로 참조
print('torch : ',torch.__version__,'사용DEVICE : ',DEVICE)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# (569, 30) (569,)
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test \
= train_test_split(x,y,train_size=0.7,
  shuffle=True,random_state=1234,stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

# sklearn에 있는 scaler 쓸 시 gpu 모드로 못함!
# print(x_train.size())
print(x_train.shape,y_train.shape) # (398, 30)

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
    
model = Model(30,1).to(DEVICE)

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








