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
print(x_train.shape,y_train.shape) # torch.Size([398, 30]) torch.Size([398, 1])
##################################
from torch.utils.data import TensorDataset,DataLoader
train_set = TensorDataset(x_train,y_train) #  x,y를 합친다.
test_set = TensorDataset(x_test,y_test) #  x,y를 합친다.
print(train_set)         # <torch.utils.data.dataset.TensorDataset object at 0x000001F4E9FA1F70>
print('==============train_set[0]===========================')
print(train_set[0])
print('==============train_set[0][0]===========================')
print(train_set[0][0])
print('==============train_set[0][1]===========================')
print(train_set[0][1])
print(len(train_set)) # 398
# x와 y  데이터 결합 
train_loader = DataLoader(train_set,batch_size=40,shuffle=True)
test_loader = DataLoader(test_set,batch_size=40)
print(train_loader)     # <torch.utils.data.dataloader.DataLoader object at 0x00000268FB3A48B0>



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

def train(model,criterion,optimizer,loader):
    #model.train()
    total_loss = 0
    for x_batch,y_batch in loader:
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis,y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader) # 평균이 필수적으로 요구되진 않는다.
EPOCHS = 300
for epoch  in range(1, EPOCHS+1):
    loss = train(model,criterion,optimizer,train_loader)
    print('epoch : {}, loss : {:.8f}'.format(epoch,loss))

#4. 평가, 예측
print("========== 평가, 예측========")
def evaluate(model,criterion,loader):
    model.eval()
    total_loss = 0
    for x_batch,y_batch in loader:
        
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis,y_batch)
            total_loss += loss.item()
    return loss.item()

loss = evaluate(model, criterion,test_loader)
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
path = './_save/'

torch.save(model.state_dict(),path + 'torch13_state_dict.pt')
# 가중치와 model 저장 











