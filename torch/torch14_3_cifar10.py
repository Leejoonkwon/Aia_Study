from turtle import forward
from torchvision.datasets import MNIST,CIFAR10
from torch.utils.data import TensorDataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import numpy as np
import torchvision.transforms as tr
  
#1. 데이터
path = './_data/torch_data/'
transf = tr.Compose(tr.ToTensor())
# train_dataset = MNIST(path,train=True,download=True,transform=trasnf)
# test_dataset = MNIST(path,train=False,download=True,transform=trasnf)
# print(train_dataset[0][0].shape) # torch.Size([1, 15, 15])
                                                                                                                       
USE_CUDA = torch.cuda.is_available() 
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
   
train_dataset = CIFAR10(path,train=True,download=True)
test_dataset = CIFAR10(path,train=False,download=True)

x_train,y_train = train_dataset.data/255., train_dataset.targets
x_test,y_test = test_dataset.data/255. , test_dataset.targets
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
# print(x_train.shape,x_test.shape) # torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])
# print(y_train.shape,y_test.shape) # torch.Size([50000]) torch.Size([10000])
# print(x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
# print(np.min(x_train.numpy()),np.max(x_train.numpy())) # 0.0 1.0
   
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
print(x_train.shape,x_test.shape) # torch.Size([60000, 784]) torch.Size([10000, 784])
                                    

train_dset = TensorDataset(x_train,y_train)
test_dset = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_dset,batch_size=100,shuffle=True)
test_loader = DataLoader(test_dset,batch_size=100,shuffle=False)

#2. 모델
class DNN(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features,100),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),)
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(0.5))       
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(0.5))        
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.output_layer =nn.Linear(100,10)
        # 평가할 때는 dropout이 미적용
    def forward(self,x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x

model = DNN(x_train.shape[1]).to(DEVICE)

#3. 컴파일,훈련
criterion = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(),lr=1e-4)

def train(model, criterion,optimizer,loader):
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch,y_batch in loader:
        x_batch,y_batch = x_batch.to(DEVICE),y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis,y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis,1)
        acc = (y_predict == y_batch).float().mean()
        epoch_acc += acc
    return epoch_loss/len(loader),epoch_acc/len(loader)
# hist = model.fit(x_train)     # hist에는 loss 와 acc가 들어가있다.
# 최종값만 반환하기 때문에 hist라고 보기에는 어렵다.
    
def evaluate(model,criterion,loader):
    model.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch,y_batch in loader:
            x_batch,y_batch = x_batch.to(DEVICE),y_batch.to(DEVICE)
            hypothesis = model(x_batch)
            loss = criterion(hypothesis,y_batch)
            epoch_loss += loss.item()
            
            y_predict = torch.argmax(hypothesis,1)
            acc = (y_predict == y_batch).float().mean()
            epoch_acc += acc
    return epoch_loss/len(loader),epoch_acc/len(loader)

epochs = 29
for epoch in range(1, epochs + 1):
    loss,acc = train(model,criterion,optimizer,train_loader)
    val_loss,val_acc = evaluate(model,criterion,test_loader)
    print('epoch : {}, loss : {:.4f} ,acc : {:.3f}, val_loss : {:.4}, val_acc : {:.3f}'.format(
        epoch,loss,acc,val_loss,val_acc))

# epoch : 29, loss : 3.4772 ,acc : 0.169, val_loss : 3.551, val_acc : 0.161
