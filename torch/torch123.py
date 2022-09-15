# 파이토치에서는 텐서플로처럼 지피유에 따로 설치하지않아도 된다.

import numpy as np
import torch
import torch.nn as nn #뉴럴네트워크
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA=torch.cuda.is_available() #쿠다에서 사용가능한 애들을 (대문자)유즈쿠다로 할게
DEVICE=torch.device('cuda'if USE_CUDA else 'cpu') #쿠다를 쓸 수 있르면 쓰고 안되면 씨피유로 할게
# print('torch:',torch.__version__,'사용DEVICE:',DEVICE)
# torch: 1.12.1 사용DEVICE: cuda
# 데이터와 모델에서만 정해주면 된다.

# 1.데이터 
x=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1, 1.1,1.2 ,1.3 ,1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]) #(2, 10)
y=np.array([11,12,13,14,15,16,17,18,19,20]) #(10,)
x_test=np.array([10,1.3]) #(2, )

# x=x.transpose() #(10, 2)
# x_test=x_test.transpose()

# print(x.shape) 

# 텐서형태로 바꿔주기
x=torch.FloatTensor(np.transpose(x)).to(DEVICE) 
y=torch.FloatTensor(y).unsqueeze(-1).to(DEVICE) 
x_test=torch.FloatTensor(np.transpose(x_test)).to(DEVICE)

print(x,y,x_test)

x_test=(x_test-torch.mean(x))/torch.std(x)
x=(x-torch.mean(x))/torch.std(x) # =StandardScaler

print(x,x_test)

# 2.모델구성
model=nn.Sequential(nn.Linear(2,5),
                    nn.Linear(5,3),
                    nn.Linear(3,4),
                    nn.Linear(4,2),
                    nn.Linear(2,1)).to(DEVICE)


# 3.컴파일 훈련
criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01) 

def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad() 
    
    hypothesis=model(x)
    loss=criterion(hypothesis,y) # =mse
    
    # loss=nn.MSELoss(hypothesis,y) #Boolean value of Tensor with more than one value is ambiguous
    
    # loss=nn.MSELoss()(hypothesis,y)
    
    # loss=F.mse_loss(hypothesis,y)
    
    # loss_func=nn.MSELoss()
    # loss=loss_func(hypothesis,y)
    
    loss.backward() 
    optimizer.step() 
    return loss.item()

epochs=2000
for epoch in range(1,epochs+1):
    loss=train(model,criterion,optimizer,x,y)
    print('epoch:{},loss:{}'.format(epoch,loss))
    
    
# 4. 평가, 예측
def evaluate(model,criterion,x,y):
    model.eval() 
    
    with torch.no_grad(): 
        y_predict=model(x) 
        results=criterion(y_predict,y) 
    return results.item()

loss2=evaluate(model,criterion,x,y)
print('평가에 대한 로스(최종 loss):',loss2)
results=model(torch.Tensor(x_test).to(DEVICE))
print('예측값:',results.item())