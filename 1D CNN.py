import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'mix_fre_more.csv')
#将数据Species列由字母转换成数字
#data['Species']=pd.factorize(data.Species)[0]
#X不要第零列,不要最后一列 第一个:代表全部的行 第二个:代表列
X = data.iloc[:,:-1].values
#Y只要特定一列
Y = data.label.values
train_x,test_x,train_y,test_y=train_test_split(X,Y)
#将数据转换成Tensor LongTensor等价于int64
train_x = torch.from_numpy(train_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.int64)
test_x = torch.from_numpy(test_x).type(torch.float32)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)

#数据只有150行故batch也要小一点
batch = 5
no_of_batches = len(data)//batch
epochs = 200

#TensorDataset()可以对tensor进行打包即合并
train_ds = TensorDataset(train_x,train_y)
#希望模型不关注训练集数据顺序故用乱序
train_dl = DataLoader(train_ds,batch_size=batch,shuffle=True)
test_ds = TensorDataset(test_x,test_y)
#对测试集不需要用乱序避免工作量增加
test_dl = DataLoader(test_ds,batch_size=batch)

#创建模型
#继承nn.Module这个类并自定义模型
class Model(nn.Module):
    #定义初始化方法
    def __init__(self):
        #继承父类所有属性
        super().__init__()
        #(初始化第一层 输入到隐藏层1)4个输入特征 32个输出特征
        self.liner_1 = nn.Linear(30,32)
        #(初始化第二层 隐藏层1到隐藏层2)32个输入特征 32个输出特征
        self.liner_2 = nn.Linear(32,32)
        #(初始化第三层 隐藏层2到输出层)32个输入特征 3个输出特征
        self.liner_3 = nn.Linear(32,16)
        self.liner_4 = nn.Linear(16,7)
        #使用F不需要初始化激活层

    #定义def forward会调用上述这些层来处理input
    def forward(self,input):
        #在第一层上对input进行调用 并激活
        x = F.relu(self.liner_1(input))
        #在第二层上对input进行调用 并激活
        x = F.relu(self.liner_2(x))
        #多分类任务并不激活，也有人写成如下所示，用softmax激活
        #x = F.softmax(self.liner_3(x))
        #首先，nn.CrossEntropyLoss()函数的输入是一个未激活的输出
        #其次，多项分类可以使用softmax函数进行激活
        #然后，使用softmax()函数进行激活是将结果映射到一个0到1的概率分布上
        #最后，如果不用softmax激活，输出最大的值依然是那个概率最高的值
        x = F.relu(self.liner_3(x))
        x = self.liner_4(x)
        return x

model = Model()
#损失函数
loss_fn = nn.CrossEntropyLoss()

def accuracy(y_pred,y_true):
    #torch.argmax将数字转换成真正的预测结果
    y_pred = torch.argmax(y_pred,dim=1)
    acc = (y_pred == y_true).float().mean()
    return acc

#便于随着训练的进行观察数值的变化
train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]

def get_model():
    #获得这个模型
    model = Model()
    #优化函数 优化的是模型所有变量即model.parameters()
    opt = torch.optim.Adam(model.parameters(),lr=0.0001)
    return model,opt

model,optim = get_model()

for epoch in range(epochs):
    for x,y in train_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # 梯度置为0
        optim.zero_grad()
        # 反向传播求解梯度
        loss.backward()
        # 优化
        optim.step()
    # 不需要进行梯度计算
    with torch.no_grad():
        epoch_accuracy = accuracy(model(train_x),train_y)
        epoch_loss = loss_fn(model(train_x), train_y).data
        epoch_test_accuracy = accuracy(model(test_x),test_y)
        epoch_test_loss = loss_fn(model(test_x), test_y).data
        print('epoch: ',epoch,'train_loss: ',round(epoch_loss.item(),4),'train_accuracy: ',round(epoch_accuracy.item(),4),
             'test_loss: ',round(epoch_test_loss.item(),4),'test_accuracy: ',round(epoch_test_accuracy.item(),4)
              )
        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_accuracy)

plt.plot(range(1,epochs+1),train_loss,label='train_loss')
plt.plot(range(1,epochs+1),test_loss,label='test_loss')
plt.plot(range(1,epochs+1),train_acc,label='train_acc')
plt.plot(range(1,epochs+1),test_acc,label='test_acc')
plt.show()
#CNN