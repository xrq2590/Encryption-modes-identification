import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv("3des_runs_1.csv") #读取csv格式的文件全部数据
val_data = pd.read_csv('3des_runs_1.csv')
data1 = np.mat(data) #数据矩阵化
data2 = np.mat(val_data)
pre_raw = 30 #需要预测值所在行数
# print (data1)
def load_data_train():
    data_train = data1[:,:] # 取出数据的第1行到26行中的第4列到47列作为训练数据
    data_val = data2[:,:]
    #print(data_train)
    return data_train,data_val

def load_data_pre():
    data_pre = data1[pre_raw, :].astype('float64') #第31行第4列到第46列为输入数据
    data_mean = data_pre.mean() #求平均值
    data_std = data_pre.std() #求标准差
    data_pre = (data_pre - data_mean) / data_std #标准化
    return data_pre

def load_data_real():
    data_real = data1[pre_raw, ]  #取出第31行的第47列数据
    return data_real

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation



def Train_Model(data_train,data_val):
    #modelfile = './modelweight' #此位置保存训练模型过程中的权重
    #y_mean_std = "./y_mean_std.txt" # 保存标准化过程中的数据，后边数据还原需要用到
    data_train = np.matrix(data_train).astype('float64')
    data_val = np.matrix(data_val).astype('float64')
    data_mean = np.mean(data_train, axis=0)#对列求平均值
    data_std = np.std(data_train, axis=0)#计算每一列的标准差
    # data_train = (data_train - data_mean) / data_std
    print(1)
    x_train = data_train[:, 0:(data_train.shape[1] - 1)] #所有数据（除最后一列）作为输入x
    y_train = data_train[:, data_train.shape[1] - 1] #所有数据的最后一列作为输出y
    x_val = data_val[:, 0:(data_val.shape[1]-1)]
    y_val = data_val[:, data_val.shape[1]-1]
    #print(x_train)
    #print(y_train)
    #模型训练
    model = Sequential()
    model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer="uniform"))
    model.add(Activation('relu'))
    model.add(Activation('relu'))
    model.add(Dense(1, input_dim=x_train.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    history=model.fit(x_train, y_train, epochs=1000, batch_size=x_train.shape[0],validation_data=(x_val,y_val))
    #model.save_weights(modelfile) #保存模型权重
    y_mean = data_mean[:, data_train.shape[1] - 1]
    y_std = data_std[:, data_train.shape[1] - 1]
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs,acc,'b',label='Training accuracy')
    plt.plot(epochs,val_acc,'r',label='validation accuracy')
    plt.legend(loc='lower right')
    plt.figure()
    plt.plot(epochs,loss,'r',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='validation loss')
    plt.legend()
    plt.show()
    print("训练完毕")

Train_Model(data,val_data)

#BP神经网络