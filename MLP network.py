from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('3des_runs.csv')

Target = ['label']
train_x = ['Distribution1','Distribution2','Distribution3','Distribution4','Distribution5','Distribution6','Distribution7','Distribution8','Distribution9','Distribution10','Distribution11','Distribution12','Distribution13','Distribution14','Distribution15','Distribution16','Distribution17','Distribution18','Distribution19','Distribution20','Distribution21','Distribution22']#,'distribution23','distribution24','distribution25','distribution26','distribution27','distribution28','distribution29','distribution30','distribution31','distribution32','distribution33','distribution34','distribution35','distribution36','distribution37','distribution38','distribution39','distribution40','distribution41','distribution42','distribution43','distribution44','distribution45','distribution46','distribution47','distribution48','distribution49','distribution50','distribution51','distribution52','distribution53','distribution54','distribution55','distribution56','distribution57','distribution58','distribution59','distribution60','distribution61','distribution62','distribution63','distribution64']

scale = StandardScaler(with_mean=True,with_std=True)
data[train_x] = scale.fit_transform(data[train_x])
traindata_x,valdata_x,traindata_y,valdata_y = train_test_split(data[train_x],data[Target],test_size=0.25,random_state=1)

MLP = MLPClassifier(hidden_layer_sizes=(40,40,40),activation='relu',alpha=0.0001,solver='adam',learning_rate='adaptive',max_iter=200,random_state=40,verbose=True,validation_fraction=0.25)
MLP.fit(traindata_x,traindata_y)
print(MLP.score(valdata_x,valdata_y))
#全连接神经网络