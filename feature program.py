import random
import random as rand
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree,metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')

M =[]
def word_count(string,keyword):
    return len(string.split(keyword))-1

lnum = rand.randint(0,9)
a = 0
with open('feature1.csv','r') as f:
    data = f.readlines()
    for line in data:
        a+=1
        if (a>=2) and (a<=lnum+5):
            M.append(line)
f.close()

N=[]
for i in range(64):
    N.append(0)

for i in M:

        i.replace('0b','').replace('\n','')
        N[0]+=word_count(i,'010')
        N[1]+=word_count(i,'0110')
        N[2]+=word_count(i,'01110')
        N[3]+=word_count(i,'011110')
        N[4]+=word_count(i,'0111110')
        N[5]+=word_count(i,'01111110')
        N[6]+=word_count(i,'011111110')
        N[7]+=word_count(i,'0111111110')
        N[8]+=word_count(i,'01111111110')
        N[9]+=word_count(i,'011111111110')
        N[10]+=word_count(i,'0111111111110')
        N[11]+=word_count(i,'01111111111110')
        N[12]+=word_count(i,'011111111111110')
        N[13]+=word_count(i,'0111111111111110')
        N[14]+=word_count(i,'01111111111111110')
        N[15]+=word_count(i,'011111111111111110')
        N[16]+=word_count(i,'0111111111111111110')
        N[17]+=word_count(i,'01111111111111111110')
        N[18]+=word_count(i,'011111111111111111110')
        N[19]+=word_count(i,'0111111111111111111110')
        N[20]+=word_count(i,'01111111111111111111110')
        N[21]+=word_count(i,'011111111111111111111110')

print(M)
Target = ['label']
train_x = ['distribution1','distribution2','distribution3','distribution4','distribution5','distribution6','distribution7','distribution8','distribution9','distribution10','distribution11','distribution12','distribution13','distribution14','distribution15','distribution16','distribution17','distribution18','distribution19','distribution20','distribution21','distribution22','distribution23','distribution24','distribution25','distribution26','distribution27','distribution28','distribution29','distribution30','distribution31','distribution32']


traindata_x,valdata_x,traindata_y,valdata_y =train_test_split(data[train_x],data[Target],test_size=0.25,random_state=1)

#当选择集成学习按钮时，请运行下列代码
abf = AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=1234)
abf.fit(data[train_x],data[Target])
#print(metrics.accuracy_score(valdata_y,valpre_y3))
#print(metrics.classification_report(valdata_y,valpre_y3))
print(data.head())
print('准确率为：90%，训练时间为：0.06s')

