import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree,metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
data = pd.read_csv('mix_fre_more.csv')

Target = ['label']
train_x = ['Distribution1','Distribution2','Distribution3','Distribution4','Distribution5','Distribution6','Distribution7','Distribution8','Distribution9','Distribution10','Distribution11','Distribution12','Distribution13','Distribution14','Distribution15','Distribution16','Distribution17','Distribution18','Distribution19','Distribution20','Distribution21','Distribution22','Distribution23','Distribution24','Distribution25','Distribution26','Distribution27','Distribution28','Distribution29','Distribution30']#,'Distribution31','Distribution32']#,'distribution33','distribution34','distribution35','distribution36','distribution37','distribution38','distribution39','distribution40','distribution41','distribution42','distribution43','distribution44','distribution45','distribution46','distribution47','distribution48','distribution49','distribution50','distribution51','distribution52','distribution53','distribution54','distribution55','distribution56','distribution57','distribution58','distribution59','distribution60','distribution61','distribution62','distribution63','distribution64']
'''
'distribution5','distribution6','distribution7','distribution8','distribution9','distribution10','distribution11','distribution12','distribution13','distribution14','distribution15','distribution16','distribution17','distribution18','distribution19','distribution20','distribution21','distribution22','distribution23','distribution24','distribution25','distribution26','distribution27','distribution28','distribution29','distribution30','distribution31','distribution32','distribution33','distribution34','distribution35','distribution36','distribution37','distribution38','distribution39','distribution40','distribution41','distribution42','distribution43','distribution44','distribution45','distribution46','distribution47','distribution48','distribution49','distribution50','distribution51','distribution52','distribution53','distribution54','distribution55','distribution56','distribution57','distribution58','distribution59','distribution60']
'''
traindata_x,valdata_x,traindata_y,valdata_y = train_test_split(data[train_x],data[Target],test_size=0.2,random_state=1)


rf = RandomForestClassifier(n_estimators=64,max_depth=15,oob_score=True,random_state=1)
rf.fit(traindata_x,np.ravel(traindata_y))
print(rf.oob_score_)


valpre_y3 = rf.predict(valdata_x)
print(metrics.accuracy_score(valdata_y,valpre_y3))
print(metrics.classification_report(valdata_y,valpre_y3))

#当选择集成学习按钮时，请运行下列代码
'''
abf = AdaBoostClassifier(n_estimators=2,learning_rate=1,random_state=1234)
abf.fit(data[train_x],np.ravel(data[Target]))
valpre_y3 = abf.predict(valdata_x)
print(metrics.accuracy_score(valdata_y,valpre_y3))
print(metrics.classification_report(valdata_y,valpre_y3))
'''
'''
clf =tree.DecisionTreeClassifier(random_state=1)
clf = clf.fit(traindata_x,traindata_y)
valpre_y = clf.predict(valdata_x)
print(metrics.accuracy_score(valdata_y,valpre_y))
print(metrics.classification_report(valdata_y,valpre_y))
'''