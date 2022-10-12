import time
import pandas_method as pd
import sklearn.metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text, export_graphviz



trade_interval=10
predict_interval=10

# 读取feature
df = pd.read_csv('data/HS300/feature/顺丰控股_feature_5.csv'.format(trade_interval,predict_interval,trade_interval))

print(df.info())
print(df.columns)

# 对不在时间范围内的数据进行处理
s_date = datetime.datetime.strptime('20210101', '%Y%m%d').date()
e_date = datetime.datetime.strptime('20220101', '%Y%m%d').date()
df['selectedDate'] =pd.to_datetime(df['selectedDate'])
# df=df[(df['selectedDate'].dt.date > s_date) & (df['selectedDate'].dt.date < e_date)]
# df = df[df['newsCount']>50]
df.drop(columns=['tradeStartDate','selectedDate'],axis=1,inplace=True)
# df.drop(columns=['tradeStartDate','tradeEndDate'],axis=1,inplace=True)
# df.drop(index=df[df['newsCount']<10].index,axis=0,inplace=True)
print(df.info())


# 选取需要的列与标签
# 因子从5-268
# 舆情列从268-280 （280不含） 标签281.282

# x是做训练与测试  x_n保持顺序，做顺序的测试
x= df.iloc[:300,53:58]
x_n=df.iloc[300:,53:58]
features_columns=x.columns

print(features_columns)

y=df.iloc[:300,-1]
y_n=df.iloc[300:,-1]
print(y[y==1].count())
print(y[y==0].count())

#train test
x_train = x.iloc[:240,:]
x_test = x.iloc[240:,:]

y_train = y.iloc[:240]
y_test = y.iloc[240:]

# 标准化数据
x = np.array(x)
# x = preprocessing.scale(x)

# acc与f1
acc_rf=[]
f1_rf=[]
acc_lr=[]
f1_lr=[]

#LSTM模型参数
#  随机森林
# LSTM模型参数= RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=10)
model= RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
y_pre = model.predict(x_test)

#特征重要性
feature_importances = model.feature_importances_
features_df = pd.DataFrame({'Features': features_columns, 'Importance': feature_importances})
features_df.sort_values('Importance', inplace=True, ascending=False)
# features_df.reset_index(drop=True,inplace=True)
# features_df.to_csv('data/HS300/feature/feature_importance.csv')

print(features_df[:10])
print(features_df[:10].sum())

# 画出特征重要性图
# plt.plot(range(features_df.shape[0]),features_df['Importance'])
# plt.show()

acc_rf.append(model.score(x_test, y_test))
f1_rf.append(sklearn.metrics.f1_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None))
print(acc_rf,f1_rf)
print('rf这里是x_n的数据',model.score(x_n,y_n))
