import time
import pandas as pd
import sklearn.metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text, export_graphviz
import pydotplus
import os


trade_interval=5
predict_interval=5

# 读取feature
df = pd.read_csv('data/HS300/feature/{}_{}/all_feature_{}.csv'.format(trade_interval,predict_interval,trade_interval))

print(df.info())
print(df.columns)

# 对不在时间范围内的数据进行处理
s_date = datetime.datetime.strptime('20210101', '%Y%m%d').date()
e_date = datetime.datetime.strptime('20220101', '%Y%m%d').date()
df['selectedDate'] =pd.to_datetime(df['selectedDate'])
# df=df[(df['selectedDate'].dt.date > s_date) & (df['selectedDate'].dt.date < e_date)]
df = df[df['newsCount']>5]
df.drop(columns=['tradeStartDate','selectedDate'],axis=1,inplace=True)
# df.drop(columns=['tradeStartDate','tradeEndDate'],axis=1,inplace=True)
# df.drop(index=df[df['newsCount']<10].index,axis=0,inplace=True)
print(df.info())


# 选取需要的列与标签
# 因子从5-268
# 舆情列从268-280 （280不含） 标签281.282

# x是做训练与测试  x_n保持顺序，做顺序的测试
x= df.iloc[:1500,272:273]
x_n=df.iloc[1500:,272:273]
features_columns=x.columns

print(features_columns)

y=df.iloc[:1500,282]
y_n=df.iloc[1500:,282]
print(y[y==1].count())
print(y[y==0].count())

# 标准化数据
x = np.array(x)
x = preprocessing.scale(x)

# 5次交叉检验
skf=StratifiedKFold(n_splits=5,shuffle=True)

# acc与f1
acc_rf=[]
f1_rf=[]

for train_index, test_index in skf.split(x,y):
    # print(test_index)
    x_train = x[train_index]
    x_test = x[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    #  随机森林
    # model= RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=10)
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
    print('这里是x_n的数据',model.score(x_n,y_n))

    # LR
    model_lr=LogisticRegressionCV()
    model_lr.fit(x_train,y_train)
    print('lr',model_lr.score(x_test,y_test))
    # print(model_lr.predict(x_test)[:10])

    #MLP
    model_mlp=MLPClassifier()
    model_mlp.fit(x_train,y_train)
    print('mlp',model_mlp.score(x_test,y_test))
    print(model_mlp.score(x_n,y_n))

    #XGB
    model_xg=XGBClassifier()
    model_xg.fit(x_train,y_train)
    print('xgb',model_xg.score(x_test,y_test))
    print(model_xg.score(x_n, y_n))

    #lgb
    model_lgb=LGBMClassifier()
    model_lgb.fit(x_train,y_train)
    print('lgb',model_lgb.score(x_test,y_test))
    print(model_lgb.score(x_n, y_n))

    #DT
    model_dt = DecisionTreeClassifier()
    model_dt.fit(x_train,y_train)
    print('dt',model_dt.score(x_test,y_test))
    print(model_dt.score(x_n, y_n))


    # # 对随机森林进行解析
    # # 第0棵树
    # tree_0 = model.estimators_[0]
    # # 查看第0棵树的各项信息
    # n_nodes = tree_0.tree_.node_count
    # print(f"n_nodes = {n_nodes}\n")
    #
    # children_left = tree_0.tree_.children_left  # 左子节点的id，-1代表其parent无左子节点
    # print(f"children_left = {children_left}\n")  # 右子节点的id，-1代表其parent无子节点
    #
    # children_right = tree_0.tree_.children_right
    # print(f"children_right = {children_right}\n")
    #
    # features = tree_0.tree_.feature  # 特征所在序号
    # print(f"features = {features}\n")
    #
    # threshold = tree_0.tree_.threshold  # 特征的切分阈值
    # print(f"threshold = {threshold}\n")
    #
    # # 数据的特征名
    # print(f"features_names = {features_columns}")
    #
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    #
    # # 绘图并导出
    # dot_data = export_graphviz(tree_0, out_file=None,
    #                            feature_names=features_columns)
    #
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    # if os.path.exists("out_0.pdf"):
    #     os.remove("out_0.pdf")
    #     graph.write_pdf("out_0.pdf")
    # else:
    #     graph.write_pdf("out_0.pdf")  # 当前文件夹生成out.png


print(np.mean(acc_rf))
print(np.mean(f1_rf))


