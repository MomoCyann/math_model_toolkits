from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sko.DE import DE
from sko.PSO import PSO

def train():
    iris=datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)


    clf_rf=RandomForestClassifier(random_state=0)
    clf_rf=clf_rf.fit(X_train, y_train)
    y_pred = clf_rf.predict(X_test)
    y_pred_proba = clf_rf.predict_proba(X_test)

    print(classification_report(y_test, y_pred, digits=5),
          "\nAUC =",roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
          "\n训练完成")
    return clf_rf

def predict(model, test):
    pred = model.predict(test)
    return pred


def get_range():
    # 获取了数据的范围，方便启发式搜索
    iris = datasets.load_iris()
    test = iris.data
    max_value = np.max(test,axis=0)
    min_value = np.min(test,axis=0)
    return list(min_value), list(max_value)


def main():
    # 训练一个分类器模型
    model = train()
    # 读取数据，优化算法产生
    iris = datasets.load_iris()
    # 数据是 个数 x 维度格式的array
    test = iris.data
    # 返回预测的标签
    print(predict(model, test))

    '''
    21D的思路是以5种特质作为目标函数，5个二分类3个以上是1加起来大于等于3，在最后一代的所有样本里
    排序得到活性比较高的
    这里演示就以鸢尾花的标签作为例子，标签大于等于1即可，再取等于2的所有样本，得到范围。

    '''

    # 获取范围 lb, ub
    lb, ub = get_range()

    # 空表 创建
    good_samples = pd.DataFrame(data=None, columns=['1', '2', '3', '4'])

    # 默认是最小化
    # 目标函数
    def obj_func(x):
        global good_samples
        # x 是一个列表
        # 封装的包不自带输出当时的特征，手动记录
        x = x.reshape(1, -1)
        if predict(model, x)[0] >= 2:
            # 保存这个满足条件的x
            temp = pd.DataFrame(x, columns=list(good_samples.columns))
            good_samples = pd.concat([good_samples, temp], ignore_index=True)
            print("喜加一")
        return -predict(model, x)[0]

    de = DE(func=obj_func, n_dim=4, size_pop=100, max_iter=5, lb=lb, ub=ub)

    best_x, best_y = de.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)

    # pso = PSO(func=obj_func, n_dim=4, pop=100, max_iter=100, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
    # pso.record_mode = True
    # pso.run()
    # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    print(good_samples.info())

if __name__ == "__main__":
    main()

