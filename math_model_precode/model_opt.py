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
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

    clf_rf = RandomForestClassifier(random_state=0)
    clf_rf = clf_rf.fit(X_train, y_train)
    y_pred = clf_rf.predict(X_test)
    y_pred_proba = clf_rf.predict_proba(X_test)

    print(classification_report(y_test, y_pred, digits=5),
          "\nAUC =", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
          "\n训练完成")
    return clf_rf

def get_range(data):
    # 获取了数据的范围，方便启发式搜索
    max_value = np.max(data,axis=0)
    min_value = np.min(data,axis=0)
    return list(min_value), list(max_value)

# 继承并修改包的类，方便保存优秀个体
class DEPlus(DE):
    def __init__(self, func, n_dim, model, F=0.5,
                 size_pop=50, max_iter=200, prob_mut=0.3,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        super(DEPlus, self).__init__(func, n_dim, F,
                 size_pop, max_iter, prob_mut,
                 lb, ub,
                 constraint_eq=tuple(), constraint_ueq=tuple())
        self.F = F
        self.V, self.U = None, None
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.crtbp()
        # 新增一个保存优秀个体的列表
        self.good_samples = []
        self.model = model

    def saving(self):
        # 新方法：用于每次迭代保存符合条件的个体
        for i in range(self.size_pop):
            if self.model.predict(self.X[i].reshape(1, -1))[0] >= 2:
                self.good_samples.append(self.X[i, :].copy())

    def run(self, max_iter=None):
        # 重写run方法，在需要的位置加入了saving方法保存符合条件的个体
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.selection()
            # record the best ones
            generation_best_index = self.Y.argmin()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))

        self.saving()
        print('good samples length is', len(self.good_samples))
        result_range_min, result_range_max = get_range(np.array(self.good_samples))
        print(result_range_min,'\n', result_range_max)
        return self.best_x, self.best_y
def main():
    # 训练一个分类器模型
    model = train()
    # 读取数据，优化算法产生
    iris = datasets.load_iris()
    # 数据是 个数 x 维度格式的array
    '''
    21D的思路是以5种特质作为目标函数，5个二分类3个以上是1加起来大于等于3，在最后一代的所有样本里
    排序得到活性比较高的
    这里演示就以鸢尾花的标签作为例子，标签大于等于1即可，再取等于2的所有样本，得到范围。
    '''

    # 获取范围 lb, ub
    lb, ub = get_range(iris.data)
    print(lb,'\n', ub)

    # 默认是最小化
    # 目标函数
    def obj_func(x):
        # x 是一个列表
        x = x.reshape(1, -1)
        return -model.predict(x)[0]

    de = DEPlus(func=obj_func, n_dim=4, model=model, size_pop=100, max_iter=5, lb=lb, ub=ub)

    best_x, best_y = de.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)

    # pso = PSO(func=obj_func, n_dim=4, pop=100, max_iter=100, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
    # pso.record_mode = True
    # pso.run()
    # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

if __name__ == "__main__":
    main()

