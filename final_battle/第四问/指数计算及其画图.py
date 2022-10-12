import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from numpy import array
import seaborn as sns


def plot_desert():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    df = pd.read_csv('沙漠化指数结果.csv')
    df = df.iloc[:,1:]

    plt.figure(figsize=(12, 8))

    all_plot = [1,2,3,4]
    stre = [2.681, 2.133, 5.382, 2.492]

    for index in range(4):
        df_t = df.loc[df['牧户'] == all_plot[index], :]
        df_t.reset_index(drop=True, inplace=True)

        plt.plot(df_t['desert'], label='牧户'+str(all_plot[index])+':放牧强度'+str(stre[index]))

        plt.scatter(range(df_t.shape[0]), df_t['desert'], marker='*')
    # plt.ylim(0, 30)
    plt.title('四个牧户不同放牧强度下沙漠化指数', fontsize=16)
    plt.legend()
    plt.xticks(range(df_t.shape[0]), df_t['年份'], fontsize=12)
    plt.xlabel('年份', fontsize=16)
    plt.ylabel('沙漠化指数', fontsize=16)
    plt.grid(axis="y")
    plt.show()

def cal_desert2():
    df = pd.read_csv('沙漠化指数所需数据.csv')
    date = df.loc[:,'年份']
    plot = df.loc[:, '牧户']
    scaler = MinMaxScaler()
    df0 = scaler.fit_transform(df.iloc[:, 2:])
    weights = np.array([0.1802,0.0787,0.0685,0.2036,0.0808,0.1282,0.0509,0.1282,0.0808])
    a = weights * df0
    result = np.sum(weights * df0, axis=1)
    result_ = pd.DataFrame([plot, date, result]).T
    result_.columns = ['牧户', '年份', 'desert']
    result_.to_csv('沙漠化指数结果.csv')
    print('1')

def shangquan():
    df0 = pd.read_csv('土壤板结化指数所需数据.csv')
    df = df0.iloc[:, 5:8]
    # 定义熵值法函数
    def cal_weight(x):
        x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
        rows = x.index.size
        cols = x.columns.size
        k = 1.0 / math.log(rows)
        lnf = [[None] * cols for i in range(rows)]
        x = array(x)
        lnf = [[None] * cols for i in range(rows)]
        lnf = array(lnf)
        for i in range(0, rows):
            for j in range(0, cols):
                if x[i][j] == 0:
                    lnfij = 0.0
                else:
                    p = x[i][j] / x.sum(axis=0)[j]
                    lnfij = math.log(p) * p * (-k)
                lnf[i][j] = lnfij
        lnf = pd.DataFrame(lnf)
        E = lnf

        d = 1 - E.sum(axis=0)
        w = [[None] * 1 for i in range(cols)]
        for j in range(0, cols):
            wj = d[j] / sum(d)
            w[j] = wj
        w = pd.DataFrame(w)
        return w
    w = cal_weight(df)
    w.index = df.columns
    w.columns = ['weight']
    print(w)
    print('运行完成!')

def plot_bjh():
    df = pd.read_csv('土壤板结化指数所需数据.csv')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(12, 8))

    all_plot = [2018, 2020]

    for index in range(len(all_plot)):
        df_t = df.loc[df['年份'] == all_plot[index], :]
        df_t.reset_index(drop=True, inplace=True)

        plt.plot(df_t['bjh'], label=str(all_plot[index])+'年')

        plt.scatter(range(df_t.shape[0]), df_t['bjh'], marker='*')
    # plt.ylim(0, 30)
    plt.title('不同年份不同放牧强度下土壤板结化指数', fontsize=16)
    plt.legend(prop={'size': 16})
    plt.xticks(range(df_t.shape[0]), df_t['放牧强度'], fontsize=12)
    plt.xlabel('放牧强度', fontsize=16)
    plt.ylabel('土壤板结化指数', fontsize=16)
    plt.grid(axis="y")
    plt.show()

def plot_final():
    df = pd.read_csv('土壤健康指数结果.csv')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(12, 8))


    plt.plot(df['土壤健康指数'], color='deepskyblue')

    plt.scatter(range(df.shape[0]), df['土壤健康指数'], marker='*', color='salmon')
    # plt.ylim(0, 30)
    plt.title('2020年不同放牧强度下土壤健康指数', fontsize=16)
    plt.xticks(range(df.shape[0]), df['放牧强度'], fontsize=12)
    plt.xlabel('放牧强度', fontsize=16)
    plt.ylabel('土壤健康指数', fontsize=16)
    plt.grid(axis="y")
    plt.show()

if __name__ == '__main__':
    # cal_desert()
    #plot_desert()
    # cal_desert2()
    # shangquan()
    # plot_bjh()
    plot_final()