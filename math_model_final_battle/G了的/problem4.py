import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity,calculate_kmo
def data_mean():
    df = pd.read_csv('../第二问/所有特征整合数据.csv')
    result = pd.DataFrame(columns=df.columns)

    year = 2012
    rindex = 0
    while True:
        if year == 2022:
            break
        data = df.loc[df['年份']==year,:]
        result.loc[rindex] = data.mean()
        rindex += 1
        year += 1
        print('s')
    result.to_csv('数据集/整理数据/all_data_ym_mean.csv')

def pca_method():
    pindex = 1
    for name in ['people', 'weather', 'ground']:
        df = pd.read_csv('数据集/整理数据/' + name + '.csv')
        df0 = df.iloc[:, 2:]
        # 先归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        df0 = scaler.fit_transform(df0)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # n_components 指明了降到几维
        pca = PCA(n_components=1)
        # 利用数据训练模型（即上述得出特征向量的过程）
        pca.fit(df0)
        result = pd.DataFrame(pca.transform(df0))
        result.to_csv('pca_' + name + '.csv')
        # 得出原始数据的降维后的结果；也可以以新的数据作为参数，得到降维结果。
        # print(pca.transform(X))
        # 打印各主成分的方差占比
        ratio = pca.explained_variance_ratio_
        ratio_sum = np.cumsum(ratio)
        print(ratio)
        print("pca.components_", pca.components_.shape)
        print("pca_var_ratio", pca.explained_variance_ratio_.shape)
        # # 绘制图形
        # plt.subplot(3, 1, pindex)
        #
        # plt.plot([i for i in range(1, df0.shape[1]+1)],
        #          ratio_sum)
        # plt.scatter(range(1, df0.shape[1]+1), ratio_sum, marker='*')
        # plt.xticks(range(1,8))
        # plt.yticks(np.arange(0, 1.2, 0.2))
        #
        # if pindex == 1:
        #     plt.title('人文因素', fontsize=16)
        # elif pindex == 2:
        #     plt.title('气象因素', fontsize=16)
        # else:
        #     plt.title('地表因素', fontsize=16)
        #
        # plt.grid()
        #
        # pindex += 1
    # plt.suptitle('不同因素的不同特征携带信息量', fontsize=16)
    plt.show()
def factor_method(df0):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    chi_square_value, p_value = calculate_bartlett_sphericity(df0)
    print(chi_square_value, p_value)
    kmo_all, kmo_model = calculate_kmo(df0)
    print(kmo_model)

if __name__ == '__main__':
    # data_mean()
    pca_method()
    # factor_method(df0)
