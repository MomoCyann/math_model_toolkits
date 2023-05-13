import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pylab
import seaborn as sns
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

def draw(column):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    train = df.loc[:123,column]
    pre = df.loc[122:, column]
    plt.plot(train, label='真实值')
    plt.plot(pre, label='预测值')
    plt.ylabel(column, fontsize=16)
    plt.legend(prop={'size': 10})
    plt.grid(True,axis='y')
    plt.title(column+'预测',fontsize=16)
    plt.xticks(range(0, df.shape[0], 3), df.loc[range(0, df.shape[0], 3), 'date'],
               rotation=45)

def draw_final():
    df = pd.read_csv('10cm湿度画图数据.csv')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    train = df.loc[:123, 'train']
    test = df.loc[91:123, 'test']
    pred = df.loc[122:, 'pred']

    plt.figure(figsize=(12, 8))
    plt.plot(train, label='真实值')
    plt.plot(pred, label='预测值')
    plt.plot(test, label='测试值')
    plt.ylabel('10cm湿度(kg/m2)')
    plt.xlabel("日期")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, df.shape[0], 3), df.loc[range(0, df.shape[0], 3), 'date'],
               rotation=45)

def draw_shidu_original():
    df = pd.read_csv('所有特征整合数据.csv')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    shidu10 = df.loc[:123, '10cm湿度(kgm2)']
    shidu40 = df.loc[:123, '40cm湿度(kgm2)']
    shidu100 = df.loc[:123, '100cm湿度(kgm2)']
    shidu200 = df.loc[:123, '200cm湿度(kgm2)']

    plt.plot(shidu10, label='10cm')
    plt.plot(shidu40, label='40cm')
    plt.plot(shidu100, label='100cm')
    plt.plot(shidu200, label='200cm')
    plt.ylabel('湿度(kg/m2)',fontsize=16)
    # plt.xlabel("日期")
    plt.legend(prop={'size': 16})
    plt.grid(True,axis='y')
    plt.xticks(range(0, df.shape[0], 3), df.loc[range(0, df.shape[0], 3), 'date'],
               rotation=45)
    plt.show()
if __name__ == '__main__':
    # pindex = 1
    # df = pd.read_csv('数据集/整理数据/所有特征整合数据.csv')
    # columns = ['降水量(mm)', '土壤蒸发量(mm)', '植被指数(NDVI)']
    # # columns = ['icstore']
    # # columns = ['40cm湿度(kgm2)', '100cm湿度(kgm2)', '200cm湿度(kgm2)']
    #
    # columns=['cp']
    #
    # for column in columns:
    #     plt.subplot(3, 1, pindex)
    #     draw(column)
    #     pindex+=1
    # plt.show()
    #     # statistic, pvalue = stats.mstats.ttest_ind(df.loc[122:143, column], df.loc[1:122, column])
    #     # print(statistic, pvalue)
    # draw_final()
    draw_shidu_original()



