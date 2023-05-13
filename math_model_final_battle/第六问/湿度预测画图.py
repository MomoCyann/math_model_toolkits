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
    plt.plot(train, label='真实值', color='deepskyblue')
    plt.plot(pre, label='预测值', color='orange',linestyle='--')
    if column == 'cp':
        plt.ylabel('植被覆盖率', fontsize=16)
    else:
        plt.ylabel(column, fontsize=16)
    plt.legend(prop={'size': 10})
    plt.grid(True,axis='y')
    if column == 'cp':
        plt.title('植被覆盖率预测',fontsize=16)
    else:
        plt.title(column+'预测', fontsize=16)
    plt.xticks(range(0, df.shape[0], 4), df.loc[range(0, df.shape[0], 4), 'date'],
               rotation=45)

def draw_shidu():
    df = pd.read_csv('湿度画图所需数据.csv')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    df = df.iloc[:141]
    train = df.loc[:123, 'train']
    pred = df.loc[122:, 'pred']

    plt.plot(train, label='真实值', color='deepskyblue')
    plt.plot(pred, label='预测值', color='orange',linestyle='--')
    plt.ylabel('10cm湿度(kg/m2)', fontsize=16)
    plt.title('10cm湿度(kgm2)预测', fontsize=16)
    plt.legend()
    plt.grid(True,axis='y')
    plt.xticks(range(0, df.shape[0], 4), df.loc[range(0, df.shape[0], 4), 'date'],
               rotation=45)

if __name__ == '__main__':
    pindex = 1
    df = pd.read_csv('../第二问/所有特征整合数据.csv')
    df = df.iloc[:141,:]
    columns = ['降水量(mm)', '土壤蒸发量(mm)', '植被指数(NDVI)']
    # columns = ['icstore']
    # columns = ['40cm湿度(kgm2)', '100cm湿度(kgm2)', '200cm湿度(kgm2)']

    columns=['cp']
    for column in columns:
        draw(column)
        plt.show()
    plt.subplot(4, 1, 1)
    draw_shidu()
    pindex = 2
    columns = ['40cm湿度(kgm2)', '100cm湿度(kgm2)', '200cm湿度(kgm2)']
    for column in columns:
        plt.subplot(4, 1, pindex)
        draw(column)
        pindex +=1


    plt.show()
        # statistic, pvalue = stats.mstats.ttest_ind(df.loc[122:143, column], df.loc[1:122, column])
        # print(statistic, pvalue)
    # draw_final()