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
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

def sarima_regression():
    # 读取数据和预设
    df = pd.read_csv('../数据集/整理数据/14-化学性质-orig.csv')

    all_plot = ['G17','G19','G21','G6','G12','G18','G8','G11','G16','G9','G13','G20']

    columns = ['SOC',
    'SIC',
    'N',]

    result = pd.DataFrame(columns=df.columns)
    rindex = 0
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    for plot in all_plot:
        df0 = df.loc[df['plot']==plot, :]
        result.loc[rindex, 'plot']=plot
        for column in columns:
            train = np.array(df0[column])
            model=ARIMA(train,order=(1,1,1)).fit()
            pred= model.forecast(1)
            result.loc[rindex, column] = pred[0]
            # df.loc[df['放牧小区（plot）']==plot, 'pred'
            plt.plot(train, label=column)
            plt.plot(pred)
        plt.ylabel(column)
        plt.xlabel("日期")
        # plt.xticks(range(0, df.shape[0], 3), df.loc[range(0, df.shape[0], 3), 'date'], rotation=45)
        # plt.show()
        print('1')
        rindex+=1
    result.to_csv('化学性质ARIMA预测.csv')

def tt_check():
    df = pd.read_excel('数据集/整理数据/化学性质数据.xlsx')
    all_plot = ['G17', 'G19', 'G21', 'G6', 'G12', 'G18', 'G8', 'G11', 'G16', 'G9', 'G13', 'G20']
    columns = ['SOC','SIC','STC','N','C/N']

    def tt_test(column):
        statistic, pvalue = stats.mstats.ttest_ind(df.loc[122:143, column], df.loc[0:122, column])
        print(statistic, pvalue)

    df.sort_values(by=['plot', 'year', 'intensity'], inplace=True)

    result = pd.DataFrame(columns=df.columns)
    rindex = 0
    for plot in all_plot:
        df0 = df.loc[df['plot']==plot, :]
        df0.reset_index(drop=True, inplace=True)
        result.loc[rindex, 'plot'] = plot
        for column in columns:
            df00 = df0.loc[:, column]
            statistic, pvalue = stats.mstats.ttest_1samp([10,8,7,6,4], [9])
            result.loc[rindex, column] = pvalue
        rindex += 1
    print('1')

if __name__ == '__main__':
    sarima_regression()
    # tt_check()
