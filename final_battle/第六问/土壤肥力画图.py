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
            pred= model.forecast(2)
            result.loc[rindex, column] = pred[0]
            result.loc[rindex+1, column] = pred[1]
            # df.loc[df['放牧小区（plot）']==plot, 'pred'
            plt.plot(train, label=column)
            plt.plot(pred)
        plt.ylabel(column)
        plt.xlabel("日期")
        # plt.xticks(range(0, df.shape[0], 3), df.loc[range(0, df.shape[0], 3), 'date'], rotation=45)
        # plt.show()
        print('1')
        rindex+=2
    result.to_csv('6_youjiwu.csv')

def plot_14file():
    df = pd.read_excel('数据集/整理数据/化学性质原数据.xlsx')
    df2 = pd.read_csv('化学性质预测.csv')

    df.sort_values(by=['plot','year','intensity'],inplace=True)
    df.reset_index(drop=True,inplace=True)

    df2.sort_values(by=['plot', 'year', 'intensity'], inplace=True)
    df2.reset_index(drop=True, inplace=True)
    print(df.head())

    grid=[(0,1),(0,2),(0,3),
          (1,1),(1,2),(1,3),
          (2,1),(2,2),(2,3),
          (3,1),(3,2),(3,3)]
    g_i=0

    plt.figure(figsize=(12,8))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.4)

    all_plot = ['G17','G19','G21','G6','G12','G18','G8','G11','G16','G9','G13','G20']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    pindex = 1
    for t in all_plot:

        df_t=df.loc[df['plot']==t,:]
        df_t.reset_index(drop=True,inplace=True)

        df2_t = df2.loc[df2['plot'] == t, :]
        df2_t.reset_index(drop=True,inplace=True)


        plt.subplot(4, 3, pindex)

        plt.plot(df_t['SOC'],label='SOC',color=colors[0])
        plt.plot(df_t['SIC'],label='SIC',color=colors[1])
        plt.plot(df_t['STC'],label='STC',color=colors[2])
        plt.plot(df_t['N'],label='N',color=colors[3])
        plt.plot(df_t['C/N'],label='C/N',color=colors[4])

        plt.plot(df2_t['SOC'],linestyle='--',color=colors[0])
        plt.plot(df2_t['SIC'],linestyle='--',color=colors[1])
        plt.plot(df2_t['STC'],linestyle='--',color=colors[2])
        plt.plot(df2_t['N'],linestyle='--',color=colors[3])
        plt.plot(df2_t['C/N'],linestyle='--',color=colors[4])


        plt.scatter(range(df_t.shape[0]),df_t['SOC'],marker='*',color=colors[0])
        plt.scatter(range(df_t.shape[0]),df_t['SIC'],marker='*',color=colors[1])
        plt.scatter(range(df_t.shape[0]),df_t['STC'],marker='*',color=colors[2])
        plt.scatter(range(df_t.shape[0]),df_t['N'],marker='*',color=colors[3])
        plt.scatter(range(df_t.shape[0]),df_t['C/N'],marker='*',color=colors[4])

        plt.scatter(range(df2_t.shape[0]), df2_t['SOC'], marker='*',color=colors[0])
        plt.scatter(range(df2_t.shape[0]), df2_t['SIC'], marker='*',color=colors[1])
        plt.scatter(range(df2_t.shape[0]), df2_t['STC'], marker='*',color=colors[2])
        plt.scatter(range(df2_t.shape[0]), df2_t['N'], marker='*',color=colors[3])
        plt.scatter(range(df2_t.shape[0]), df2_t['C/N'], marker='*',color=colors[4])

        plt.xticks(range(df_t.shape[0]),df_t['year'],fontsize=12)
        plt.title(t,fontsize=12)
        plt.ylim(0,30)
        plt.grid(True)

        pindex+=1

    plt.show()


if __name__ == '__main__':
    # sarima_regression()
    plot_14file()