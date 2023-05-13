import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import statsmodels.api as sm
from fastdtw import fastdtw
from statsmodels.tsa.seasonal import STL
"""
price_growth_fiexd
索引介绍：
1为年份
2-51 为原始数据
51-100 为各变量对应的增长率
100-149 为各变量增长率较上期差额
149-158 为9类别增长率
158 为 总增长率-9 个类别 均值增长率
159 为 总增长率-较上期差额

price_growth_m2m
索引介绍：
1为年份
2-51 为原始数据
51-100 为各变量对应的增长率
100-109 为9类别增长率
109 为 总增长率-9 个类别 均值增长率
110 为 总增长率-较上期差额

edu_gr_alltype
索引介绍
0是季度
1-6 为定基增长率
7-12 为定基指数的差距
13-18 环比
"""


# 养老教育数据整理
def edu_med_data_preprocess():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    df = pd.read_csv('data/edu.csv')
    df = df.iloc[1:-1, :]
    df = df.T
    columns = df.iloc[0, 1:].tolist()
    df.set_index(1, inplace=True)
    df.columns = columns
    df = df.iloc[1:, :]
    df['newindex'] = np.arange(len(df) - 1, -1, -1)
    df.sort_values('newindex', inplace=True)
    df.drop('newindex', axis=1, inplace=True)

    # 居然是累计值
    for col in [0,2,4,6,8,10]:
        row = 31
        count = 1
        while row > 0:
            if count <= 3:
                df.iloc[row, col] = int(df.iloc[row, col]) - int(df.iloc[row-1, col])+1
                count += 1
                row -= 1
            else:
                count = 1
                row -= 1
        df.iloc[-1, col] = int(df.iloc[-1, col]) - int(df.iloc[-2, col])
    result_value = df.iloc[:, [0,2,4,6,8,10]]
    result_value.to_csv('data/edu_value.csv')
    # 定基
    for col in [1, 3, 5, 7, 9, 11]:
        row = 0
        while row <= 33:
            if row == 0:
                df.iloc[row, col] = ''
            df.iloc[row, col] = (int(df.iloc[row, col-1])-int(df.iloc[0, col-1])) / int(df.iloc[0, col-1]) + 1
            row +=1
    result_gr = df.iloc[:, [1, 3, 5, 7, 9, 11]]

    # 定基绝对差值
    for col in [1, 3, 5, 7, 9, 11]:
        df.iloc[:, col] = df.iloc[:, col] - df.iloc[:, col].shift(1)
    result_dif = df.iloc[:, [1, 3, 5, 7, 9, 11]]

    # 环比
    for col in [1, 3, 5, 7, 9, 11]:
        row = 0
        while row <= 33:
            if row == 0:
                df.iloc[row, col] = ''
            df.iloc[row, col] = int(df.iloc[row, col - 1]) / int(df.iloc[row-1, col - 1]) - 1
            row += 1
    df.iloc[0, :] = ''
    result_m2m = df.iloc[:, [1, 3, 5, 7, 9, 11]]

    final_result = pd.concat([result_gr, result_dif], axis=1)
    final_result = pd.concat([final_result, result_m2m], axis=1)
    final_result.to_csv('data/edu_gr_alltype.csv')

# 养老教育
def edu_plot_value(df):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(12, 8))
    plt.plot(df.iloc[:, 0], df.iloc[:, 3], label='教育文娱支出(元)', color='deepskyblue')
    plt.plot(df.iloc[:, 0], df.iloc[:, 4], label='医疗保健支出(元)', color='salmon')
    plt.title('2014年第一季度-2022年第二季度教育和养老支出数据',size=18)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('支出(元)', fontsize=16)
    plt.legend(prop={'size':16})
    plt.xticks(rotation=45)
    plt.show()
    print('s')

def edu_plot(df):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(12, 8))
    plt.plot(df.iloc[:, 0], df.iloc[:, 15], label='教育文娱支出环比', color='deepskyblue')
    plt.plot(df.iloc[:, 0], df.iloc[:, 16], label='医疗保健支出环比', color='salmon')
    plt.title('2014年第一季度-2022年第二季度教育和养老支出环比',size=18)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('环比', fontsize=16)
    plt.legend(prop={'size':16})
    plt.xticks(rotation=45)
    plt.show()
    print('s')

def edu_del_season(df):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    decomposition_edu = sm.tsa.seasonal_decompose(df.iloc[1:, 15], model='addictive', period=4)
    decomposition_med = sm.tsa.seasonal_decompose(df.iloc[1:, 16], model='addictive', period=4)
    # decomposition_edu.plot()
    # decomposition_med.plot()
    trend_edu = decomposition_edu.trend
    trend_med = decomposition_med.trend
    plt.figure(figsize=(12, 8))
    plt.plot(df.iloc[1:, 0], trend_edu, label='教育文娱支出环比', color='deepskyblue')
    plt.plot(df.iloc[1:, 0], trend_med, label='医疗保健支出环比', color='salmon')
    plt.title('2014年第一季度-2022年第二季度教育和养老支出去季节性后环比', size=18)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('环比', fontsize=16)
    plt.legend(prop={'size': 16})
    plt.xticks(rotation=45)
    plt.show()

def dtw_method(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    raw_data_edu = df.iloc[:, 3]
    raw_data_med = df.iloc[:, 4]
    decomposition_edu = sm.tsa.seasonal_decompose(df.iloc[:, 3], model='addictive', period=4)
    decomposition_med = sm.tsa.seasonal_decompose(df.iloc[:, 4], model='addictive', period=4)
    trend_edu = decomposition_edu.trend
    trend_med = decomposition_med.trend
    trend_edu_med = (trend_med + trend_edu)/2
    price = pd.read_csv('data/price_fixed_GR.csv')
    # price = price.iloc[2:, :]
    price.reset_index(drop=True, inplace=True)
    # plt.plot(df.index, df.loc[:, '总增长率9'], c='r', label='9', linewidth=3)
    # plt.plot(df.index, df.loc[:, '总增长率9_dif'], c='b', label='9_dif', alpha=0.5)
    # plt.plot(df.index, df.loc[:, '黑色金属GR'], label='黑色金属GR')
    # plt.plot(df.index, df.loc[:, '有色金属GR'], label='有色金属GR')
    # plt.plot(df.index, df.loc[:, '化工产品GR'], label='化工产品GR')
    # plt.plot(df.index, df.loc[:, '石油天然气GR'], label='石油天然气GR')
    # plt.plot(df.index, df.loc[:, '煤炭GR'], label='煤炭GR')
    # plt.plot(df.index, df.loc[:, '非金属建材GR'], label='非金属建材GR')
    # plt.plot(df.index, df.loc[:, '农产品GR'], label='农产品GR')
    # plt.plot(df.index, df.loc[:, '农业生产资料GR'], label='农业生产资料GR')
    # plt.plot(df.index, df.loc[:, '林产品GR'], label='林产品GR')
    avg_price = price.iloc[:, 158] # 9类平均增长率
    another_price = price.iloc[:, 149]

    def dtw(df):
        x = np.array(avg_price)
        y = np.array(df.dropna())
        manhattan_distance = lambda x, y: np.abs(x-y)
        d, path = fastdtw(x, y, dist=manhattan_distance)
        print('DTW = {}'.format(d))

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(avg_price, label='9类商品平均增长率', color='deepskyblue')
        plt.xlabel('时间', fontsize=14)
        plt.ylabel('增长率', fontsize=16)
        plt.legend(prop={'size': 16})
        plt.subplot(1, 2, 2)
        plt.plot(df, label='医疗保健增长率', color='salmon')
        # plt.title('九类商品与黑色金属增长率', size=18)
        plt.xlabel('时间', fontsize=14)
        plt.ylabel('增长率', fontsize=16)
        plt.legend(prop={'size': 16})
        # plt.xticks(rotation=45)
        plt.show()
    dtw(another_price)
    dtw(trend_med)
    dtw(raw_data_med)


df = pd.read_csv('data/edu_gr_alltype.csv')
df_value = pd.read_csv('data/edu_value.csv')
# edu_med_data_preprocess()
# edu_plot_value(df_value)
# edu_plot(df)
# edu_del_season(df)
dtw_method(df)