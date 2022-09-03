import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

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
"""

# 画单一增长率图
def plot_GrowthRate(df):
    print(df.info())
    df=df.loc[2:,:]
    df.reset_index(drop=True,inplace=True)

    for column in df.columns[159:]:
        plt.plot(df.index, df.loc[:, column])
        plt.title(column)
        plt.ylabel('增长率')
        plt.xlabel('年份')
        plt.ylim(-0.2, 0.2)
        plt.plot(df.index,df.index*0+0.05,c='r')
        plt.plot(df.index,df.index*0+-0.05,c='r')
        plt.plot(df.index, df.index * 0 + 0.01, c='r')
        plt.plot(df.index, df.index * 0 + -0.01, c='r')
        # plt.xticks(df.index,df.iloc[:,1],rotation=90,size=4)

        x_major_locator = plt.MultipleLocator(12)
        # 把x轴的刻度间隔设置为1，并存在变量里
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xticks(range(0, 280, 12), df['yyyymmxx'][df.index % 12 == 0], rotation=45)

        plt.show()

# 9个类别增长率\总体增长率图
def plot_9GrowthRate(df):
    print(df.info())
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    plt.plot(df.index,df.loc[:,'总增长率9'],c='r',label='9',linewidth=3)
    plt.plot(df.index,df.loc[:,'总增长率9_dif'],c='b',label='9_dif',alpha=0.5)
    plt.plot(df.index,df.loc[:,'黑色金属GR'],label='黑色金属GR')
    plt.plot(df.index,df.loc[:,'有色金属GR'],label='有色金属GR')
    plt.plot(df.index,df.loc[:,'化工产品GR'],label='化工产品GR')
    plt.plot(df.index,df.loc[:,'石油天然气GR'],label='石油天然气GR')
    plt.plot(df.index,df.loc[:,'煤炭GR'],label='煤炭GR')
    plt.plot(df.index,df.loc[:,'非金属建材GR'],label='非金属建材GR')
    plt.plot(df.index,df.loc[:,'农产品GR'],label='农产品GR')
    plt.plot(df.index,df.loc[:,'农业生产资料GR'],label='农业生产资料GR')
    plt.plot(df.index,df.loc[:,'林产品GR'],label='林产品GR')

    x_major_locator = plt.MultipleLocator(12)
    # 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.xticks(range(0,280,12),df['yyyymmxx'][df.index%12==0],rotation=45)

    plt.legend()
    # plt.ylim(-0.5, 0.5)
    plt.show()

# pearson相关性热力图
def plot_50var_corr_heatmap(df):
    df=df.iloc[:,100:149]

    # 归一化
    df=df.apply(lambda x:preprocessing.scale(x))

    figure,ax=plt.subplots(figsize=(12,12))
    sns.heatmap(df.corr(),square=True,annot=False,ax=ax)
    plt.show()
