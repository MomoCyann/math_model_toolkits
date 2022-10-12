import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import numpy as np

# 画附件14不同放牧强度下化学性质的分布图
def plot_14file():
    df = pd.read_excel('数据集/整理数据/化学性质数据.xlsx')

    df.sort_values(by=['plot','year','intensity'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    print(df.head())

    grid=[(0,1),(0,2),(0,3),
          (1,1),(1,2),(1,3),
          (2,1),(2,2),(2,3),
          (3,1),(3,2),(3,3)]
    g_i=0

    plt.figure(figsize=(12,8))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.4)

    all_plot = ['G17','G19','G21','G6','G12','G18','G8','G11','G16','G9','G13','G20']
    pindex = 1
    for t in all_plot:

        df_t=df.loc[df['plot']==t,:]
        df_t.reset_index(drop=True,inplace=True)

        plt.subplot(4, 3, pindex)

        plt.plot(df_t['SOC'],label='SOC')
        plt.plot(df_t['SIC'],label='SIC')
        plt.plot(df_t['STC'],label='STC')
        plt.plot(df_t['N'],label='N')
        plt.plot(df_t['C/N'],label='C/N')

        plt.scatter(range(df_t.shape[0]),df_t['SOC'],marker='*')
        plt.scatter(range(df_t.shape[0]),df_t['SIC'],marker='*')
        plt.scatter(range(df_t.shape[0]),df_t['STC'],marker='*')
        plt.scatter(range(df_t.shape[0]),df_t['N'],marker='*')
        plt.scatter(range(df_t.shape[0]),df_t['C/N'],marker='*')

        plt.xticks(range(df_t.shape[0]),df_t['year'],fontsize=12)
        plt.title(t,fontsize=12)
        plt.ylim(0,30)
        plt.grid(True)
        plt.legend()

        pindex+=1

    plt.show()




plot_14file()