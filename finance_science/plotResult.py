import glob
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
from matplotlib import pyplot as plt
import datetime
import time
import shutil
from GetTradeFeature import FeatureAndLabel
from sklearn.utils import shuffle
import jieba as jb
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

# 将几种策略的结果合并在一个df
def concat_result():
    df1=pd.read_csv('data/result/mix_results_30_0.csv')
    df2=pd.read_csv('data/result/ma_results_30_0.csv')
    df1['均线策略']=df2['收益率']
    df1=df1.iloc[:,1:]
    df1.columns=['公司名称','舆情收益','基准收益','均线收益']

    for index in df1.index:
        if df1.loc[index,'舆情收益']==1 or df1.loc[index,'舆情收益']==-1:
            df1.drop(index=index,axis=0,inplace=True)


    df1.to_csv('data/result/result_30.csv')
    print(df1.info())

# 画3种收益率曲线
def plot_profit():
    df = pd.read_csv('data/result/result_30.csv')
    df['舆情收益'] = df['舆情收益'].apply(lambda x: x-1)
    df['基准收益']=df['基准收益'].apply(lambda x: x-1)
    df['均线收益']=df['均线收益'].apply(lambda x: x-1)
    print(df.info())
    df=df[df['基准收益']>0]
    print(df.info())
    print('舆情收益'+str(df['舆情收益'].mean()))
    print('基准收益'+str(df['基准收益'].mean()))
    print('均线收益'+str(df['均线收益'].mean()))


    # x=range(df.shape[0])
    x=df['公司名称']
    print(x)

    plt.plot(x,df['舆情收益'],c='r',label='舆情投资策略')
    plt.plot(x,df['基准收益'],c='b',label='自然增长率',alpha=0.4)
    plt.plot(x,df['均线收益'],c='g',label='移动双均线',alpha=0.4)
    plt.xticks(rotation=45) # 旋转90度
    plt.scatter(x,df['舆情收益'],c='r',marker='*')
    plt.scatter(x,df['基准收益'],c='b',marker='o',alpha=0.4)
    plt.scatter(x,df['均线收益'],c='g',marker='^',alpha=0.4)
    plt.legend()
    plt.show()

    count=0
    for index in df.index:
        if df.loc[index,'舆情收益']>df.loc[index,'基准收益']:
            count+=1

    print(count)

# 画单篇新闻情感分类10个模型的准确率
def plot_acc():
    x_data=['MLP','TextCNN','RNN','LSTM','BiLSTM']
    y_data=[0.70399,0.696,0.69,0.698,0.694]
    y2_data=[0.7,0.7285,0.71,0.713,0.7415]
    x_width = range(0, len(x_data))
    x2_width = [i + 0.3 for i in x_width]
    #
    plt.bar(x_width, y_data, lw=0.5, fc="royalblue", width=0.3, label="词向量")
    plt.bar(x2_width, y2_data, lw=0.5, fc="lightcoral", width=0.3, label="句向量")

    plt.xticks(range(0, 5), x_data)
    plt.ylabel('准确率')
    plt.xlabel('模型')
    # plt.yticks(np.arange(0.6,0.8,0.05),y_data)
    plt.ylim(ymax=0.8,ymin=0.5)
    plt.legend()
    plt.show()

# 新闻热度排行
def plot_news_rank():
    # df_info=pd.DataFrame({'name':'','news_counts':''},index=[0])
    # index=0
    # for f in glob.glob('data/HS300_50/newsData_clear/*.csv'):
    #     df=pd.read_csv(f)
    #     df_info.loc[index]=[df.loc[0,'secShortName'],df.shape[0]]
    #     index+=1
    #
    # print(df_info.head())
    #
    # df_info.sort_values(by='news_counts',inplace=True)
    # print(df_info['news_counts'])
    # df_info.reset_index(drop=True,inplace=True)
    #
    # df_info.to_csv('data/HS300_50/news_counts.csv')
    df_info=pd.read_csv('data/HS300_50/news_counts.csv')

    plt.bar(range(df_info.shape[0]),np.array(df_info['news_counts']))
    plt.xticks(range(df_info.shape[0]),df_info['name'])
    plt.xticks(rotation=45)
    plt.show()

# 画50个股票的云图
def draw_cloud(read_name):
    image = Image.open('data/HS300_50/o.png')  # 作为背景轮廓图
    graph = np.array(image)
    # 参数分别是指定字体、背景颜色、最大的词的大小、使用给定图作为背景形状
    wc = WordCloud(font_path='msyh.ttc',background_color='white',max_words=100, mask=graph)
    df = pd.read_csv(read_name)  # 读取词频文件, 因为要显示中文，故编码为gbk
    name = df['name']  # 词
    value = df['news_counts']  # 词的频率
    for i in range(len(name)):
        name[i] = str(name[i])
    dic = dict(zip(name, value))  # 词频以字典形式存储
    wc.generate_from_frequencies(dic)  # 根据给定词频生成词云
    image_color = ImageColorGenerator(graph)#生成词云的颜色
    wc.to_file('词云.png')  # 图片命名

# 画不同权重下，某家公司的舆情因子表现
def plot_dif_weight():
    df1 = pd.read_csv('finance_science/data/HS300_50/feature/30_30/比亚迪_feature_30.csv')
    df2 = pd.read_csv('finance_science/data/HS300_50/feature/30_30_weight/比亚迪_feature_30.csv')
    df3 = pd.read_csv('finance_science/data/HS300_50/feature/30_30_weight_date/比亚迪_feature_30.csv')

    plt.plot(df1['sentimentFactor'], c='r', alpha=0.5, label='等权')
    plt.plot(df2['sentimentFactor'], c='b', alpha=0.5, label='按数量划分区间')
    plt.plot(df3['sentimentFactor'], c='g', alpha=0.5, label='按日期划分区间')
    plt.legend()
    plt.show()

# draw_cloud('data/HS300_50/news_counts.csv')
plot_profit()

