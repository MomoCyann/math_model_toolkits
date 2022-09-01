import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from minepy import MINE
"""
price_growth
索引介绍：
1为年份
2-52 为原始数据
52-102 为各变量对应的增长率
102-111 为9类别增长率
111 为 50个变量均值增长率
112 为 9 个类别 均值增长率
"""


# 插值
def interpolate():
    df = pd.read_csv('data/price.csv')
    for i in range(2, 52):
        print(df.columns[i])
        df.iloc[:, i] = df.iloc[:, i].apply(lambda x: str(x).replace(u'\xa0', u''))
        df.iloc[:, i] = df.iloc[:, i].apply(lambda x: float(x))
    df = df.apply(lambda x: x.interpolate())

    df = df.iloc[:, 1:]
    df.to_csv('data/price_1.csv')

# 计算每一个变量的环比增长率
def count_rowthRate():
    df = pd.read_csv('data/price_1.csv')
    for column in df.columns[2:]:
        print(column)
        for index in df.index[1:]:
            GR_column_name = column + '_GR'
            df.loc[index, GR_column_name] = df.loc[index, column] / df.loc[index - 1, column] - 1
    print(df.info())
    print(df.head(-5))
    df = df.iloc[:, 1:]
    df.to_csv('data/price_growthRate.csv')

# 计算类内平均
def splitByCategory():
    df = pd.read_csv('data/price_growthRate.csv')
    for index in df.index[1:]:
        # iloc 里的column 左闭右开
        # 第一类 黑色金属
        df.loc[index,'黑色金属GR']=df.iloc[index,52:58].mean()
        # 第二类 有色金属
        df.loc[index, '有色金属GR'] = df.iloc[index, 58:62].mean()
        # 第三类 化工产品
        df.loc[index, '化工产品GR'] = df.iloc[index, 62:72].mean()
        # 第四类 石油天然气
        df.loc[index, '石油天然气GR'] = df.iloc[index, 72:78].mean()
        # 第五类 煤炭
        df.loc[index, '煤炭GR'] = df.iloc[index, 78:84].mean()
        # 第六类 非金属建材
        df.loc[index, '非金属建材GR'] = df.iloc[index, 84:88].mean()
        # 第七类 农产品
        df.loc[index, '农产品GR'] = df.iloc[index, 88:96].mean()
        # 第八类 农业生产资料
        df.loc[index, '农业生产资料GR'] = df.iloc[index, 96:99].mean()
        # 第九类 林产品
        df.loc[index, '林产品GR'] = df.iloc[index, 99:102].mean()
        # 总指标
        df.loc[index,'总增长率50'] = df.iloc[index, 52:102].mean()
        df.loc[index,'总增长率9'] = df.iloc[index, 102:111].mean()
    df=df.iloc[:,1:]
    df.to_csv('data/price_growth.csv')

# 画单一增长率图
def plot_GrowthRate():
    df = pd.read_csv('data/price_growthRate.csv')
    print(df.info())
    for column in df.columns[52:]:
        plt.plot(df.index, df.loc[:, column])
        plt.title(column)
        plt.ylabel('增长率')
        plt.xlabel('年份')
        plt.ylim(-0.4, 0.4)
        # plt.xticks(df.index,df.iloc[:,1],rotation=90,size=4)
        plt.show()

# 9个类别增长率
# 总体增长率图
def plot_9GrowthRate():
    df = pd.read_csv('data/price_growth.csv')
    print(df.info())
    # plt.plot(df.index,df.loc[:,'总增长率9'],c='r',label='9')
    # plt.plot(df.index,df.loc[:,'总增长率50'],c='b',label='50')
    # plt.plot(df.index,df.loc[:,'黑色金属GR'],label='黑色金属GR')
    # plt.plot(df.index,df.loc[:,'有色金属GR'],label='有色金属GR')
    # plt.plot(df.index,df.loc[:,'化工产品GR'],label='化工产品GR')
    # plt.plot(df.index,df.loc[:,'石油天然气GR'],label='石油天然气GR')
    # plt.plot(df.index,df.loc[:,'煤炭GR'],label='煤炭GR')
    # plt.plot(df.index,df.loc[:,'非金属建材GR'],label='非金属建材GR')
    # plt.plot(df.index,df.loc[:,'农产品GR'],label='农产品GR')
    # plt.plot(df.index,df.loc[:,'农业生产资料GR'],label='农业生产资料GR')
    plt.plot(df.index,df.loc[:,'林产品GR'],label='林产品GR')
    plt.legend()
    plt.ylim(-0.5, 0.5)
    plt.show()

# 总体增长率图
def plot_GrowthRate():
    df = pd.read_csv('data/price_growth.csv')
    print(df.info())
    plt.plot(df.index,df.loc[:,'总增长率9'],c='r',label='9')
    plt.plot(df.index,df.loc[:,'总增长率50'],c='b',label='50')
    plt.legend()
    plt.ylim(-0.1, 0.1)
    plt.show()

# 3sigma 输出一个表格 那些产品的什么时间，增长率异常
def sigma3_rules():
    data = pd.read_csv('data/price_growth.csv')
    date = data.iloc[:, 1]
    data = data.iloc[:, 52:-11]
    data = pd.concat([date, data], axis=1)
    empty = data.drop(data.index[0:])
    data = data.iloc[:, 1:]

    mean = data.mean()
    std = data.std()
    drop_indices = []
    for index, row in data.iterrows():
        for i in range(row.size):
            if np.absolute((row[i] - mean[i])) > 3 * std[i]:
                empty.loc[index, row.index[i]] = date.iloc[index]
            else:
                empty.loc[index, row.index[i]] = ''
    empty.to_csv('data/3sigma.csv')


# pearson相关性热力图
def plot_50var_corr_heatmap():
    df = pd.read_csv('data/price_growth.csv')
    df=df.iloc[:,102:111]

    # 归一化
    df=df.apply(lambda x:preprocessing.scale(x))

    figure,ax=plt.subplots(figsize=(12,12))
    sns.heatmap(df.corr(),square=True,annot=False,ax=ax)
    plt.show()

# 灰色关联分析
def grey_relation_analysis():
    DataFrame = pd.read_csv('data/price_fixed_GR.csv')
    DataFrame = DataFrame.iloc[1:,52:102]
    DataFrame.reset_index(drop=True, inplace=True)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 无量纲化
    def dimensionlessProcessing(df):
        newDataFrame = pd.DataFrame(index=df.index)
        columns = df.columns.tolist()
        for c in columns:
            d = df[c]
            MAX = d.max()
            MIN = d.min()
            MEAN = d.mean()
            newDataFrame[c] = ((d - MEAN) / (MAX - MIN)).tolist()
        return newDataFrame

    def GRA_ONE(gray, m=0):
        # 读取为df格式
        gray = dimensionlessProcessing(gray)
        # 标准化
        std = gray.iloc[:, m]  # 为标准要素
        gray.drop(str(m), axis=1, inplace=True)
        ce = gray.iloc[:, 0:]  # 为比较要素
        shape_n, shape_m = ce.shape[0], ce.shape[1]  # 计算行列

        # 与标准要素比较，相减
        a = zeros([shape_m, shape_n])
        for i in range(shape_m):
            for j in range(shape_n):
                a[i, j] = abs(ce.iloc[j, i] - std[j])

        # 取出矩阵中最大值与最小值
        c, d = amax(a), amin(a)

        # 计算值
        result = zeros([shape_m, shape_n])
        for i in range(shape_m):
            for j in range(shape_n):
                result[i, j] = (d + 0.5 * c) / (a[i, j] + 0.5 * c)

        # 求均值，得到灰色关联值,并返回
        result_list = [mean(result[i, :]) for i in range(shape_m)]
        result_list.insert(m, 1)
        return pd.DataFrame(result_list)

    df = DataFrame.copy()
    list_columns = [
        str(s) for s in range(len(df.columns)) if s not in [None]
    ]
    df_local = pd.DataFrame(columns=list_columns)
    df.columns = list_columns
    for i in range(len(df.columns)):
        df_local.iloc[:, i] = GRA_ONE(df, m=i)[0]

    def ShowGRAHeatMap(DataFrame):
        colormap = plt.cm.RdBu
        ylabels = DataFrame.columns.values.tolist()
        f, ax = plt.subplots(figsize=(14, 14))
        ax.set_title('GRA HeatMap')

        # 设置展示一半，如果不需要注释掉mask即可
        mask = np.zeros_like(DataFrame)
        mask[np.triu_indices_from(mask)] = True

        with sns.axes_style("white"):
            sns.heatmap(DataFrame,
                        cmap="YlGnBu",
                        annot=False,
                        mask=mask,
                        )
        plt.yticks(rotation=0)
        plt.show()
    df_local.columns = DataFrame.columns
    df_local['column'] = DataFrame.columns
    df_local.set_index('column', inplace=True)
    ShowGRAHeatMap(df_local)

def mic(df):
    df = pd.read_csv('data/price_fixed_GR.csv')
    df = df.iloc[1:, 52:102]
    df.reset_index(drop=True, inplace=True)

    def MIC_matirx(dataframe, mine):

        data = np.array(dataframe)
        n = len(data[0, :])
        result = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                mine.compute_score(data[:, i], data[:, j])
                result[i, j] = mine.mic()
                result[j, i] = mine.mic()
        RT = pd.DataFrame(result)
        return RT

    mine = MINE(alpha=0.6, c=15)
    data_wine_mic = MIC_matirx(df, mine)
    data_wine_mic.columns = df.columns
    data_wine_mic['column'] = df.columns
    data_wine_mic.set_index('column', inplace=True)

    def ShowHeatMap(DataFrame):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title('MIC', y=1.05, size=15)
        sns.heatmap(DataFrame.astype(float), square=True, cmap="YlGnBu",annot=False)
        plt.yticks(rotation=0)
        plt.show()
    ShowHeatMap(data_wine_mic)

def edu_med():
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
                df.iloc[row, col] = int(df.iloc[row, col]) - int(df.iloc[row-1, col])
                count += 1
                row -= 1
            else:
                count = 1
                row -= 1
        df.iloc[-1, col] = int(df.iloc[-1, col]) - int(df.iloc[-2, col])
    for col in [1, 3, 5, 7, 9, 11]:
        row = 0
        while row <= 33:
            if row == 0:
                df.iloc[row, col] = ''
            df.iloc[row, col] = (int(df.iloc[row, col-1])-int(df.iloc[0, col-1])) / int(df.iloc[0, col-1])
            row +=1
    result = df.iloc[:, [1, 3, 5, 7, 9, 11]]

    for col in [1, 3, 5, 7, 9, 11]:
        df.iloc[:, col] = df.iloc[:, col] - df.iloc[:, col].shift(1)
    df.iloc[:1,:] = 0
    result2 = df.iloc[:, [1, 3, 5, 7, 9, 11]]
    final_result = pd.concat([result, result2], axis=1)
    final_result.to_csv('data/edu_gr_dif.csv')

# grey_relation_analysis()
# mic()
edu_med()






