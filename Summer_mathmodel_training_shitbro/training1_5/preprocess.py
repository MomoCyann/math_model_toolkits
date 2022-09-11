import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from minepy import MINE
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

"""
index_type:
fixed - 定基 以20140101 为基期
m2m   - 环比 以上一期来基准
"""

def count_index(index_type):
    # 导入数据
    df=pd.read_excel('data/流通领域重要生产资料价格.xlsx',header=0)
    # 删除开头部分连续多行的空行，以及空列
    df.dropna(axis=1,how='all',inplace=True)
    df.drop(range(1,25),axis=0,inplace=True)
    df.drop(columns='人造板（1220*2440*15mm）',inplace=True)
    df.reset_index(drop=True,inplace=True)
    print(df.info())

    # 数据清洗及插值
    for i in range(50):
        print(df.columns[i])
        df.iloc[:, i] = df.iloc[:, i].apply(lambda x: str(x).replace(u'\xa0', u''))
        df.iloc[:, i] = df.iloc[:, i].apply(lambda x: float(x))
    df = df.apply(lambda x: x.interpolate())

    # 计算每个变量的定基指数
    # 　以2014年1月1日为基期，计算定基指数
    var_columns=df.columns[1:]
    if index_type=='fixed':
        for column in var_columns:
            print(column)
            for index in df.index:
                GR_column_name = column + '_fixed_GR'
                df.loc[index, GR_column_name] = df.loc[index, column] / df.loc[0, column]

        for column in var_columns:
            dif_column_name=column+'_dif_GR'
            GR_column_name = column + '_fixed_GR'
            df[dif_column_name]=df[GR_column_name]-df[GR_column_name].shift(1)
            df.loc[[0,1],dif_column_name]=0
    print(df.info())

    # 环比
    if index_type=='m2m':
        for column in var_columns:
            print(column)
            for index in df.index[2:]:
                GR_column_name = column + '_m2m_GR'
                df.loc[index, GR_column_name] = df.loc[index, column] / df.loc[index - 1, column] - 1
        df.fillna(0.0,inplace=True)

    # 计算类别定基指数
    for index in df.index:
        # iloc 里的column 左闭右开
        # 第一类 黑色金属
        df.loc[index,'黑色金属GR']=df.iloc[index,50:56].mean()
        # 第二类 有色金属
        df.loc[index, '有色金属GR'] = df.iloc[index, 56:60].mean()
        # 第三类 化工产品
        df.loc[index, '化工产品GR'] = df.iloc[index, 60:70].mean()
        # 第四类 石油天然气
        df.loc[index, '石油天然气GR'] = df.iloc[index, 70:76].mean()
        # 第五类 煤炭
        df.loc[index, '煤炭GR'] = df.iloc[index, 76:82].mean()
        # 第六类 非金属建材
        df.loc[index, '非金属建材GR'] = df.iloc[index, 82:86].mean()
        # 第七类 农产品
        df.loc[index, '农产品GR'] = df.iloc[index, 86:94].mean()
        # 第八类 农业生产资料
        df.loc[index, '农业生产资料GR'] = df.iloc[index, 94:97].mean()
        # 第九类 林产品
        df.loc[index, '林产品GR'] = df.iloc[index, 97:99].mean()
        # 总指标
        # df.loc[index,'总增长率50'] = df.iloc[index, 50:100].mean()
        df.loc[index,'总增长率9'] = df.iloc[index, 99:108].mean()
    df['总增长率9_dif'] = df['总增长率9']-df['总增长率9'].shift(1)
    df.loc[0,'总增长率9_dif'] = 0

    print(df.columns)
    df.to_csv('data/price_{}_GR.csv'.format(index_type))

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
    list = np.arange(51, 100).tolist()
    print(list)
    list1 = np.arange(149, 158)
    for i in list1:
        list.append(i)
    list.append(158)
    DataFrame = DataFrame.iloc[1:, list]
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
    df_local.to_csv('data/grey59.csv')
    # ShowGRAHeatMap(df_local)

def mic():
    df = pd.read_csv('data/price_fixed_GR.csv')
    list = np.arange(51, 100).tolist()
    print(list)
    list1 = np.arange(149, 158)
    for i in list1:
        list.append(i)
    list.append(158)
    df = df.iloc[1:, list]
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
    data_wine_mic.to_csv('data/mic59.csv')

    def ShowHeatMap(DataFrame):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title('MIC', y=1.05, size=15)
        sns.heatmap(DataFrame.astype(float), square=True, cmap="YlGnBu",annot=False)
        plt.yticks(rotation=0)
        plt.show()
    #ShowHeatMap(data_wine_mic)



#grey_relation_analysis()
mic()






