import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import io
from PIL import Image
from minepy import MINE
from sklearn import preprocessing

def save_png_to_tiff(name):
    '''
    保存图表为PNG和TIFF两种格式
    :param name: 文件名
    :return: tiff-dpi：200 → 2594x1854
    '''
    plt.savefig('./fig_preview/' + name + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=200, pad_inches=.1, bbox_inches='tight')
    # Load this image into PIL
    png2 = Image.open(png1)
    # Save as TIFF
    png2.save('./fig_preview/' + name + ".tiff")
    png1.close()

def draw_feature(data):
    '''
    选择整型、浮点型变量各16个画出分布图
    :param data: 特征
    :return: 特征分布图，png 1600x1000, tiff-dpi：200 → 2594x1854
    '''
    # 数据准备
    # 遍历每列的类型，分成整型和浮点型
    columns_int = []
    columns_float = []
    for column_index in data.columns:
        if str(type(data[column_index][0])) == "<class 'numpy.int64'>":
            columns_int.append(column_index)
        if str(type(data[column_index][0])) == "<class 'numpy.float64'>":
            columns_float.append(column_index)
    # 随机选择16个特征做分布图，整形做条形图，浮点型会多一个拟合曲线
    columns_int_samples = random.sample(columns_int, 16)
    columns_float_samples = random.sample(columns_float, 16)
    print("选取的整型变量为：" + str(columns_int_samples))
    print("选取的浮点变量为：" + str(columns_float_samples))
    # # 根据实际情况也可手动指定
    # columns_int_samples = []
    # columns_float_samples = []
    # 清理空值
    data = data.dropna()

    # 开始画图
    # 预设
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 整体画布大小 1600x1000 tiff-dpi：200 → 2594x1854
    plt.figure(figsize=(16, 10))
    # 绘制16个整型变量子图
    for i in range(1,17):
        plt.subplot(4,4,i)
        sns.distplot(data[columns_int_samples[i-1]], bins=15, kde=False, hist=True, hist_kws={'histtype':'stepfilled'},
                     color='deepskyblue')
    plt.subplots_adjust(hspace=0.35)
    # 可选添加标题
    title = ''
    plt.suptitle(title, fontsize=20)
    # 文件名
    name = '整型变量直方图'
    save_png_to_tiff(name)
    plt.show()

    # 绘制16个浮点型变量子图
    plt.figure(figsize=(16, 10))
    for i in range(1,17):
        plt.subplot(4,4,i)
        sns.distplot(data[columns_float_samples[i-1]], hist=True, norm_hist=False, color='deepskyblue')
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    # 可选添加标题
    title = ''
    plt.suptitle(title, fontsize=20)
    # 文件名
    name = '浮点变量直方图'
    save_png_to_tiff(name)
    plt.show()

def palette(arg:str):
    '''
    调色板
    fade: 12色 红黄绿蓝紫渐变
    simple: 蓝到红渐变 6色
    rainbow: 彩虹 12色
    :return:
    '''
    # 标准彩色调色板，12色 红黄绿蓝紫渐变
    fade = sns.hls_palette(12, l=0.7, s=0.9)
    # sns.palplot(fade)

    # 简约 蓝到红渐变 6色
    simple = sns.diverging_palette(240, 10, sep=12)
    # sns.palplot(simple)
    # # cmap = sns.diverging_palette(200, 20, as_cmap=True)

    # 彩虹 12色
    rainbow = sns.color_palette('rainbow', 12)

    # 自定义
    colors = ["deepskyblue", "salmon"]
    # colors = ['#00bfff', '#fa8072']
    custom = sns.color_palette(colors)

    dict = {
        "fade": fade,
        "simple": simple,
        "rainbow": rainbow
    }
    choice = dict.get(arg)
    # 设置调色板
    sns.set_palette(choice)
    sns.palplot(choice)
    plt.show()

def grey_relation_analysis(DataFrame):
    '''
    输出特征的灰色关联系数热力图
    :return:
    '''
    # DataFrame = pd.read_csv('data/price_fixed_GR.csv')
    ## 选择列数的预处理
    # list = np.arange(51, 100).tolist()
    # print(list)
    # list1 = np.arange(149, 158)
    # for i in list1:
    #     list.append(i)
    # list.append(158)
    # DataFrame = DataFrame.iloc[1:, list]
    # DataFrame.reset_index(drop=True, inplace=True)

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
    df_local.to_csv('data/grey.csv')
    ShowGRAHeatMap(df_local)

def mic(df):
    # df = pd.read_csv('data/price_fixed_GR.csv')
    ## 选择列数的预处理
    # list = np.arange(51, 100).tolist()
    # print(list)
    # list1 = np.arange(149, 158)
    # for i in list1:
    #     list.append(i)
    # list.append(158)
    # df = df.iloc[1:, list]
    # df.reset_index(drop=True, inplace=True)

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
    data_mic = MIC_matirx(df, mine)
    data_mic.columns = df.columns
    data_mic['column'] = df.columns
    data_mic.set_index('column', inplace=True)
    data_mic.to_csv('data/mic.csv')

    def ShowHeatMap(DataFrame):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title('MIC', y=1.05, size=15)
        sns.heatmap(DataFrame.astype(float), square=True, cmap="YlGnBu",annot=False)
        plt.yticks(rotation=0)
        plt.show()
    ShowHeatMap(data_mic)

if __name__ == '__main__':
    print('com')