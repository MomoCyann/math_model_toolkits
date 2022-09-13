import missingno as msno
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import io
from PIL import Image
from minepy import MINE
from sklearn import preprocessing

def del_same_feature(data):
    '''
    删除方差为0的列（全部相同）
    :param data: 读取的数据集，只含特征，dataframe
    :returns: 清洗后的dataframe
    '''
    # 全类型通用，包括字符串
    data = data.loc[:, (data != data.iloc[0]).any()]
    # 仅适用于数值类型
    #data.drop(data.columns[data.std() == 0], axis=1, inplace=True)
    return data

def del_perc_same_feature(data, threshold):
    '''
    删除相同比列高于阈值的列
    :param data: 读取的数据集，只含特征，dataframe
    :param threshold: 阈值，某个值占比超过阈值的列会被删除
    :return: 清洗后的dataframe，并打印所删除的特征名
    '''
    # 待删除的列存放
    column_indexs = []
    # 按列遍历dataframe
    for column_index, row_data in data.iteritems():
        counts = row_data.value_counts(normalize=True)
        # 若占比最大的某个值超过阈值，则记入待删除列
        if counts.iloc[0] >= threshold:
            column_indexs.append(column_index)
    print("删除的特征为：" + str(column_indexs))
    data = data.drop(labels=column_indexs, axis=1)
    return data

def del_std_small_feature(data, threshold):
    '''
    删除方差小于阈值的列
    :param data: 读取的数据集，只含特征，dataframe
    :param threshold: 阈值，方差小于阈值的列会被删除
    :return: 清洗后的dataframe
    '''
    # 待删除的列存放
    column_indexs = []
    # 按列遍历dataframe
    for column_index, row_data in data.iteritems():
        counts = row_data.std()
        # 若占比最大的某个值超过阈值，则记入待删除列
        if counts <= threshold:
            column_indexs.append(column_index)
    print("删除的特征为：" + str(column_indexs))
    data = data.drop(labels=column_indexs, axis=1)
    return data

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

def sigma3_rules(data):
    '''
    根据3σ法则，删除某特征为异常值的样本，即删掉某些行，因为他的某个特征是异常值。
    :param data: 特征集
    :return: 清洗后的数据集，并打印所删除样本索引
    '''
    mean = data.mean()
    std = data.std()
    drop_indices = []
    for index, row in data.iterrows():
        tmp = (row - mean).abs() > 3 * std
        if tmp.any():
            drop_indices.append(index)
    data.drop(drop_indices, inplace=True)
    no = [i + 1 for i in drop_indices]
    print(no)
    return data

def del_perc_null_feature(data, threshold):
    '''
    删除缺失值比例大于阈值的特征
    :param data: 特征集
    :param threshold: 缺失值比例的阈值
    :return: 清洗后的特征集，并打印被
    '''
    # 根据需要可以把表格的0转换为空值
    data[data == 0] = np.nan

    nan_perc = data.isnull().sum() / len(data)
    # 筛选缺失值占比大于阈值的列
    nan_columns = list(nan_perc[nan_perc>threshold].index)
    print("删除的特征为：" + str(nan_columns))
    data = data.drop(labels=nan_columns, axis=1)
    return data

def fill_null(data):
    '''
    填充缺失值，支持将0转换成空值处理，方法包含：前后填充、均值填充、线性插值
    :param data:
    :return:
    '''
    # 根据需要可以把表格的0转换为空值
    data[data == 0] = np.nan
    # 打印有缺失值的列和缺失值数目
    nan_count = data.isnull().sum()
    print(nan_count[nan_count>0])

    # 选择一个填充方法 ↓

    # # 取后一个有效值填充
    # data = data.fillna(method='bfill')
    # # 取前一个有效值填充
    # data = data.fillna(method='ffill')
    # 填充列的平均值
    data = data.fillna(data.mean())
    # # 线性插值
    # data = data.interpolate()
    return data

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

def mic_features(df):
    '''
    这个是返回变量间的关系。
    :param df:
    :return:
    '''
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

def mic_target(df, target):
    '''
    这个是返回变量与被预测量。
    :param df:
    :return:
    '''
    def MIC_matirx(dataframe, mine):

        data = np.array(dataframe)
        # 特征数量
        n = len(data[0, :])
        result = np.zeros([1, n])

        for i in range(n):
            mine.compute_score(data[:, i], target)
            result[0, i] = mine.mic()
        RT = pd.DataFrame(result)
        return RT

    mine = MINE(alpha=0.6, c=15)
    data_mic = MIC_matirx(df, mine)
    data_mic.columns = df.columns
    data_mic.to_csv('mic_target.csv')

    def ShowHeatMap(DataFrame):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title('MIC', y=1.05, size=15)
        sns.heatmap(DataFrame.astype(float), square=True, cmap="YlGnBu",annot=False)
        plt.yticks(rotation=0)
        plt.show()
    # ShowHeatMap(data_mic)

if __name__ == '__main__':
    # setting
    feature_file = './dataset/Molecular_Descriptor.xlsx'
    target_file = './dataset/ER_activity.xlsx'
    data = pd.read_excel(feature_file)
    #删掉第一列，分子结构，只保留特征
    data = data.iloc[:,1:]
    print(data.info)
    # [1974 rows x 729 columns]>

    # testing
    data = del_same_feature(data)
    print(data.info)
    # [1974 rows x 504 columns]>

    data = del_perc_same_feature(data, 0.9)
    print(data.info)
    # [1974 rows x 362 columns] >

    data = del_std_small_feature(data, 0.05)
    # [1974 rows x 341 columns] >

    target = pd.read_excel(target_file)
    target = target.iloc[:,2]

    mic_target(data, target)
    print('complete')




    # data = pd.read_csv('./dataset/test_data.csv')
    # draw_feature(data)

    # palette('fade')

    # data = pd.read_excel("./dataset/附件一：325个样本数据.xlsx", header=2)
    # # 剔除前面的序号和时间 取非操作变量的前面一些行
    # data = data.iloc[:, 2:]
    # data = del_perc_null_feature(data, 0.2)
    # data = fill_null(data)
    # print(data.info)