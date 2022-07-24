import missingno as msno
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import io
from PIL import Image

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
    :return: 清洗后的dataframe
    '''
    # 待删除的列存放
    column_indexs = []
    # 按列遍历dataframe
    for column_index, row_data in data.iteritems():
        counts = row_data.value_counts(normalize=True)
        # 若占比最大的某个值超过阈值，则记入待删除列
        if counts.iloc[0] >= threshold:
            column_indexs.append(column_index)
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
    print(columns_float)
    print(columns_int)
    # 随机选择16个特征做分布图，整形做条形图，浮点型会多一个拟合曲线
    columns_int_samples = random.sample(columns_int, 16)
    columns_float_samples = random.sample(columns_float, 16)
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
        sns.distplot(data[columns_int_samples[i-1]], kde=False, hist=True, hist_kws={'histtype':'stepfilled'},
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

def palette():
    '''
    调色板
    :return:
    '''
    # 标准彩色调色板，12色
    sns.palplot(sns.hls_palette(12, l=0.7, s=0.9))
    sns.palplot(sns.diverging_palette(240, 10, sep=12))
    # cmap = sns.diverging_palette(200, 20, as_cmap=True)
    sns.palplot(sns.color_palette('rainbow', 12))

    # 自定义
    colors = ["deepskyblue", "salmon"]
    # colors = ['#00bfff', '#fa8072']
    sns.palplot(sns.color_palette(colors))
    plt.show()

if __name__ == '__main__':
    # # setting
    # file = './dataset/Molecular_Descriptor.xlsx'
    # data = pd.read_excel(file)
    # #删掉第一列，分子结构，只保留特征
    # data = data.iloc[:,1:]
    # print(data.info)
    # # [1974 rows x 729 columns]>
    #
    #
    # # testing
    # data = del_same_feature(data)
    # print(data.info)
    # # [1974 rows x 504 columns]>
    #
    # data = del_perc_same_feature(data, 0.9)
    # print(data.info)
    # # [1974 rows x 362 columns] >
    #
    # data = del_std_small_feature(data, 0.05)
    # # [1974 rows x 341 columns] >

    data = pd.read_csv('test_data.csv')
    draw_feature(data)
    # palette()