import missingno as msno
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random

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

def draw_feature(data):
    # 遍历每列的类型
    columns_int = []
    columns_float = []
    for column_index in data.columns:
        if str(type(data[column_index][0])) == "<class 'numpy.int64'>":
            columns_int.append(column_index)
        if str(type(data[column_index][0])) == "<class 'numpy.float64'>":
            columns_float.append(column_index)
    print(columns_float)
    print(columns_int)
    # 随机选择16个做分布图
    columns_int_samples = random.sample(columns_int, 16)
    columns_float_samples = random.sample(columns_float, 16)
    data = data.dropna()
    for i in range(1,17):
        plt.subplot(4,4,i)
        sns.distplot(data[columns_int_samples[i-1]], kde=False, hist=True)
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    for i in range(1,17):
        plt.subplot(4,4,i)
        sns.distplot(data[columns_float_samples[i-1]], hist=True, norm_hist=False)
    # plt.subplots_adjust(wspace=10, hspace=10)
    plt.show()


if __name__ == '__main__':
    # setting
    file = './dataset/Molecular_Descriptor.xlsx'
    data = pd.read_excel(file)
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

    draw_feature(data)