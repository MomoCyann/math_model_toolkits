import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import re
import glob

def del_same_news(data):
    '''
    新闻数据去重。方法：把需要保存下来的新闻，包括独一无二的、和重复里面决定留下来的新闻的ID保存下来，在原始数据中保留这些ID的行。
    :param data:
    :return:
    '''
    # 复制一份数据
    data_copy = data.copy()
    # 去掉标点符号 方便比较
    data_copy['newsTitle'] = data_copy['newsTitle'].apply(lambda x: re.sub(r'[\W]', "", str(x)))
    data_copy['newsSummary'] = data_copy['newsSummary'].apply(lambda x: re.sub(r'[\W]', "", str(x)))
    # 重复数据全部去除→clean
    clean = data_copy.drop_duplicates(subset=['newsTitle'], keep=False)
    # 这个操作可以单独取出重复新闻
    duplic = data_copy.append(clean).drop_duplicates(keep=False)
    # 按时间与标题排序
    duplic = duplic.sort_values(by=['newsPublishTime','newsTitle'])


    # 优先保留来源者等于发布者
    # 先根据标题分组
    news_group = list(duplic.groupby(by='newsTitle'))
    # 需留存的的新闻ID
    # clean新闻的ID
    news_ID = list(clean.iloc[:, 0])
    for i in news_group:
        # 对于每一组，先检验有没有来源者等于发布者的新闻
        if (i[1]['newsOriginSource']==i[1]['newsPublishSite']).any():
            news_ID.append(i[1].iloc[:, 0].loc[i[1]['newsOriginSource']==i[1]['newsPublishSite']].iloc[0])
        # 否则选最早的
        else:
            news_ID.append(i[1].iloc[0, 0])
    # 根据ID保留新闻
    data = data.loc[news_ID, :]
    # 排序
    data.sort_index(inplace=True)
    data.reset_index(inplace=True, drop=True)
    data.index = data.index + 1
    data = data.iloc[:,1:]
    return data

def clear_data():
    '''
    遍历所有数据，进行去重
    :return:
    '''
    for filename in glob.glob("./data/*.csv"):
        data = pd.read_csv(filename)
        print(data.info)
        data = del_same_news(data)
        print('data cleared')
        print(data.info)

        # 保存
        name = re.sub('[^\u4e00-\u9fa5]+', '', filename)
        data.to_csv('./clear_data/' + name + '1-12.csv')

if __name__ == "__main__":

    clear_data()

