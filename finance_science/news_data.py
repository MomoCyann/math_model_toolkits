import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import re
import glob

def del_same_news(data):
    '''
    新闻数据去重。
    :param data:
    :return:
    '''
    # 重复数据全部去除→clean
    clean = data.drop_duplicates(subset=['newsTitle'], keep=False)
    # 这个操作可以单独取出重复新闻
    duplic = data.append(clean).drop_duplicates(keep=False)
    # 按时间与标题排序
    duplic = duplic.sort_values(by=['newsPublishTime','newsTitle'])
    # 去掉标点符号
    duplic['newsTitle'] = duplic['newsTitle'].apply(lambda x: re.sub(r'[\W]', "", str(x)))
    duplic['newsSummary'] = duplic['newsSummary'].apply(lambda x: re.sub(r'[\W]', "", str(x)))

    # 优先保留来源者等于发布者
    # 先根据标题分组
    news_group = list(duplic.groupby(by='newsSummary'))
    # 需留存的的新闻ID
    news_ID = []
    for i in news_group:
        # 对于每一组，先检验有没有来源者等于发布者的新闻
        if (i[1]['newsOriginSource']==i[1]['newsPublishSite']).any():
            news_ID.append(i[1].iloc[:,0].loc[i[1]['newsOriginSource']==i[1]['newsPublishSite']].iloc[0])
        # 否则选最早的
        else:
            news_ID.append(i[1].iloc[0, 0])
    # 根据ID保留新闻
    duplic_source = duplic.loc[news_ID, :]
    # 与无重复的新闻合并
    data = pd.concat([clean, duplic_source], axis=0)
    # 排序
    data.sort_index(inplace=True)
    data.reset_index(inplace=True, drop=True)
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

    #clear_data()
    data = pd.read_csv("./data/海康威视1-12.csv")
    print(data.info)
    data = del_same_news(data)
    print('data cleared')
    print(data.info)
