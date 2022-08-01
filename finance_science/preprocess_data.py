import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import re
import glob
import jieba as jb


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
    duplic = duplic.sort_values(by=['newsPublishTime', 'newsTitle'])

    # 优先保留来源者等于发布者
    # 先根据标题分组
    news_group = list(duplic.groupby(by='newsTitle'))
    # 需留存的的新闻ID
    # clean新闻的ID
    news_ID = list(clean.iloc[:, 0])
    for i in news_group:
        # 对于每一组，先检验有没有来源者等于发布者的新闻
        if (i[1]['newsOriginSource'] == i[1]['newsPublishSite']).any():
            news_ID.append(i[1].iloc[:, 0].loc[i[1]['newsOriginSource'] == i[1]['newsPublishSite']].iloc[0])
        # 否则选最早的
        else:
            news_ID.append(i[1].iloc[0, 0])
    # 根据ID保留新闻
    data = data.loc[news_ID, :]
    # 排序
    data.sort_index(inplace=True)
    data.reset_index(inplace=True, drop=True)
    data.index = data.index + 1
    data = data.iloc[:, 1:]
    return data


def spiltWord(df, stopwords):
    for index in df.index:
        print(index)
        # 若某行摘要为空，则删除该行数据
        if pd.isnull(df.loc[index, 'newsSummary']):
            df.drop(index, axis=0, inplace=True)
            print(f"----------------------{index}")
            continue

        company_name = str(df.loc[index, 'secShortName'])
        if str(df.loc[index, 'newsTitle']).find(company_name) == -1 and str(df.loc[index, 'newsSummary']).find(
                company_name) == -1:
            df.drop(index=index, axis=0, inplace=True)
            print(1, index)
            continue

        # -T表示Tiltle， -S表示Summary
        # 分词
        segT = jb.lcut(df.loc[index, 'newsTitle'])
        segS = jb.lcut(df.loc[index, 'newsSummary'])

        # 去停用词
        splitWordT = []
        for w in segT:
            # 若某词在停用词表内或为空格，则去除
            if w not in stopwords and w != ' ':
                splitWordT.append(w)

        splitWordS = []
        for w in segS:
            # 若某词在停用词表内或为空格，则去除
            if w not in stopwords and w != ' ':
                splitWordS.append(w)

        # 将列表转字符串
        df.loc[index, 'splitTile'] = ','.join(splitWordT)
        df.loc[index, 'splitSummary'] = ','.join(splitWordS)


def split_main(files_path):
    files = glob.glob(files_path)

    # 停用词
    with open('stopwords/hit_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [s.rstrip() for s in f.readlines()]
    # analyse.set_stop_words('stopwords/hit_stopwords.txt')

    # 添加本地词库
    jb.load_userdict('stopwords/userDict.txt')

    for f in files:
        df = pd.read_csv(f)
        print(df.info())
        # 带后缀 .csv 形式
        company_name = f.split('\\')[1]

        spiltWord(df, stopwords)

        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        df.reset_index(drop=True, inplace=True)
        print(df.info())
        print(df.head())
        df.to_csv(f"data/split_data/{company_name}")


def clear_data():
    '''
    遍历所有数据，进行去重
    :return:
    '''
    for filename in glob.glob("./data/original_data/*.csv"):
        data = pd.read_csv(filename)
        print(data.info)
        data = del_same_news(data)
        print('data cleared')
        print(data.info)

        # 保存
        name = re.sub('[^\u4e00-\u9fa5]+', '', filename)
        data.to_csv('./data/clear_data/' + name + '1-12.csv')

# 合并当前目录下所有csv文件
def concat_all_company(file_path):
    files=glob.glob(file_path)
    df = pd.read_csv(files[0])
    for f in files[1:]:
        df_temp = pd.read_csv(f)
        df = pd.concat([df, df_temp], axis=0)

    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    df.reset_index(drop=True, inplace=True)
    #csv总表存储路径, 获取clear_data 等二级目录
    df.to_csv(f"./data/{files[0].split('//')[1]}/all_data.csv")

"""
数据处理逻辑
1. 去重,去除重复新闻,即标题一致
    具体逻辑: 标题一致,保留来源者=发布者;  标题不一致,保留发布时间最早的那篇
2. 去除无关新闻
    无关新闻: 标题/摘要不含该公司名称
3. 分词
    将标题/摘要 去除停用词,并进行分词,以',' 间隔
"""

if __name__ == "__main__":
    clear_data()

    # file_path 是去重后的存储路径
    file_path = ''
    split_main(file_path)
