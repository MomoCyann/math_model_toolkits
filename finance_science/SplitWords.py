import jieba as jb
import pandas_method as pd
from jieba import analyse


def spiltWord(df):
    for index in df.index:
        print(index)
        #若某行摘要为空，则删除该行数据
        if pd.isnull(df.loc[index, 'newsSummary']):
            df.drop(index,axis=0,inplace=True)
            print(f"----------------------{index}")
            continue

        #-T表示Tiltle， -S表示Summary
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


df = pd.read_csv('data/all_data.csv')
print(df.info())
# 停用词
with open('stopwords/hit_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [s.rstrip() for s in f.readlines()]
# analyse.set_stop_words('stopwords/hit_stopwords.txt')

# 添加本地词库
jb.load_userdict('stopwords/userDict.txt')

spiltWord(df)

df=df.loc[:,~df.columns.str.contains('Unnamed')]
print(df.info())
print(df.head())
df.to_csv('clear_data/all_data_split_word.csv')

