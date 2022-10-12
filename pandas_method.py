import pandas as pd

df4=df1.drop(labels=['A','B','C'],axis=1)

# 索引 重设索引 重置索引
df_local.set_index('column', inplace=True)
df.reset_index(drop=True, inplace=True)

# 读取 不读取 第一列
df = pd.read_csv(fpath,index_col=False)
# 保存 不保存 第一列 索引
data.to_csv('数据集/整理数据/附件3、土壤湿度2022—2012年.csv', index=0)
# 重排 从大到小 从小到大 重新排列 某一列
df.sort_values(by=['time_list'], ascending=True, inplace=True)

# 筛选 范围 数据 区间
stu = stu.loc[stu['age'].apply(lambda a:18<=a<=30)]
'''
def age_18_to30(a):# 留下18<=年龄<=30
    return 18 <= a <= 30

# 留下 85<=score
def level_a(s):
    return 85 <= s

# 使用loc会生成一个新的series
stu = stu.loc[stu['age'].apply(age_18_to30)]
# 或者用下lambda表达式:
# stu = stu.loc[stu['age'].apply(lambda a:18<=a<=30)]
'''

# dtype 类型
df_outliers.dtypes

# 空表 创建
df = pd.DataFrame(data=None, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest',
                                          'sentimentFactor'])

# 列表 插入 dataframe 追加 加入
dataFrame = dataFrame.append(pd.DataFrame(myList, columns=['Country', 'Rank', 'Points']), ignore_index=True)