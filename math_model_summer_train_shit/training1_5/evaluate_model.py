import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

'''
1.某年发生异常波动的次数
2.低于垄断的次数
3.低于垄断指数的商品数目
4.教育支出的基期增长率
5.养老支出的基期增长率
'''
def edu_year():
    df = pd.read_csv('data/edu_value.csv')
    result = pd.DataFrame(data=None, columns=['date', 'edu_value', 'med_value', 'edu_gr', 'med_gr'])

    index = 0
    date = 2014
    edu_payment_year = 0
    med_payment_year = 0
    while index < df.shape[0]:
        edu_payment_year += df.iloc[index, 1]
        med_payment_year += df.iloc[index, 2]
        if (index + 1) % 4 == 0:
            result.loc[(index + 1) / 4, 'edu_value'] = edu_payment_year
            result.loc[(index + 1) / 4, 'med_value'] = med_payment_year
            result.loc[(index + 1) / 4, 'date'] = date
            edu_payment_year = 0
            med_payment_year = 0
            index += 1
            date += 1
        else:
            index += 1
    result.reset_index(drop=True, inplace=True)
    # 计算增长率
    index = 1
    while index < result.shape[0]:
        result.loc[index, 'edu_gr'] = result.loc[index, 'edu_value'] / result.loc[0, 'edu_value']
        result.loc[index, 'med_gr'] = result.loc[index, 'med_value'] / result.loc[0, 'med_value']
        index += 1
    result.to_csv('data/edu_year.csv')
    print('complete')


def features():
    df_outliers = pd.read_csv('data/outliersResult.csv', index_col=0)
    df_outliers.sort_values(by=['month'], inplace=True)
    df_outliers.reset_index(drop=True, inplace=True)
    print(df_outliers.dtypes)
    result = pd.DataFrame(data=None, columns=['date', 'outlier_count', 'hold_count', 'hold_mean', 'edu_gr', 'med_gr'])

    count = 15
    index_count = 0
    df_edu = pd.read_csv('data/edu_year.csv')
    for date in range(20150101, 20220101,10000):
        result.loc[index_count, 'date'] = date
        # 统计每一年的异常数目
        outliers = df_outliers[df_outliers['month'].apply(lambda x:date<=x<=(date+10000))]
        result.loc[index_count, 'outlier_count'] = outliers.shape[0]
        # 统计垄断
        df_hold = pd.read_csv('data/problem2_7year/' + str(count) + '.csv')
        hold_count = df_hold[df_hold['total'].apply(lambda x: x <= 0.4)]
        hold_mean = df_hold['total'].mean()
        result.loc[index_count, 'hold_count'] = hold_count.shape[0]
        result.loc[index_count, 'hold_mean'] = hold_mean
        # 统计教育养老
        result.loc[index_count, 'edu_gr'] = df_edu.loc[index_count + 1, 'edu_gr']
        result.loc[index_count, 'med_gr'] = df_edu.loc[index_count + 1, 'med_gr']
        index_count += 1
        count += 1
    result.to_csv('data/evaluate_features.csv')
    print('complete')

def minmax_scaler():
    df = pd.read_csv('data/evaluate_features.csv')
    df = df.iloc[:, 2:]
    scaler = MinMaxScaler()
    result = scaler.fit_transform(df)
    pd.DataFrame(result).to_csv('data/evaluate_features_minmax.csv')
    print('complete')

def plot():

    df = pd.read_csv('data/evaluate_features.csv')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(12, 8))
    plt.plot(df.loc[:, 'date'], df.loc[:, 'score'], label='国民经济质量评价得分', color='salmon')
    plt.title('2015-2021年国民经济质量评价得分', size=18)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('得分', fontsize=16)
    plt.legend(prop={'size': 16})
    # plt.xticks(rotation=45)
    plt.show()

def more_features_preprocess():
    df = pd.read_csv('data/more_features.csv')

    print('ss')
    df = df.iloc[1:-7, :]
    df = df.T
    columns = df.iloc[0, 1:].tolist()
    df.set_index(1, inplace=True)
    df.columns = columns
    df = df.iloc[1:, :]
    df['newindex'] = np.arange(len(df) - 1, -1, -1)
    df.sort_values('newindex', inplace=True)
    df.drop('newindex', axis=1, inplace=True)
    df = df.iloc[3:, :]

    # minmax_scaler
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    pd.DataFrame(df).to_csv('data/more_features_minmax.csv')



def all_features():
    df_5features = pd.read_csv('data/evaluate_features_minmax.csv', header=0, index_col=0)
    df_more_fea = pd.read_csv('data/more_features_minmax.csv', header=0, index_col=0)
    df_all = pd.concat([df_5features,df_more_fea], axis=1)
    df_all.to_csv('data/evaluate_14features_minmax.csv')

    df_5features = pd.read_csv('data/evaluate_features.csv', header=0, index_col=0)
    df_more_fea = pd.read_csv('data/more_features_washed.csv', header=0, index_col=0)
    df_more_fea.reset_index(drop=True, inplace=True)
    df_all = pd.concat([df_5features, df_more_fea], axis=1)
    df_all.to_csv('data/evaluate_14features.csv')
    print('complete')
# edu_year()
# features()
# minmax_scaler()
plot()
# more_features_preprocess()
# all_features()