import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import glob

# 合并当前目录下所有csv文件
def concat_all_company(file_path):
    files=glob.glob(file_path)
    df = pd.read_excel(files[0])
    for f in files[1:]:
        df_temp = pd.read_excel(f)
        df = pd.concat([df, df_temp], axis=0)

    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['年份','月份'], ascending=[False,True], inplace=True)

    #csv总表存储路径, 获取clear_data 等二级目录
    df.to_excel("数据集/基本数据/附件8、锡林郭勒盟气候2012-2022/weather.xls", index=0)

# 合并当前目录下所有csv文件
def merge_all_company(file_path):
    files=glob.glob(file_path)
    df = pd.read_excel(files[0])
    for f in files[1:]:
        df_temp = pd.read_excel(f)
        df = pd.merge(df, df_temp, on=['年份','月份','经度(lon)','纬度(lat)'], how='left')

    df = df.loc[:, ~df.columns.str.contains('Unnamed','站点号')]
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['年份','月份'], ascending=[True,True], inplace=True)
    y = df.pop('年份')
    m = df.pop('月份')
    df.insert(0, '月份', m)
    df.insert(0, '年份', y)
    #csv总表存储路径, 获取clear_data 等二级目录
    df.to_csv("数据集/整理数据/all_data1.csv", index=0)


if __name__ == '__main__':

    # concat_all_company('数据集/基本数据/附件8、锡林郭勒盟气候2012-2022/*.xls')
    merge_all_company('数据集/基本数据/temp/*.xls')