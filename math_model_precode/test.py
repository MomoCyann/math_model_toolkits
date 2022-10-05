import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
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

def plot_box_indicator(df):
    '''
    :param df: 各模型评价指标
    :return:   4个评价指标的箱线图
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    sns.set_theme(style="whitegrid")

    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    # 对所有指标进行遍历画图
    for column in df.columns[:1]:
        ax = sns.boxplot(x="model", y=column, data=df, hue='model', dodge=False,
                    showmeans=True,
                    meanprops={"marker": "d",
                               "markerfacecolor": "white",
                               "markeredgecolor": "black",},
                    palette=sns.diverging_palette(240, 10, sep=12))
        model_labels = ['KNN', '多层感知机', '随机森林回归', '支持向量机回归', 'XGBoost']

        n = 0
        for i in model_labels:
            ax.legend_.texts[n].set_text(i)
            n += 1

        plt.show()

if __name__ == '__main__':
    df = pd.read_csv('dataset/regression_result.csv')
    sns.set_theme(style="whitegrid")
    # palette('simple')
    plot_box_indicator(df)