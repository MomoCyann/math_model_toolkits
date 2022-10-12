import time
import pandas as pd
import sklearn.metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import datetime

import os
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def get_eval_indicator(y_test, y_pre):
    '''
    :param y_test: 真实值
    :param y_pre: 预测值（模型预测出来的)
    :return: 4种评价指标
    返回回归任务的4种评价指标
    '''
    mae = mean_absolute_error(y_test, y_pre)
    mse = mean_squared_error(y_test, y_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_pre))
    r2 = r2_score(y_test, y_pre)
    return mae, mse, rmse, r2


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
        ax = sns.boxplot(x="LSTM模型参数", y=column, data=df, hue='LSTM模型参数', dodge=False,
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


def train_data(X, Y):
    print(X.shape)
    print(Y.shape)

    # 创建df，存放n个模型的4项指标
    df = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'R2'])
    df_index = 0  # index递增，用于存放数据

    # df_final_test = pd.DataFrame()

    # 模型库，每次分出训练集与测试集，均在该模型库中遍历训练
    # models = [KNeighborsRegressor(weights='distance', n_neighbors=7, algorithm='kd_tree'),
    #           MLPRegressor(solver='sgd', max_iter=1700, learning_rate_init=0.001, hidden_layer_sizes=(256,),
    #                        batch_size=128, alpha=0.0001, activation='tanh',),
    #           SVR(kernel='rbf', C=0.7),
    #           RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_leaf=1),
    #           XGBRegressor(n_estimators=600, max_depth=3, gamma=0.2, min_child_weight=4,
    #                            subsample=0.7, colsample_bytree=0.8,
    #                            reg_alpha=0.05, reg_lambda=0.1)]
    models=[KNeighborsRegressor(),MLPRegressor(),SVR(),RandomForestRegressor(),XGBRegressor()]
    models_name = ['KNN', 'MLP', 'SVR', 'RF', 'XGBoost']  # 模型名字，方便画图

    # # 测试模型
    # models = [XGBRegressor(n_estimators=600, max_depth=3, gamma=0.2, min_child_weight=4,
    #                        subsample=0.7, colsample_bytree=0.8,
    #                        reg_alpha=0.05, reg_lambda=0.1),
    #           XGBRegressor(),
    #           RandomForestRegressor(n_estimators=1000, max_depth=25, min_samples_leaf=1),
    #           RandomForestRegressor()
    #           ]
    # models_name = ['XGBoost', 'XGBoost_o', 'RF', 'RF_o']

    # 十次十折交叉检验
    skf = RepeatedKFold(n_repeats=2, n_splits=10, random_state=17)

    for train_index, test_index in skf.split(X, Y):
        # 获取训练集与测试集
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        print(x_train.shape, x_test.shape)
        print(y_train.shape, y_test.shape)

        model_index = 0
        for model in models:
            model_name = models_name[model_index]
            print(f"当前模型{model_name}")

            if model_name == 'XGBoost':
                eval_set = [(x_train, y_train), (x_test, y_test)]
                model.fit(x_train, y_train, eval_set=eval_set, eval_metric='mae',verbose=True,early_stopping_rounds=10)
                # results = LSTM模型参数.evals_result()
                # plt.plot(results['validation_0']['mae'], label='train')
                # plt.plot(results['validation_1']['mae'], label='test')
                # plt.legend()
                # plt.show()

            else:
                model.fit(x_train, y_train)


            # 获取训练集评价指标
            y_pre_tr = model.predict(x_train)
            mae, mse, rmse, r2 = get_eval_indicator(y_train, y_pre_tr)

            df.loc[df_index, 'LSTM模型参数'] = model_name
            df.loc[df_index, 'MAE_t'] = mae
            df.loc[df_index, 'MSE_t'] = mse
            df.loc[df_index, 'RMSE_T'] = rmse
            df.loc[df_index, 'R2_t'] = r2
            print(f"train+{r2}")
            # 获取测试集的评价指标
            y_pre = model.predict(x_test)
            mae, mse, rmse, r2 = get_eval_indicator(y_test, y_pre)

            df.loc[df_index, 'MAE'] = mae
            df.loc[df_index, 'MSE'] = mse
            df.loc[df_index, 'RMSE'] = rmse
            df.loc[df_index, 'R2'] = r2
            print(f"test+{r2}")
            model_index += 1
            df_index += 1

            # y_final_pre = LSTM模型参数.predict(X_test)
            # df_final_test = pd.concat([df_final_test, pd.DataFrame(y_final_pre)], axis=1)
    # print(df.info())

    # 将数据按model排序，同一个model排在一起
    df.sort_values(by='LSTM模型参数', inplace=True)

    print(df.head(15))

    df.to_csv('dataset/regression_result.csv')
    df2 = pd.read_csv('dataset/regression_result.csv')
    print(df2.groupby('LSTM模型参数').mean())
    df2.groupby('LSTM模型参数').mean().to_csv('dataset/regression_result_mean.csv')
    # row_mean = df_final_test.mean(axis=1)
    # row_mean.to_csv('dataset/regression_final_result.csv')

    # 画各评价指标箱线图
    plot_box_indicator(df)


if __name__ == '__main__':
    # 获取自变量
    # X = pd.read_excel('dataset/Molecular_Descriptor.xlsx', index_col=[0], sheet_name='training')
    # X_test = pd.read_excel('dataset/Molecular_Descriptor.xlsx', index_col=[0], sheet_name='test')

    X = pd.read_csv('dataset/molecular_remove0_scale.csv')

    df_feature = pd.read_csv('dataset/final_feature.csv')
    print(X.info())
    X = X.loc[:, df_feature.iloc[:20, 1]]

    print(X.info())

    # X_test = X_test.loc[:, df_feature.iloc[:20, 1]]
    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X))
    # X_test = pd.DataFrame(scaler.transform(X_test))


    # 获取因变量
    Y = pd.read_excel('dataset/ERα_activity.xlsx', index_col=[0], sheet_name='training')
    # 因变量归一化(如有必要)
    # scaler = preprocessing.MinMaxScaler()
    # Y = pd.DataFrame(columns=Y.columns, data=scaler.fit_transform(Y))
    Y = Y.iloc[:, 1]
    Y_test = Y.copy()
    # 训练并画图
    # train_data(X, Y, X_test)
    train_data(X, Y)
