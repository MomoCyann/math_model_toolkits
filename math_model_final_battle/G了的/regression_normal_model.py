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
from sklearn.preprocessing import MinMaxScaler

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

def train_data(column, test_size=0.8):
    # 创建df，存放n个模型的4项指标
    result = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'R2'])
    df_index = 0  # index递增，用于存放数据

    models=[KNeighborsRegressor(),SVR(),RandomForestRegressor()]
    models_name = ['KNN', 'SVR', 'RF']  # 模型名字，方便画图

    df = pd.read_csv('../数据集/整理数据/all_data_ym.csv')
    df = df.iloc[:-4]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


    def create_dataset(dataset, look_back):
        # 这里的look_back与timestep相同
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)

    train_size = int(len(df) * test_size)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_ = scaler.fit_transform(np.array(df[column]).reshape(-1, 1))


    train = df_[:train_size]
    test = df_[train_size:]

    result = pd.DataFrame(columns=['step', 'MAE', 'MSE', 'RMSE', 'R2'])
    result_index = 0
    # 训练数据太少 look_back并不能过大
    for set in [12]:

        look_back = set
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        x_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
        x_test = np.reshape(testX, (testX.shape[0], testX.shape[1]))
        y_train = trainY.reshape(-1,1)
        y_test = testY.reshape(-1,1)
        model_index = 0
        for model in models:
            model_name = models_name[model_index]
            print(f"当前模型{model_name}")
            mae, mse, rmse, r2 = 0, 0, 0, 0
            times = 1
            for i in range(times):
                model.fit(x_train, y_train)


                # 获取训练集评价指标
                y_pre_tr = model.predict(x_train)
                mae_, mse_, rmse_, r2_ = get_eval_indicator(y_train, y_pre_tr)
                mae += mae_
                mse += mse_
                rmse += rmse_
                r2 += r2_
            mae /= times
            mse /= times
            rmse /= times
            r2 /= times
            result.loc[df_index, 'LSTM模型参数'] = model_name
            result.loc[df_index, 'step'] = set
            result.loc[df_index, 'MAE_t'] = mae
            result.loc[df_index, 'MSE_t'] = mse
            result.loc[df_index, 'RMSE_T'] = rmse
            result.loc[df_index, 'R2_t'] = r2
            print(f"train+{r2}")
            # 获取测试集的评价指标
            y_pre = model.predict(x_test)
            mae, mse, rmse, r2 = get_eval_indicator(y_test, y_pre)

            result.loc[df_index, 'MAE'] = mae
            result.loc[df_index, 'MSE'] = mse
            result.loc[df_index, 'RMSE'] = rmse
            result.loc[df_index, 'R2'] = r2
            print(f"test+{r2}")
            model_index += 1
            df_index += 1

            # 反归一化
            trainPredict = scaler.inverse_transform(y_pre_tr.reshape(-1, 1))
            trainY = scaler.inverse_transform(y_train.reshape(-1, 1))
            testPredict = scaler.inverse_transform(y_pre.reshape(-1, 1))
            testY = scaler.inverse_transform(y_test.reshape(-1, 1))


            # plt.plot(trainY)
            # plt.plot(trainPredict, color='salmon')
            # plt.plot(testY)
            # plt.plot(testPredict, color='salmon')
            # plt.show()

    # 将数据按model排序，同一个model排在一起
    result.sort_values(by='LSTM模型参数', inplace=True)

    print(result.head(15))

    result.to_csv('result_' + column + '_垃圾模型_步长' + str(set)+'.csv')
    df2 = pd.read_csv('result_' + column + '_垃圾模型_步长' + str(set)+'.csv')
    print(df2.groupby('LSTM模型参数').mean())
    df2.groupby(['LSTM模型参数', 'step']).mean().to_csv('result_' + column + '_垃圾模型_步长' + str(set)+'.csv')
    # row_mean = df_final_test.mean(axis=1)
    # row_mean.to_csv('dataset/regression_final_result.csv')

    # # 画各评价指标箱线图
    # plot_box_indicator(df)


if __name__ == '__main__':

    test_size = 0.8
    columns = ['降水量(mm)']
    for column in columns:
        train_data(column, test_size)
