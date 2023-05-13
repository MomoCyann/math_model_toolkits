import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import *
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D, GlobalAveragePooling1D, SimpleRNN
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
import keras.losses
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 预处理，生成日期

def to_date():
    df = pd.read_csv('../数据集/整理数据/all_data1.csv')
    df['年份'] = df['年份'].astype(str)
    df['月份'] = df['月份'].astype(str)
    print(df.dtypes)
    df['date'] = df['年份'] + df['月份']
    df['date'] = pd.to_datetime(df['date'], format='%Y%m', errors='coerce').dt.to_period('m')
    df.drop(labels=['站点号'], axis=1, inplace=True)
    date = df.pop('date')
    df.insert(0, 'date', date)
    df.to_csv("数据集/整理数据/all_data.csv", index=0)
    print('1')

def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX),numpy.array(dataY)

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

def lstm_self():
    df = pd.read_csv('../数据集/整理数据/all_data_ym.csv')
    print(df.dtypes)
    print(df.head())
    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    df = df.iloc[:-4,:]

    train_size = int(len(df) * 0.8)

    def create_dataset(dataset, look_back):
        # 这里的look_back与timestep相同
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)

    # for column in ['40cm湿度(kgm2)', '100cm湿度(kgm2)']:
    for column in ['降水量(mm)']:

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_10cm = scaler.fit_transform(np.array(df[column]).reshape(-1, 1))

        train_10cm = df_10cm[:train_size]
        test_10cm = df_10cm[train_size:]

        result = pd.DataFrame(columns=['step','MAE', 'MSE', 'RMSE', 'R2'])
        result_index = 0
        # 训练数据太少 look_back并不能过大
        for set in [1]:
            look_back = set
            trainX, trainY = create_dataset(train_10cm, look_back)
            testX, testY = create_dataset(test_10cm, look_back)
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


            mae, mse, rmse, r2 = 0, 0, 0, 0
            times = 1
            for i in range(times):
                keras.backend.clear_session()
                # create and fit the LSTM network
                model = Sequential()
                model.add(LSTM(64, return_sequences=True, input_shape=(None, 1)))
                model.add(LSTM(32, ))
                # LSTM模型参数.add(Dropout(0.2))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                earlyStop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, mode='min',
                                                          verbose=1,
                                                          restore_best_weights=True)
                model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2,callbacks=[earlyStop])
                # 保存模型
                model.save('./LSTM模型参数/LSTM')
                # # #加载模型
                # # LSTM模型参数 = tf.keras.models.load_model('./LSTM模型参数/LSTM')

                trainPredict = model(trainX)
                testPredict = model(testX)
                mae_, mse_, rmse_, r2_ = get_eval_indicator(testY, testPredict)
                mae += mae_
                mse += mse_
                rmse += rmse_
                r2 += r2_
            mae /= times
            mse /= times
            rmse /= times
            r2 /= times
            # 反归一化
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform(trainY)
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform(testY)
            result.loc[result_index, 'step'] = set
            result.loc[result_index, 'MAE'] = mae
            result.loc[result_index, 'MSE'] = mse
            result.loc[result_index, 'RMSE'] = rmse
            result.loc[result_index, 'R2'] = r2
            result_index += 1

        # 预测
        for i in range(timerange):
            tezheng = df_10cm[-set-1:]
            print('1')
        # 反归一化原始数据
        df_10cm = scaler.inverse_transform(df_10cm)
        plt.plot(df['date'], df_10cm)

        plt.ylabel(column)
        plt.xlabel("日期")
        plt.xticks(range(0, df.shape[0], 3), df.loc[range(0, df.shape[0], 3), 'date'], rotation=45)

        plt.show()
        # print('timestep', set)
        result.to_csv('regression_result_no_val_' + str(column) + '.csv')

def predict(column):
    df = pd.read_csv('../数据集/整理数据/all_data_ym.csv')
    print(df.dtypes)
    print(df.head())
    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    df = df.iloc[:-4, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array(df[column]).reshape(-1, 1))

    #加载模型
    model = tf.keras.models.load_model('./LSTM模型参数/LSTM')
    # 预测
    timerange = 10
    step = 1
    result = pd.DataFrame(columns=['result'])
    tezheng = df[-step:]
    x = tezheng[column]
    xtest = np.array(x)
    for i in range(timerange):
        xtest = scaler.transform(xtest.reshape(-1, 1))
        # xtest = np.reshape(np.array(x), (x.shape[0], x.shape[1], 1))
        y = model.predict(xtest)
        a = scaler.inverse_transform(y)[0][0]
        result.loc[i, 'result'] = a

        xtest = np.append(xtest[0], a)
        xtest = np.delete(xtest,0)
        print('1')

    print('hello world')

def main():
    # arima_model()
    # lstm_self()
    column = '降水量(mm)'
    predict(column)
    print('hello world')
if __name__ == '__main__':
    main()
