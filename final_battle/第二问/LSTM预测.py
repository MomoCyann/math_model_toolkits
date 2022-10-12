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

def arima_model():
    df = pd.read_csv('../数据集/整理数据/all_data_ym.csv')
    df.index = pd.to_datetime(df["date"], format="%Y-%m-%d")
    print(df.dtypes)
    print(df.head())
    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


    train = df[df.index <= pd.to_datetime("202012", format="%Y%m", errors='coerce')]
    test = df[df.index >= pd.to_datetime("202012", format="%Y%m", errors='coerce')]
    train = train.loc[:,'降水量(mm)']
    test = test.loc[:, '降水量(mm)']
    test = test.iloc[:-4]
    plt.plot(train, color="black")
    plt.plot(test, color="red")
    plt.ylabel("40cm湿度(kg/m2)")
    plt.xlabel("Date")
    plt.xticks(rotation=45)

    y = train
    ARIMAmodel = ARIMA(y, order = (2, 3, 2))
    ARIMAmodel = ARIMAmodel.fit()
    y_pred = ARIMAmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df['Predictions'] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out = y_pred_df["Predictions"]
    plt.plot(y_pred_out, color="Yellow", label="ARIMA Predictions")
    plt.legend()
    arma_rmse = np.sqrt(mean_squared_error(test, y_pred_df["Predictions"]))
    print("RMSE: ", arma_rmse)

    plt.show()

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

def create_dataset(dataset, look_back):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(dataset.shape[0] - look_back+1):
        a = dataset[i:(i + look_back), :dataset.shape[1]-1]
        dataX.append(a)
        dataY.append(dataset[i + look_back-1, dataset.shape[1]-1])
    return np.array(dataX), np.array(dataY).reshape(-1, 1)

def lstm_feature():
    df0 = pd.read_csv('所有特征整合数据.csv')
    df0.index = pd.to_datetime(df0["date"], format="%Y-%m-%d")
    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    df = df0.iloc[:123,:]

    df_final = df0.iloc[118:,:]

    train_size = int(len(df) * 0.7)

    # train = df[df.index <= pd.to_datetime("202012", format="%Y%m", errors='coerce')]
    # test = df[df.index >= pd.to_datetime("202012", format="%Y%m", errors='coerce')]
    # train = train.loc[:, '降水量(mm)']
    # test = test.loc[:, '降水量(mm)']
    # test = test.iloc[:-4]

    for h in ['10cm湿度(kgm2)']:
        # df_final_test = df.iloc[123 - step:, :]

        scaler = MinMaxScaler()
        df_10cm_feature = scaler.fit_transform(np.array(df.loc[:,['降水量(mm)','土壤蒸发量(mm)', '植被指数(NDVI)','cp',h]]))
        df_final_10cm_feature = scaler.transform(np.array(df_final.loc[:,['降水量(mm)','土壤蒸发量(mm)', '植被指数(NDVI)','cp',h]]))
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        # scaler_x.fit(np.array(df.loc[:,['降水量(mm)','土壤蒸发量(mm)', 'icstore']]))
        scaler_x.fit(np.array(df.loc[:, ['降水量(mm)', '土壤蒸发量(mm)', '植被指数(NDVI)','cp']]))
        scaler_y.fit(np.array(df.loc[:,[h]]))
        # train_X_10cm_feature = df_10cm_feature[:train_size, :2]
        # train_Y_10cm_feature = df_10cm_feature[:train_size, 2]
        # test_X_10cm_feature = df_10cm_feature[train_size:, :2]
        # test_Y_10cm_feature = df_10cm_feature[train_size:, :2]

        train_10cm_feature = df_10cm_feature[:train_size]
        test_10cm_feature = df_10cm_feature[train_size:]

        testX_final, testY_final = create_dataset(df_final_10cm_feature, 6)

        result = pd.DataFrame(columns=['step','MAE', 'MSE', 'RMSE', 'R2'])
        result_index = 0
        # 训练数据太少 look_back并不能过大
        for set in [6]:
            look_back = set
            trainX, trainY = create_dataset(train_10cm_feature, look_back)
            testX, testY = create_dataset(test_10cm_feature, look_back)
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], df_10cm_feature.shape[1]-1))
            testX = np.reshape(testX, (testX.shape[0], testX.shape[1], df_10cm_feature.shape[1]-1))

            # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(32, return_sequences=True, input_shape=(None, df_10cm_feature.shape[1]-1)))
            model.add(LSTM(16))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            earlyStop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, mode='min', verbose=1,
                                      restore_best_weights=True)
            model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2,callbacks=[earlyStop])
            # # 保存模型
            # LSTM模型参数.save('./LSTM模型参数/LSTM_6step')
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            finalPredict = model.predict(testX_final)

            mae, mse, rmse, r2 = get_eval_indicator(testY, testPredict)
            # 反归一化
            trainPredict = scaler_y.inverse_transform(trainPredict)
            trainY = scaler_y.inverse_transform(trainY)
            testPredict = scaler_y.inverse_transform(testPredict)
            testY = scaler_y.inverse_transform(testY)
            finalPredict = scaler_y.inverse_transform(finalPredict)

            result.loc[result_index, 'step'] = set
            result.loc[result_index, 'MAE'] = mae
            result.loc[result_index, 'MSE'] = mse
            result.loc[result_index, 'RMSE'] = rmse
            result.loc[result_index, 'R2'] = r2
            result_index += 1


            # plt.plot(trainY)
            # plt.plot(trainPredict)
            # plt.show()
            # plt.plot(testY)
            # plt.plot(testPredict)
            # plt.show()
            test_data = pd.DataFrame(data=testPredict)
            final_result = pd.DataFrame(data=finalPredict)
            test_data.to_csv('test_data.csv')
            final_result.to_csv('10cm湿度LSTM预测数据.csv')


        print('ss')
        result.to_csv('regression_result_feature_' + h + '.csv')

def lstm_step(step):
    df = pd.read_csv('所有特征整合数据.csv')
    df.index = pd.to_datetime(df["date"], format="%Y-%m-%d")
    print(df.dtypes)
    print(df.head())
    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    df = df.iloc[123-step:,:]

    for h in ['10cm湿度(kgm2)']:
        scaler = MinMaxScaler(feature_range=(0, 1))

        df_10cm_feature = scaler.fit_transform(np.array(df.loc[:,['降水量(mm)','土壤蒸发量(mm)', '植被指数(NDVI)','cp',h]]))
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        # scaler_x.fit(np.array(df.loc[:,['降水量(mm)','土壤蒸发量(mm)', 'icstore']]))
        scaler_x.fit(np.array(df.loc[:, ['降水量(mm)', '土壤蒸发量(mm)', '植被指数(NDVI)','cp']]))
        scaler_y.fit(np.array(df.loc[:,[h]]))
        # train_X_10cm_feature = df_10cm_feature[:train_size, :2]
        # train_Y_10cm_feature = df_10cm_feature[:train_size, 2]
        # test_X_10cm_feature = df_10cm_feature[train_size:, :2]
        # test_Y_10cm_feature = df_10cm_feature[train_size:, :2]


        result = pd.DataFrame(columns=['step','MAE', 'MSE', 'RMSE', 'R2'])
        result_index = 0
        # 训练数据太少 look_back并不能过大
        for set in [1,2,3,4,5,6,7,8,9,10,11,12]:
            look_back = set
            trainX, trainY = create_dataset(train_10cm_feature, look_back)
            testX, testY = create_dataset(test_10cm_feature, look_back)
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], df_10cm_feature.shape[1]-1))
            testX = np.reshape(testX, (testX.shape[0], testX.shape[1], df_10cm_feature.shape[1]-1))

            # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(32, input_shape=(None, df_10cm_feature.shape[1]-1)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            earlyStop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, mode='min', verbose=1,
                                      restore_best_weights=True)
            model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2,callbacks=[earlyStop])

            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            mae, mse, rmse, r2 = get_eval_indicator(testY, testPredict)
            # 反归一化
            trainPredict = scaler_y.inverse_transform(trainPredict)
            trainY = scaler_y.inverse_transform(trainY)
            testPredict = scaler_y.inverse_transform(testPredict)
            testY = scaler_y.inverse_transform(testY)

            result.loc[result_index, 'step'] = set
            result.loc[result_index, 'MAE'] = mae
            result.loc[result_index, 'MSE'] = mse
            result.loc[result_index, 'RMSE'] = rmse
            result.loc[result_index, 'R2'] = r2
            result_index += 1
            # plt.plot(trainY)
            # plt.plot(trainPredict)
            # plt.show()
            # plt.plot(testY)
            # plt.plot(testPredict)
            # plt.show()

        print('ss')
        result.to_csv('regression_result_feature_' + h + '.csv')
def main():
    # arima_model()
    # lstm_self()
    lstm_feature()
    print('hello world')
if __name__ == '__main__':
    main()
