import pandas_method as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import *
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
import keras.losses
from sklearn.preprocessing import MinMaxScaler

'''
0 - 中性
1 - 利好
2 - 利空
'''
# 特征数量
feature_num = 0

# f1 score callback
class Metrics(keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        # print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        print(classification_report(val_targ, val_predict, digits=5))
        return

def load_data(path):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = pd.read_csv(path)
    # 标签与张量
    interval=20
    label_interval=20
    delete_columns=[]
    labels = pd.DataFrame(data=None, columns=['label'])
    for index in data.index:
        if index + interval+ label_interval <= data.shape[0]:
            timesteps_data = data.loc[index : index + interval-1, :]
            label_judge = data.loc[index + interval+ label_interval-1, 'closePrice'] - data.loc[index+ interval-1, 'closePrice']
            if label_judge > 0:
                labels.loc[index, 'label'] = 1
            else:
                labels.loc[index, 'label'] = 0
            if index == 0:
                dataset = timesteps_data.iloc[:, 6:].to_numpy()
            elif index == 1:
                dataset = np.stack((dataset, timesteps_data.iloc[:, 6:].to_numpy()), axis=0)
            else:
                dataset = np.vstack((dataset, (timesteps_data.iloc[:, 6:].to_numpy())[np.newaxis, :, :]))

    dataset_temp = np.reshape(dataset,(-1,263))

    dataset_temp_frame = pd.DataFrame(dataset_temp)
    print(dataset_temp_frame.shape)
    dataset_temp_frame = dataset_temp_frame.dropna(axis=1)
    print("fillna~")
    print(dataset_temp_frame.shape)
    feature_num = dataset_temp_frame.shape[1]

    dataset_temp = scaler.fit_transform(dataset_temp_frame)
    dataset = dataset_temp.reshape((int(dataset_temp.shape[0]/interval), interval, dataset_temp.shape[1]))
    print(labels['label'].value_counts())

    # 顺序采样
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, shuffle=False)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

    print(len(x_train), 'train examples')
    print(len(x_test), 'test examples')
    print(len(x_val), 'val examples')
    return x_train, x_test, x_val, y_train, y_test, y_val

def label_to_tensor(labels):
    # labels独热编码
    print(labels.value_counts())
    labels = pd.DataFrame(to_categorical(labels)).to_numpy()
    return labels

def net_NN():
    # 全连接网络 76%
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(300,), name='layer1'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1, callbacks=Metrics(valid_data=(x_val, y_val)))
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy", accuracy)
    # # 保存模型
    # LSTM模型参数.save('./BERT/LSTM模型参数/NN')


def net_LSTM():
    # LSTM
    # 优化器

    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
    # myReduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001,
    #                                 cooldown=0, min_lr=0)
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=(20, 263)))
    model.add(LSTM(units=128))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val), verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy", accuracy)
    # # 保存模型
    # LSTM模型参数.save('./BERT/LSTM模型参数/LSTM')
    # #加载模型
    # LSTM模型参数 = tf.keras.models.load_model('./LSTM模型参数/text_classifier')


def net_BiLSTM():
    # BiLSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True), input_shape=(1, 768)))
    model.add(Bidirectional(LSTM(units=128)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=60, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1, callbacks=Metrics(valid_data=(x_val, y_val)))
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy", accuracy)
    # # 保存模型
    # LSTM模型参数.save('./BERT/LSTM模型参数/BiLSTM')
    # #加载模型
    # LSTM模型参数 = tf.keras.models.load_model('./LSTM模型参数/text_classifier')


if __name__ == '__main__':
    # 读数据
    path = 'data/HS300/tradeData/179_贵州茅台_trade_20-22.csv'
    x_train, x_test, x_val, y_train, y_test, y_val = load_data(path)
    y_train = label_to_tensor(y_train)
    y_test = label_to_tensor(y_test)
    y_val = label_to_tensor(y_val)
    # 直接分类 72%
    # LSTM模型参数 = Sequential()
    # LSTM模型参数.add(Dense(3, activation='softmax',input_shape=(768,)))

    # 若选用LSTM
    # 先把X的2维数组变成3维数组, 不用重塑Y

    '''
    只需在下方调整方法名，即可选用不同网络训练
    注意：注释上方的重塑数组shape，在LSTM取消注释
    结果：
    # 全连接 epoch-30
    loss: 0.5125 - accuracy: 0.7673 - val_loss: 0.5351 - val_accuracy: 0.7592 - val_f1: 0.7592 - val_recall: 0.7592 - val_precision: 0.7592
    测试集 Accuracy 0.758499026298523

    # LSTM epoch-30
    loss: 0.4807 - accuracy: 0.7843 - val_loss: 0.5275 - val_accuracy: 0.7616 - val_f1: 0.7616 - val_recall: 0.7616 - val_precision: 0.7616
    测试集 Accuracy 0.7653766870498657

    # BiLSTM epoch-30
    loss: 0.4779 - accuracy: 0.7882 - val_loss: 0.5171 - val_accuracy: 0.7643 - val_f1: 0.7643 - val_recall: 0.7643 - val_precision: 0.7643
    测试集 Accuracy 0.76434326171875
    '''
    net_LSTM()