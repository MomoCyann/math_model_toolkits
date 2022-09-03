import pandas_method as pd
import numpy as np
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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''
0 - 中性
1 - 利好
2 - 利空
'''
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min', verbose=1,
                              restore_best_weights=True)
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

def df_to_tensor(df):
    # labels独热编码
    labels = df.iloc[:, -1].to_frame()
    print(labels.value_counts())
    labels = pd.DataFrame(to_categorical(labels)).to_numpy()
    df = df.iloc[:,1:-1].to_numpy()

    return df, labels

def load_data(path):
    data = pd.read_csv(path)
    # 把标签-1替换成2
    data['label'] = data['label'].replace(-1, 2)
    print(data['label'].value_counts())
    # 打乱数据
    data = shuffle(data)
    # 分层抽样
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["label"]):
        train = data.loc[train_index]
        test = data.loc[test_index]

    # # 随机抽样
    # train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)

    print(len(train), 'train examples')
    print(len(val), 'val examples')
    print(len(test), 'test examples')
    return train, test, val

def load_data_wv(path):
    data = pd.read_csv(path)
    print(data.isnull().sum())
    data = data.fillna(0)
    print("fillna~")
    print(data.isnull().sum())
    # 把标签-1替换成2
    data['label'] = data['label'].replace(-1, 2)
    print(data['label'].value_counts())
    # 打乱数据
    data = shuffle(data)
    # 分层抽样
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["label"]):
        train = data.loc[train_index]
        test = data.loc[test_index]

    # # 随机抽样
    # train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)

    print(len(train), 'train examples')
    print(len(val), 'val examples')
    print(len(test), 'test examples')
    return train, test, val

def reshape_to_3d(array):
    rows = array.shape[0]
    cols = array.shape[1]
    array = array.reshape((rows,1,cols))
    return array

def reshape_to_3d_CNN(array):
    rows = array.shape[0]
    cols = array.shape[1]
    array = array.reshape((rows,cols,1))
    return array

def net_NN():
    # 全连接网络 76%
    model = Sequential()
    model.add(Dense(256, activation='relu',input_shape=(300,),name='layer1'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1, callbacks=[earlyStop,
                                                                              Metrics(valid_data=(x_val, y_val))])
    loss, accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0

    print(classification_report(y_test, y_pred, digits=5))
    print(roc_auc_score(y_test, y_pred))
    print("Accuracy", accuracy)
    # 保存模型
    model.save('./BERT/model/NN')

def net_LSTM():
    # LSTM
    # 优化器
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape_1))
    model.add(LSTM(units=128))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=300, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1, callbacks=[earlyStop,
            Metrics(valid_data=(x_val, y_val))])
    loss, accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0

    print(classification_report(y_test, y_pred, digits=5))
    print(roc_auc_score(y_test, y_pred))
    print("Accuracy", accuracy)
    #保存模型
    model.save('./BERT/model/LSTM')
    # #加载模型
    # model = tf.keras.models.load_model('./model/text_classifier')

def net_BiLSTM():
    # BiLSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256,return_sequences=True),input_shape=input_shape_1))
    model.add(Bidirectional(LSTM(units=128)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=60, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1, callbacks=[earlyStop,
            Metrics(valid_data=(x_val, y_val))])
    loss, accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0

    print(classification_report(y_test, y_pred, digits=5))
    print(roc_auc_score(y_test, y_pred))
    print("Accuracy", accuracy)
    #保存模型
    model.save('./BERT/model/BiLSTM')
    # #加载模型
    # model = tf.keras.models.load_model('./model/text_classifier')

def net_CNN():
    # CNN
    # 优化器
    model = Sequential()
    model.add(Conv1D(512, 1, activation='relu', input_shape=input_shape_1))
    model.add(Conv1D(256, 1, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=300, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1, callbacks=[earlyStop,
            Metrics(valid_data=(x_val, y_val))])
    loss, accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0

    print(classification_report(y_test, y_pred, digits=5))
    print(roc_auc_score(y_test, y_pred))
    print("Accuracy", accuracy)
    #保存模型
    model.save('./BERT/model/CNN')
    # #加载模型
    # model = tf.keras.models.load_model('./model/text_classifier')

def net_RNN():
    # LSTM
    # 优化器
    model = Sequential()
    model.add(SimpleRNN(units=256, return_sequences=True, input_shape=input_shape_1))
    model.add(SimpleRNN(units=128))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=300, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1, callbacks=[earlyStop,
            Metrics(valid_data=(x_val, y_val))])
    loss, accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0

    print(classification_report(y_test, y_pred, digits=5))
    print(roc_auc_score(y_test, y_pred))
    print("Accuracy", accuracy)
    #保存模型
    model.save('./BERT/model/RNN')
    # #加载模型
    # model = tf.keras.models.load_model('./model/text_classifier')

if __name__ == '__main__':
    bert = 1
    if bert == 1:
        # 读取数据_BERT
        # path = 'BERT/embedding_title_label_v2.csv'
        path = 'BERT/selected_embedding_title_label.csv'
        train, test, val = load_data(path)
        input_shape_1 = (1, 768)
    else:
        # 读取数据_wv
        path = 'word_vector/selected_sentence_vector.csv'
        train, test, val = load_data_wv(path)
        input_shape_1 = (1, 300)

    x_train, y_train = df_to_tensor(train)
    x_test, y_test = df_to_tensor(test)
    x_val, y_val = df_to_tensor(val)

    # 直接分类 72%
    # model = Sequential()
    # model.add(Dense(3, activation='softmax',input_shape=(768,)))

    # 若选用LSTM
    # 先把X的2维数组变成3维数组, 不用重塑Y
    x_train, x_test, x_val = reshape_to_3d(x_train), reshape_to_3d(x_test), reshape_to_3d(x_val)

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
    net_CNN()