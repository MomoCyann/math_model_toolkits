import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import *
from sklearn.model_selection import StratifiedShuffleSplit
import keras_metrics

'''
0 - 中性
1 - 利好
2 - 利空
'''

def df_to_tensor(df):
    # labels独热编码
    labels = df.iloc[:, -1].to_frame()
    print(labels.value_counts())
    labels = pd.DataFrame(to_categorical(labels)).to_numpy()
    df = df.iloc[:,1:-1].to_numpy()

    return df, labels

def load_data(path):
    data = pd.read_csv(path)
    data['label'] = data['label'].replace(-1, 2)
    print(data['label'].value_counts())
    # 分层抽样
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["label"]):
        train = data.loc[train_index]
        test = data.loc[test_index]

    # # 随机抽样
    # train, test = train_test_split(data, test_size=0.2, random_state=42)
    # train, val = train_test_split(train, test_size=0.2, random_state=42)

    print(len(train), 'train examples')
    print(len(test), 'test examples')
    return train, test

#在每个epoch的末尾计算f1
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        print("\n"+classification_report(val_targ, val_predict))
        # report = classification_report(val_targ, val_predict, output_dict=True)

        _val_f1 = f1_score(val_targ, val_predict, average='binary')
        _val_recall = recall_score(val_targ, val_predict, average='binary')
        _val_precision = precision_score(val_targ, val_predict, average='binary')

        # _risk_recall = report['1']['recall']
        # _risk_precison = report['1']['precision']
        # _risk_f1 = report['1']['f1-score']

        logs['val_precision'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1

        # logs['risk_precision'] = _risk_precison
        # logs['risk_recall'] = _risk_recall
        # logs['risk_f1'] = _risk_f1


        return

if __name__ == '__main__':
    # 读取数据
    path = 'BERT/embedding_title_label_v2.csv'
    train, test = load_data(path)

    batch_size = 64
    x_train, y_train = df_to_tensor(train)
    x_test, y_test = df_to_tensor(test)

    # 网络
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu',input_shape=(768,),name='layer1'))
    model.add(tf.keras.layers.Dense(128, activation='relu',name='layer2'))
    model.add(tf.keras.layers.Dropout(0.5,name='dropout'))
    model.add(tf.keras.layers.Dense(3, activation='softmax',name='output'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           keras_metrics.f1_score(),
                           keras_metrics.precision(),
                           keras_metrics.recall()
                           ])
    history = model.fit(x_train, y_train ,epochs = 30, batch_size=64,
                        validation_split = 0.2 ,verbose = 1)
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy", accuracy)

    # #保存模型
    # model.save('./model/text_classifier')
    # #加载模型
    # model = tf.keras.models.load_model('./model/text_classifier')