import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split

data = pd.read_csv()

train, test = train_test_split(data, test_size=0.2,random_state=42)
train, val = train_test_split(train, test_size=0.2,random_state=42)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dataframe, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 64
train_dataset = df_to_dataset(train, batch_size=batch_size)
val_dataset = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_dataset = df_to_dataset(test, shuffle=False, batch_size=batch_size)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, activation='relu',input_shape=(768,),name='layer1'))
model.add(tf.keras.layers.Dense(128, activation='relu',name='layer2'))
model.add(tf.keras.layers.Dropout(0.5,name='dropout'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid',name='output'))


model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_dataset,epochs=1000,
                    validation_data=val_dataset,verbose=1)

loss, accuracy = model.evaluate(test_dataset)
print("Accuracy", accuracy)



#保存模型
model.save('./model/text_classifier')
#加载模型
model = tf.keras.models.load_model('./model/text_classifier')