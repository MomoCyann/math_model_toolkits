import tensorflow_text as text
import tensorflow_hub as hub
import pandas_method as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertConfig
import torch
'''
模型名	                        MODEL_NAME
RoBERTa-wwm-ext-large	hfl/chinese-roberta-wwm-ext-large
RoBERTa-wwm-ext	hfl/chinese-roberta-wwm-ext
BERT-wwm-ext	hfl/chinese-bert-wwm-ext
BERT-wwm	hfl/chinese-bert-wwm
RBT3	hfl/rbt3
RBTL3	hfl/rbtl3
'''

config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

def embedding_title(text):
    input_id = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    output = model(input_id["input_ids"])

    pytorch_tensor = output[1]
    np_tensor = pytorch_tensor.detach().numpy()
    return pd.DataFrame(np_tensor)

    # tf_tensor = tf.convert_to_tensor(np_tensor)
    # return tf_tensor

def embedding_news(text):
    input_id = tokenizer(text, padding=True, truncation=True, max_length=752, return_tensors="pt")
    output = model(input_id["input_ids"])

    pytorch_tensor = output[1]
    np_tensor = pytorch_tensor.detach().numpy()
    return pd.DataFrame(np_tensor)

# df['sentiment']
# df['newsTitle']
# df['newsSummary']
def merge_title_summary(df):
    df['newsTitle'] = df['newsTitle'] + '，'
    df['news'] = df['newsTitle'] + df['newsSummary']
    # # 统计字符串长度
    # df['title_length'] = df['newsTitle'].map(lambda x: len(str(x)))
    # df['news_length'] = df['news'].map(lambda x: len(str(x)))
    # df.to_csv('length_view.csv')
    return df

if __name__ == '__main__':
    # 读取数据
    # df = pd.read_csv('data/all_clear_data.csv')
    df = pd.read_csv('data/selected_newsData_split.csv')
    print(df.head())
    df = df[['newsTitle', 'newsSummary','sentiment']]
    print(df['sentiment'].value_counts())

    # 合并标题和摘要
    df = merge_title_summary(df)

    for index, rows in df.iterrows():
        title = df['newsTitle'][index]
        news = df['news'][index]
        if index == 0:
            title_embedding = embedding_title(title)
            title_embedding['label'] = df['sentiment'][index]

            # news_embedding = embedding_news(news)
            # news_embedding['label'] = df['sentiment'][index]
        else:
            t_embedding = embedding_title(title)
            t_embedding['label'] = df['sentiment'][index]
            title_embedding = title_embedding.append(t_embedding)

            # n_embedding = embedding_news(news)
            # n_embedding['label'] = df['sentiment'][index]
            # news_embedding = news_embedding.append(n_embedding)
        print(index)
    title_embedding.to_csv('./BERT/embedding_title_label.csv')
    title_embedding.to_csv('./BERT/selected_embedding_title_label.csv')
    # news_embedding.to_csv('./BERT/embedding_news_label.csv')

    print("end")

