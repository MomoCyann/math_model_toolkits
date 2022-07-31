import tensorflow_text as text
import tensorflow_hub as hub
import pandas as pd
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

def embedding_text(text):
    input_id = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    output = model(input_id["input_ids"])

    print(output)
    pytorch_tensor = output[1]
    np_tensor = pytorch_tensor.detach().numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor)
    return tf_tensor

# df['sentiment']
# df['newsTitle']
# df['newsSummary']
if __name__ == '__main__':
    df = pd.read_csv('./clear_data/all_data.csv')
    print(df.head())
    df = df[['newsTitle', 'newsSummary','sentiment']]
    print(df['sentiment'].value_counts())
    test = list(df.loc[:3,'newsTitle'])
    tensor = embedding_text(test)
    print("1")

