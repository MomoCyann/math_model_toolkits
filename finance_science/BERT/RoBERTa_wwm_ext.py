from transformers import BertTokenizer, BertModel, BertConfig

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

sentences = ["今天天气真好","今天是大晴天","有你在，就是晴天。"]

input_id = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
print(input_id)
print(input_id["input_ids"].shape)

output = model(input_id["input_ids"])

print(output)

import torch
import tensorflow as tf
pytorch_tensor = output[1]
np_tensor = pytorch_tensor.detach().numpy()
tf_tensor = tf.convert_to_tensor(np_tensor)

print(tf_tensor.shape)
import tensorflow_text as text
import tensorflow_hub as hub