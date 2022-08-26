import pandas as pd
import gensim
import numpy as np
from gensim.models import word2vec as wv

def get_vocabulary():
    print("中文测试")
    # df = pd.read_excel('./data/all_clear_data.xlsx')
    df = pd.read_csv('./data/selected_newsData_split.csv')
    df['str_spl']=df.splitSummary.str.split(',')
    print(df.head(2))
    all_word = []
    for index, rows in df.iterrows():
        all_word.extend(rows['str_spl'])

    print(len(all_word))
    # 去除重复
    vocabulary = list(set(all_word))
    print(len(vocabulary))
    return vocabulary

# 用于过滤在word2vec中的词
def get_vocabulary_vector(vocabulary):
    # 载入已下载的word2vec解压后的模型
    print("start word2vec load ......")
    vec_path = './word_vector/sgns.financial.word.bz2'
    wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(vec_path,
                                                     binary=False, encoding="utf8",
                                                     unicode_errors='ignore')  # C text format
    print("word2vec load succeed")

    # 所有文本构建词汇表，words_cut 为分词后的list，每个元素为以空格分隔的str.
    # vocabulary = list(set([word for item in words_cut for word in item.split()]))

    # 构建词汇-向量字典
    vocabulary_vector = {}
    for word in vocabulary:
        if word in wv_from_text:
            vocabulary_vector[word] = wv_from_text[word]
    # 储存词汇-向量字典，由于json文件不能很好的保存numpy词向量，故使用csv保存
    pd.DataFrame(vocabulary_vector).to_csv("./word_vector/vocabulary_vector.csv")

def load_dic(vec_data):
    vocabulary_vector = dict(vec_data)
    # 此时需要将字典中的词向量np.array型数据还原为原始类型，方便以后使用
    for key, value in vocabulary_vector.items():
        vocabulary_vector[key] = np.array(value)
    print("vocabulary vector load succeed")
    # 至此可以使用字典方式快速读取词向量，第一次构建耗时，之后只需读取该csv，速度提升很多啦..
    return vocabulary_vector

if __name__ == "__main__":
    # 先取出所有可能用到的词
    vocabulary = get_vocabulary()

    # 将词向量存储为字典
    get_vocabulary_vector(vocabulary)
    # # 读取词汇-向量字典，csv转字典

    vec_data = pd.read_csv("./word_vector/vocabulary_vector.csv")
    # 获取词典对照
    vocabulary_vector = load_dic(vec_data)
    # df = pd.read_excel('./data/all_clear_data.xlsx')
    df = pd.read_csv('./data/selected_newsData_split.csv')
    df['str_spl'] = df.splitSummary.str.split(',')
    sentence_vec = np.zeros((df.shape[0], 300))
    for index, rows in df.iterrows():
        count = 0
        for word in rows['str_spl']:
            word_vec = vocabulary_vector.get(word)
            if word_vec is not None:
                sentence_vec[index] += word_vec
                count += 1
        print(index)
        print(sentence_vec[index])
        sentence_vec[index] = sentence_vec[index] / count
        print(count)
        print(sentence_vec[index])
    result = pd.DataFrame(sentence_vec)
    result['label'] = df['sentiment']
    result.to_csv("./word_vector/selected_sentence_vector.csv")
    print("complete")

