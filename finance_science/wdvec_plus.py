import pandas
import gensim
from gensim.models import word2vec as wv
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

def method():
    print("中文测试")
    df = pandas.read_excel('./data/all_clear_data.xlsx')


    dic = './word_vector/sgns.financial.word.bz2'
    model = gensim.models.KeyedVectors.load_word2vec_format(dic, encoding="utf-8")
    vocab = model.index_to_key
    print(vocab[:5])
    word = '欧几里得'
    vec = model.vectors[model.key_to_index [word]]

    print('词向量长度：', vec.shape)
    print('词向量：\n', vec)

    return df

if __name__ == '__main__':
    method()
    print()