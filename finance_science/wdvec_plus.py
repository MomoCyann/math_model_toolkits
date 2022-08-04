import pandas
import gensim


def method():
    df = pandas.read_excel('/data/all_clear_data.xlsx')

    fd = '/word_vector/sgns.financial.word.bz2'
    model = gensim.models.KeyedVectors.load_word2vec_format(fd, encoding="utf-8")
    vocab = model.index2entity
    print(vocab[:5])
    word = '欧几里得'
    vec = model.wv.vectors[model.wv.vocab[word].index]

    print('词向量长度：', vec.shape)
    print('词向量：\n', vec)

    return df

if __name__ == '__main__':
    print()