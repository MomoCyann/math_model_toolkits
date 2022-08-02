import pandas as pd

data = pd.read_csv('BERT/embedding_title_label.csv')

data.reset_index(inplace=True, drop=True)
data.index = data.index + 1
data = data.iloc[:, 1:]

data.to_csv('BERT/embedding_title_label_v2.csv')

print('s')