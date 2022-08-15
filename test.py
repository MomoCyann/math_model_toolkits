import pandas as pd
path = 'finance_science/word_vector/sentence_vector.csv'
data = pd.read_csv(path)
print(data.isnull().sum())
data = data.fillna(0)
print("fillna~")
print(data.isnull().sum())
input()
data.to_csv('finance_science/word_vector/sentence_vector.csv')