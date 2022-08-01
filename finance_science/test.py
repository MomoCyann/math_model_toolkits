import glob

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

f= glob.glob('clear_data/*.csv')
print(f)

df=pd.read_csv('clear_data/宁德时代1-12.csv')
print(df['sentiment'].value_counts())

df=pd.read_csv('split_data/宁德时代1-12.csv')
print(df['sentiment'].value_counts())

df=pd.read_csv('split_data/宁德时代1-12_remove0.csv')
print(df['sentiment'].value_counts())