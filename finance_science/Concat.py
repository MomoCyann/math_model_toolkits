import pandas as pd
import glob

files=glob.glob('clear_data/*.csv')


df =pd.read_csv(files[0])
for f in files[1:]:
    df_temp =pd.read_csv(f)
    df=pd.concat([df,df_temp],axis=0)



df=df.loc[:,~df.columns.str.contains('Unnamed')]
print(df.info())
print(df.head())
df.to_csv('clear_data/all_data.csv')