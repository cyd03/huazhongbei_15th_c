import pandas as pd
# 读取数据
df = pd.read_excel('data/附件1：污染物浓度数据.xlsx')
print(df)
df['date'] = df['年'].map(str)+"/"+df['月'].map(str)+"/"+df['日'].map(str)
pd.to_datetime(df['date'])
#打印查看效果
print(df['date'])
data2 = pd.read_excel('data/附件2：气象数据.xlsx')
data2['date'] = data2['V04001'].map(str)+"/"+data2['V04002'].map(str)+"/"+data2['V04003'].map(str)
pd.to_datetime(data2['date'])
 #将结果输出在另一个 csv 文件中
df=df.drop(labels='年',axis=1)
df=df.drop(labels='月',axis=1)
df=df.drop(labels='日',axis=1)
filename='data/附件一污染物浓度.csv'
df.to_csv(filename,encoding='gb2312',index=False)
 #将结果输出在另一个 csv 文件中
data2=data2.drop(labels='V04001',axis=1)
data2=data2.drop(labels='V04002',axis=1)
data2=data2.drop(labels='V04003',axis=1)
filename='data/附件二气象数据.csv'
data2.to_csv(filename,encoding='gb2312',index=False)