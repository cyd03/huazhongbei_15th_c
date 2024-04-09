import pandas as pd
import numpy as np

from scipy.interpolate import PchipInterpolator

df =pd.read_excel('data/附件一污染物浓度.xlsx')


def set_null(data):
	data_columns=data.columns
	for i in data_columns:
		data[data[i]==0]=None
	return data


def interp_column(s):
	known_x=s[s.notnull()].index
	knon_y=s[s.notnull()].values

	f=PchipInterpolator(known_x,knon_y)
	return pd.Series(f(s.index).round(1))
def set_quality(data):
	quality=[]
	AQI=data['AQI'].tolist()
	for i in AQI:
		if i>=0 and i<=50:
			quality.append('优')
		elif i>=51 and i<=100:
			quality.append('良')
		elif i>=101 and i<=150:
			quality.append('轻度污染')
		elif i>=151 and i<=200:
			quality.append('中度污染')
		elif i>=201 and i<=300:
			quality.append('重度污染')
		else:
			quality.append('严重污染')
	quality=np.array(quality).transpose()
	return quality
	pass

set_null(df)
df_drop=df.drop(labels='质量等级',axis=1)
df_drop = df_drop.drop(labels='date',axis=1)
df_interpolated =df_drop.apply(interp_column,axis=0)
df_interpolated['质量等级'] = df['质量等级']
df_interpolated['date'] = df['date']
quality=set_quality(df_interpolated)
df_interpolated['质量等级'] = quality
df_interpolated.to_excel('clean_data/附件一污染物浓度(clean).xlsx',index=False)
