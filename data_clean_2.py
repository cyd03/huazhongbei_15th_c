import pandas as pd
import numpy as np

from scipy.interpolate import PchipInterpolator

def interp_column(s):
	known_x=s[s.notnull()].index
	knon_y=s[s.notnull()].values

	f=PchipInterpolator(known_x,knon_y)
	return pd.Series(f(s.index).round(1))

df=pd.read_excel('data/附件二气象数据.xlsx')
df_drop=df.drop(labels='date',axis=1)
deviation = np.abs(df_drop-df_drop.mean())
std = df_drop.std()
outlier_mask=deviation>3*std
df_drop[outlier_mask]=np.nan
df_interpolated =df_drop.apply(interp_column,axis=0)
df_interpolated['date']=df['date']
df_interpolated.to_excel('clean_data/附件二气象数据(clean).xlsx',index=False)









