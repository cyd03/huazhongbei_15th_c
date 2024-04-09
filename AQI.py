import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

data=pd.read_excel('clean_data/指标综合.xlsx')
corr_matrix = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix,cmap='coolwarm',annot=True)
plt.title('Correlation distribution plot')
plt.show()

X=data[['PM10', 'O3', 'SO2','PM2.5','NO2', 'CO', 'V13305',
       'V10004_700', 'V11291_700', 'V12001_700', 'V13003_700']]
y=data['AQI']
rf=RandomForestRegressor(n_estimators=200,random_state=42)
rf.fit(X,y)


feature_importance=rf.feature_importances_
feature_names=X.columns
sorted_idx=feature_importance.argsort()

plt.barh(range(len(sorted_idx)),feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)),feature_names[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()



