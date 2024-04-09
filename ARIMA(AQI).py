import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from pmdarima import auto_arima

df=pd.read_excel('clean_data/AQI.xlsx')
df['date'] = pd.to_datetime(df['date'])
df=df.set_index('date')
df=df[['AQI']].dropna()
df.index = pd.date_range(start='2015-01-01',periods=len(df),freq='D')
colors = ['r', 'g', 'b', 'm']
# 定 义ARIMA模 型 参 数
p = 2
d = 1
q = 2
# 进 行3 步、 5 步、 7步 和12步 预 测 并 计 算RMSE
steps = [3,5,7,12]
errors = []
i=0
# model = auto_arima(df, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, trace=True)
for s in steps:
# 拟 合ARIMA模 型
# 划 分 训 练 集 和 测 试 集
    train = df.iloc[:-12]
    test = df.iloc[-12:]
# 进 行 多 步 预 测
    forecast = [0]*s
    predictions = []
    for i in range(1,s+1):
        model = ARIMA(train , order=(p, d, q))
        model_fit = model.fit()
        forecast[i-1] = model_fit.forecast().iloc[0]
        predictions.append(forecast[i-1])
# 将 预 测 值 添 加 到 训 练 集 中
        train = pd.concat([train , pd.DataFrame(forecast[i-1], columns=["AQI"],
        index=[train.index[-1]+pd.DateOffset(days=1)])])
# 计 算RMSE
    error = np.sqrt(np.mean((predictions - test[:s].values.reshape(-1)) ** 2))
    errors.append(error)
    plt.plot(pd.date_range(test.index[0], periods=s, freq=test.index.freq),
    predictions[:s], label=f'{s}-step Forecast (RMSE={error:.3f})')
    plt.plot(pd.date_range(test.index[0], periods=s, freq=test.index.freq), test[:s
    ], label=f'{s}-step Actual (RMSE={error:.3f})')
# 可 视 化 预 测 结 果
plt.legend()
plt.show()

