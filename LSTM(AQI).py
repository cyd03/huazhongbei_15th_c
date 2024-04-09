# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
# date	AQI	PM2.5	O3	PM10	NO2	V10004_700	V13003_700

# 读取数据
df = pd.read_excel('clean_data/AQI.xlsx')
df['date'] = pd.to_datetime(df['date'])
df=df.set_index('date')
df.index = pd.date_range(start='2015-01-01',periods=len(df),freq='D')
# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['AQI','PM2.5','O3','PM10','NO2','V10004_700','V13003_700']])

def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, :])
        y.append(data[i+time_steps])  # 预测PM2.5
    return np.array(X), np.array(y)

time_steps = 50
X, y = create_dataset(scaled_data, time_steps)

# 划分训练集和测试集
train_size = int(len(X) * 0.99)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
y_test_original = df['AQI'].iloc[train_size+time_steps:]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=7))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)

# 预测
y_pred_normalized = model.predict(X_test)

# 逆标准化预测值
y_pred = scaler.inverse_transform(y_pred_normalized)[:,0]
# 可视化
import matplotlib.pyplot as plt

plt.plot(y_test_original.index, y_test_original, label='Actual AQI')
plt.plot(y_test_original.index, y_pred, label='Predicted AQI')
plt.legend()
plt.show()
