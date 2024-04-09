import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
# 读取数据
df = pd.read_excel('clean_data/PM2.5.xlsx')
df['date'] = pd.to_datetime(df['date'])
df=df.set_index('date')
df.index = pd.date_range(start='2015-01-01',periods=len(df),freq='D')
# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['PM2.5', 'PM10', 'AQI', 'CO', 'V12001_700', 'NO2', 'O3']])

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
y_test_original = df['PM2.5'].iloc[train_size+time_steps:]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=7))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)

# 预测
y_pred_normalized = model.predict(X_test)
new_data = scaled_data[train_size:]
last_data = new_data[-time_steps:]
predictions = []
for i in range(12):
    y_pred_next=model.predict(np.array([last_data[-time_steps:]]))
    y_pred_next1 = scaler.inverse_transform(y_pred_next)
    predictions.append(y_pred_next1[:,0])
    last_data = np.vstack((last_data,y_pred_next))
predictions = np.array(predictions).reshape(1,-1)
plt.plot(pd.date_range(start='2023-4-30',periods=len(predictions[0]),freq='D'),predictions[0])
plt.show()