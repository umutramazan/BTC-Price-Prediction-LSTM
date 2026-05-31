import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Input

veriler = pd.read_csv("BTC-USD.csv")

veriler['Date'] = pd.to_datetime(veriler['Date'])
series = veriler['Adj Close']

veriler.set_index('Date', inplace=True)

training_data = veriler['Adj Close']['2014-09-18':'2020-12-31']
validation_data = veriler['Adj Close']['2021-01-01':'2024-01-01']

training_set = training_data.values.reshape(-1, 1)
validation_set = validation_data.values.reshape(-1, 1)

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
validation_set_scaled = sc.transform(validation_set)

def create_sequences(data, seq_length=60):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(training_set_scaled)
X_validation, y_validation = create_sequences(validation_set_scaled)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

model = keras.Sequential([
    Input(shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(units=50, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=50, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=50),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

btc_test = veriler[['Adj Close']]['2024-01-01':'2024-05-11']
real_stock_price = btc_test['Adj Close'].values.reshape(-1, 1)

dataset_total = pd.concat((veriler['Adj Close'], btc_test['Adj Close']), axis=0)
inputs = dataset_total[len(dataset_total) - len(btc_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.figure(figsize=(12, 6))
plt.plot(btc_test.index, real_stock_price, color='red', label='Real BTC Price')
plt.plot(btc_test.index, predicted_stock_price, color='blue', label='Predicted BTC Price')
plt.title('BTC Price Prediction')
plt.xlabel('Date')
plt.ylabel('BTC Price (USD)')
plt.legend()
plt.xticks(rotation=45)  
plt.tight_layout()
plt.show()

predicted_prices = series.tolist()

for i in range(30):
    last_known_price = predicted_prices[-len(series):]
    last_known_price_scaled = sc.transform(np.array(last_known_price).reshape(-1, 1))
    predicted_price_scaled = model.predict(last_known_price_scaled.reshape(1, len(series), 1)) 
    predicted_price = sc.inverse_transform(predicted_price_scaled) 
    print(f"Tahmin {i+1}. gün: {predicted_price[0][0]}")
    
    predicted_prices.append(predicted_price[0][0])  
    
    series = np.append(series, predicted_price[0][0])  

    X_new, y_new = create_sequences(sc.transform(series[-len(series):].reshape(-1, 1)))
    X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))
    model.fit(X_new, y_new, epochs=1, batch_size=32, verbose=0) 
    
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices[-60:], label='Son 30 Gün Tahminleri', color='red') 
plt.plot(predicted_prices[-60:-30], label='Son 60-30', color='blue')  
plt.xlabel('Günler')
plt.ylabel('BTC Fiyatı (USD)')
plt.title('Son 30 Günlük BTC Fiyat Tahmini')
plt.legend()
plt.show()

plt.figure(figsize=(24, 12))
plt.plot(predicted_prices[:-30], label='Tüm Tahmin', color='blue')
plt.plot(range(len(predicted_prices)-30, len(predicted_prices)), predicted_prices[-30:], label='Son 30 Gün Tahminleri', color='red')  # Son 30 günü kırmızıyla çiz
plt.xlabel('Günler')
plt.ylabel('BTC Fiyatı (USD)')
plt.title('Son 30 Günlük BTC Fiyat Tahmini')
plt.legend()
plt.show()