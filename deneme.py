import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import joblib

# Kullanıcı arayüzü için menü fonksiyonu
def display_menu():
    print("\nMerhaba hoş geldiniz. Aşağıdakilerden hangi işlemi yapmak istiyorsunuz?")
    print("1) Model Eğit")
    print("2) Tahmin Et")
    print("3) Kripto Para Grafik")
    print("4) Kripto Para Destek ve Direnç Gösteren Grafik")
    print("5) Kripto Para Hareketli Ortalamalar Gösteren Grafik")
    print("0) Çıkış Yap")
    choice = int(input("Seçiminizi yapınız (1, 2, 3, 4, 5, 0): "))
    return choice

# Kripto para grafik gösterme (kullanıcı etkileşimli)
def plot_crypto_chart_interactive(df):
    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Kapanış Fiyatları'))
    fig.update_layout(
        title="Kripto Para Fiyat Grafiği",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD)",
        xaxis_rangeslider_visible=True,
        dragmode='pan',  # Grafiği sağa sola kaydırma modu
    )
    
    # Adding drawing tools
    fig.update_layout(
        newshape=dict(line_color='cyan'),
        modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
    )
    
    fig.show()

# Destek ve direnç seviyelerini gösteren grafik
def plot_support_resistance(df):
    df = df.reset_index(drop=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Kapanış Fiyatları'))

    def calculate_support_resistance(df, window=5):
        levels = []
        for i in range(window, len(df) - window):
            high_range = df['Close'][i-window:i+window]
            low_range = df['Close'][i-window:i+window]
            if df['Close'][i] == high_range.max():
                levels.append((i, df['Close'][i], 'resistance'))
            elif df['Close'][i] == low_range.min():
                levels.append((i, df['Close'][i], 'support'))
        return levels

    def filter_levels(levels, tolerance=0.02):
        filtered_levels = []
        for level in levels:
            if not any(abs(level[1]-x[1])/x[1] < tolerance for x in filtered_levels):
                filtered_levels.append(level)
        return filtered_levels

    levels = calculate_support_resistance(df)
    levels = filter_levels(levels)

    for level in levels:
        if level[2] == 'support':
            fig.add_hline(y=level[1], line_color='green', line_dash="dash")
        elif level[2] == 'resistance':
            fig.add_hline(y=level[1], line_color='red', line_dash="dash")

    fig.update_layout(
        title="Destek ve Direnç Seviyeleri",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD)",
        xaxis_rangeslider_visible=True,
        dragmode='pan',  # Grafiği sağa sola kaydırma modu
    )
    fig.show()

# Hareketli ortalamaları gösteren grafik
def plot_moving_averages(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Kapanış Fiyatları'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='20 Günlük Ortalama'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], mode='lines', name='50 Günlük Ortalama'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA100'], mode='lines', name='100 Günlük Ortalama'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], mode='lines', name='200 Günlük Ortalama'))

    fig.update_layout(
        title="Hareketli Ortalamalar",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD)",
        xaxis_rangeslider_visible=True,
        dragmode='pan',  # Grafiği sağa sola kaydırma modu
    )
    fig.show()

def fetch_crypto_data(crypto_symbol, start_date):
    df = yf.download(crypto_symbol, start=start_date)
    df['Date'] = df.index
    df = df[['Date', 'Close']]
    df.reset_index(drop=True, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def split_data(dataframe, test_size):
    position = int(round(len(dataframe) * (1 - test_size)))
    train = dataframe[:position]
    test = dataframe[position:]
    return train, test, position

def scale_data(train_data, test_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return scaler, train_scaled, test_scaled

def create_features(data, lookback):
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# ModelCheckpoint callback içinde en iyi epoch'u kaydetmek için
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best_epoch = epoch
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)

def train_model(model, X_train, y_train, X_test, y_test, callbacks, epochs, batch_size):
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        shuffle=False
    )
    return history

def plot_loss(history, best_epoch):
    plt.figure(figsize=(14, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Checkpoint')
    plt.legend(loc="upper right")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title("Training and Validation Loss", fontsize=16)
    plt.show()

def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test, batch_size=20)
    print("\nTest loss: %.4f" % loss)

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

def make_predictions(model, input_data, scaler):
    predicted_price = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

def main():
    warnings.filterwarnings("ignore")

    while True:
        choice = display_menu()
        if choice == 0:
            print("Çıkış yapılıyor...")
            break

        start_date = "2018-01-01"
        crypto_symbol = input("Lütfen bir kripto para sembolü giriniz (örn., BTC-USD): ")
        df = fetch_crypto_data(crypto_symbol, start_date)
        
        if df.empty:
            print("Geçersiz kripto para sembolü veya veri çekme hatası. Lütfen tekrar deneyin.")
            continue

        if choice == 1:
            train, test, position = split_data(df, 0.25)

            train_close = train[['Close']].values
            test_close = test[['Close']].values

            scaler, train_scaled, test_scaled = scale_data(train_close, test_close)

            # Scaler'ı kaydet
            joblib.dump(scaler, 'scaler9.pkl')

            lookback = 25
            X_train, y_train = create_features(train_scaled, lookback)
            X_test, y_test = create_features(test_scaled, lookback)

            X_train = np.reshape(X_train, (X_train.shape[0], lookback, 1))
            X_test = np.reshape(X_test, (X_test.shape[0], lookback, 1))

            model = build_model((lookback, 1))

            callbacks = [
                CustomModelCheckpoint(filepath="best_crypto_model_v9.h5", monitor="val_loss", mode="min",
                                    save_best_only=True, save_weights_only=False, verbose=1)
            ]

            history = train_model(model, X_train, y_train, X_test, y_test, callbacks, epochs=100, batch_size=20)

            # Checkpointin gerçekleştiği epoch'u al
            best_epoch = callbacks[0].best_epoch

            # Kaybı çiz ve checkpoint epochunu göster
            plot_loss(history, best_epoch)

            # En iyi modeli yükle
            best_model = load_model("best_crypto_model_v9.h5")
            evaluate_model(best_model, X_test, y_test)

            train_predict = best_model.predict(X_train)
            test_predict = best_model.predict(X_test)

            train_predict = inverse_transform(scaler, train_predict)
            test_predict = inverse_transform(scaler, test_predict)

            y_train = inverse_transform(scaler, y_train.reshape(-1, 1))
            y_test = inverse_transform(scaler, y_test.reshape(-1, 1))

            train_prediction_df = df[lookback:position]
            train_prediction_df["Predicted"] = train_predict

            test_prediction_df = df[position + lookback:]
            test_prediction_df["Predicted"] = test_predict

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Real Values'))
            fig.add_trace(go.Scatter(x=train_prediction_df['Date'], y=train_prediction_df["Predicted"], mode='lines', name='Train Predicted', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_prediction_df['Date'], y=test_prediction_df["Predicted"], mode='lines', name='Test Predicted', line=dict(color='red')))
            fig.update_layout(
                title="Kripto Para Fiyat Tahmini",
                xaxis_title="Tarih",
                yaxis_title="Fiyat (USD)",
                xaxis_rangeslider_visible=True,
                dragmode='pan',
            )
            fig.show()

            # Tahmin yap ve sonucu göster
            input_data = df["Close"].values[-lookback:]
            input_data = scaler.transform(input_data.reshape(-1, 1))
            input_data = input_data.reshape(1, lookback, 1)

            predicted_price = make_predictions(best_model, input_data, scaler)
            print(f"Predicted closing price for {datetime.now().strftime('%Y-%m-%d')}: {predicted_price}")


        elif choice == 2:
            lookback = 25
            input_data = df["Close"].values[-lookback:]
            scaler = joblib.load('scaler9.pkl')
            input_data_scaled = scaler.transform(input_data.reshape(-1, 1))
            input_data_scaled = input_data_scaled.reshape(1, lookback, 1)

            best_model = load_model("best_crypto_model_v9.h5")
            predicted_price = make_predictions(best_model, input_data_scaled, scaler)
            print(f"Predicted closing price for {datetime.now().strftime('%Y-%m-%d')}: {predicted_price}")

        elif choice == 3:
            # Kripto Para Grafik (kullanıcı etkileşimli)
            plot_crypto_chart_interactive(df)
        elif choice == 4:
            # Kripto Para Destek ve Direnç Gösteren Grafik
            plot_support_resistance(df)
        elif choice == 5:
            # Kripto Para Hareketli Ortalamalar Gösteren Grafik
            plot_moving_averages(df)
        else:
            print("Geçersiz seçim yaptınız. Lütfen tekrar deneyin.")

if __name__ == "__main__":
    main()
