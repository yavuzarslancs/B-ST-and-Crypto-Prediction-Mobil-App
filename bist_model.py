import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import joblib

def fetch_stock_data(stock_symbol, start_date):
    df = yf.Ticker(stock_symbol).history(start=start_date, period="max")
    df['Date'] = df.index
    df = df[['Date'] + [col for col in df.columns if col != 'Date']]
    df.reset_index(drop=True, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[['Date', 'Close']]

def get_user_stock_symbol():
    stock_symbol = input("Lütfen bir hisse senedi sembolü girin (Örnek: XU100.IS): ")
    return stock_symbol

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

def build_model(lookback):
    model = Sequential()
    model.add(LSTM(units=50, activation="relu", return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def train_model(model, X_train, y_train, X_test, y_test, callbacks, epochs, batch_size):
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        shuffle=False
    )
    return history

def plot_loss(history, checkpoint_epoch):
    plt.figure(figsize=(14, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.axvline(checkpoint_epoch, color='r', linestyle='--', label='Checkpoint')
    plt.legend(loc="upper right")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.ylim([0, max(plt.ylim())])
    plt.title("Training and Validation Loss", fontsize=16)
    plt.show()

def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test, batch_size=20)
    print("\nTest loss:%.1f%%" % (100.0 * loss))

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

def make_predictions(model, input_data, scaler):
    predicted_price = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

def main():
    warnings.filterwarnings("ignore")
    
    print("1. Model Eğit")
    print("2. Tahmin Yap")
    choice = input("Lütfen bir seçenek girin (1 veya 2): ")
    
    if choice == '1':
        start_date = "2010-07-27"
        stock_symbol = get_user_stock_symbol()

        df = fetch_stock_data(stock_symbol, start_date)

        train, test, position = split_data(df, 0.25)

        train_close = train[['Close']].values
        test_close = test[['Close']].values

        scaler, train_scaled, test_scaled = scale_data(train_close, test_close)

        # scaler'ı kaydediyoruz
        joblib.dump(scaler, "bist_scaler.pkl")

        lookback = 25
        X_train, y_train = create_features(train_scaled, lookback)
        X_test, y_test = create_features(test_scaled, lookback)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        model = build_model(lookback)

        checkpoint_filepath = "best_bist_model_v1.h5"
        checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, verbose=1)
        callbacks = [checkpoint]

        history = train_model(model, X_train, y_train, X_test, y_test, callbacks, epochs=100, batch_size=20)

        checkpoint_epoch = np.argmin(history.history["val_loss"])

        plot_loss(history, checkpoint_epoch)
        evaluate_model(model, X_test, y_test)

        best_model = tf.keras.models.load_model(checkpoint_filepath)

        input_data = df["Close"].values[-lookback:]
        input_data = scaler.transform(input_data.reshape(-1, 1))
        input_data = input_data.reshape(1, lookback, 1)

        predicted_price = make_predictions(best_model, input_data, scaler)
        print(f"{datetime.now()} için tahmin edilen kapanış fiyatı:", predicted_price)

    elif choice == '2':
        stock_symbol = get_user_stock_symbol()
        model = tf.keras.models.load_model("best_bist_model_v1.h5")
        scaler = joblib.load("bist_scaler.pkl")

        df = fetch_stock_data(stock_symbol, "2010-07-27")

        lookback = 25
        input_data = df["Close"].values[-lookback:]
        input_data = scaler.transform(input_data.reshape(-1, 1))
        input_data = input_data.reshape(1, lookback, 1)

        predicted_price = make_predictions(model, input_data, scaler)
        print(f"{datetime.now()} için tahmin edilen kapanış fiyatı:", predicted_price)

if __name__ == "__main__":
    main()
