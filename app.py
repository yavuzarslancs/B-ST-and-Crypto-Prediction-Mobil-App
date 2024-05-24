from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import logging
import joblib

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

# Modelleri ve scaler'ları yükleyin
crypto_model = load_model('best_crypto_model_v9.h5')
bist_model = load_model('best_bist_model_v1.h5')
crypto_scaler = joblib.load('scaler9.pkl')
# BIST için scaler'ı yükleyin, eğer varsa
# bist_scaler = joblib.load('bist_scaler.pkl')

def fetch_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    df['Date'] = df.index
    df = df[['Date', 'Close']]
    df.reset_index(drop=True, inplace=True)
    df.loc[:, "Date"] = pd.to_datetime(df["Date"])
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        symbol = data['symbol']
        is_crypto = data['is_crypto']
        lookback = 25
        start_date = "2018-01-01"
        end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

        df = fetch_data(symbol, start_date, end_date)
        logging.debug(f"Fetched data for {symbol}: {df.tail()}")  # Son verileri kontrol edin

        input_data = df["Close"].values[-lookback:]
        input_data = input_data.reshape(-1, 1)

        if is_crypto:
            input_data = crypto_scaler.transform(input_data)
            model = crypto_model
        else:
            # BIST için scaler'ı kullanın, eğer varsa
            input_data = crypto_scaler.transform(input_data)  # bist_scaler.transform(input_data) olarak değiştirin, eğer bist_scaler varsa
            model = bist_model

        input_data = input_data.reshape(1, lookback, 1)
        predicted_price = model.predict(input_data)
        predicted_price = crypto_scaler.inverse_transform(predicted_price)[0][0]  # bist_scaler.inverse_transform(predicted_price) olarak değiştirin, eğer bist_scaler varsa
        logging.debug(f"Predicted price: {predicted_price}")

        return jsonify({'predicted_price': float(predicted_price)})
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/plot_crypto_chart_interactive', methods=['POST'])
def plot_crypto_chart_interactive():
    data = request.get_json()
    symbol = data['symbol']
    start_date = "2018-01-01"
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
    df = fetch_data(symbol, start_date, end_date)
    
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    prices = df['Close'].tolist()

    return jsonify({'dates': dates, 'prices': prices})

@app.route('/plot_support_resistance', methods=['POST'])
def plot_support_resistance():
    data = request.get_json()
    symbol = data['symbol']
    start_date = "2018-01-01"
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
    df = fetch_data(symbol, start_date, end_date)
    df = df.reset_index(drop=True)
    
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

    support_levels = [level[1] for level in levels if level[2] == 'support']
    resistance_levels = [level[1] for level in levels if level[2] == 'resistance']
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    prices = df['Close'].tolist()

    return jsonify({'dates': dates, 'prices': prices, 'support_levels': support_levels, 'resistance_levels': resistance_levels})

@app.route('/plot_moving_averages', methods=['POST'])
def plot_moving_averages():
    try:
        data = request.get_json()
        symbol = data['symbol']
        start_date = "2018-01-01"
        end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

        df = fetch_data(symbol, start_date, end_date)

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        response_data = {
            'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].tolist(),
            'ma20': df['MA20'].tolist(),
            'ma50': df['MA50'].tolist(),
            'ma100': df['MA100'].tolist(),
            'ma200': df['MA200'].tolist(),
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
