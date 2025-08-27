# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, LearningRateScheduler
import requests
from io import StringIO, BytesIO
import base64
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData # Import FundamentalData for overview
from datetime import timedelta, datetime
import os
import pickle
from keras.models import load_model
import hashlib

def get_model_paths(stock_symbol):
    model_dir = os.path.join("saved_models", stock_symbol)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    return model_path, scaler_path

def get_forecast_cache_path(symbol, forecast_options):
    key = f"{symbol}_{forecast_options}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    path = os.path.join("saved_models", symbol, f"forecast_{hash_key}.pkl")
    return path

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": "http://127.0.0.1:5502"},
    r"/forecast": {"origins": "http://127.0.0.1:5502"},
    r"/get_stock_quote": {"origins": "http://127.0.0.1:5502"},
    r"/get_historical_data": {"origins": "http://127.0.0.1:5502"},
    r"/get_custom_history_plot": {"origins": "http://127.0.0.1:5502"}
})
# Alpha Vantage API key (Replace with your actual key)
api_key = '3RG3EUZR3KJCFF6L'

# --- Utility Functions (Adapted from aa.py) ---
def get_usd_to_inr():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            rate = data['rates'].get('INR', None)
            if rate:
                print(f"Live USD to INR rate: ₹{rate}")
                return rate
            else:
                print("INR rate not found in response.")
        else:
            print("Error fetching exchange rate.")
    except Exception as e:
        print(f"Exception occurred while fetching USD to INR: {e}")
    return 83.5  # fallback

def fetch_listing_status(api_key):
    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            return data[['symbol', 'name']]
        else:
            print(f"Error fetching listing status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching listing status: {e}")
        return None

def get_stock_symbol(company_name, listings_df):
    match = listings_df[listings_df['name'].str.contains(company_name, case=False, na=False)]
    if not match.empty:
        stock_symbol = match.iloc[0]['symbol']
        print(f"Company '{company_name}' corresponds to stock symbol: {stock_symbol}")
        return stock_symbol
    else:
        print(f"No stock symbol found for '{company_name}'.")
        return None

def create_sequences(data, sequence_len):
    X, y = [], []
    for i in range(sequence_len, len(data)):
        X.append(data[i - sequence_len:i, :])
        y.append(data[i, 0])  # Predicting 'Close'
    return np.array(X), np.array(y)

def lr_scheduler(epoch, lr):
    return max(0.0001, lr * 0.9)

def rescale_predictions(preds, scaler, num_features):
    # Create a dummy array with the correct number of features for inverse_transform
    padded = np.concatenate([preds, np.zeros((preds.shape[0], num_features - 1))], axis=1)
    return scaler.inverse_transform(padded)[:, 0]

def plot_to_base64(plt):
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

# --- Main Prediction Logic ---
def run_prediction(company_name):
    plots = {}
    metrics = {}

    listings_df = fetch_listing_status(api_key)
    if listings_df is None:
        return {"error": "Could not fetch stock listings. Please check API key or network."}

    stock_symbol = get_stock_symbol(company_name, listings_df)
    model_path, scaler_path = get_model_paths(stock_symbol)

    # Check if cached model exists
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"✅ Loading cached model for {stock_symbol}")
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        skip_training = True
    else:
        skip_training = False

    if stock_symbol == None:
        return {"error": f"No stock symbol found for '{company_name}'. Please check the company name."}

    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')
        if data.empty:
            return {"error": f"No historical data found for {stock_symbol}."}
        data = data[::-1]
        data = data[['4. close', '5. volume', '1. open', '2. high', '3. low']]
        data.columns = ['Close', 'Volume', 'Open', 'High', 'Low']
    except Exception as e:
        return {"error": f"Error fetching stock data for {stock_symbol}: {e}"}

    # Feature engineering
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['52_Week_High'] = data['Close'].rolling(window=252).max()
    data['52_Week_Low'] = data['Close'].rolling(window=252).min()
    data['4_Week_High'] = data['High'].rolling(window=20).max()
    data['4_Week_Low'] = data['Low'].rolling(window=20).min()
    data['Prev_Close'] = data['Close'].shift(1)
    data['Next_Open'] = data['Open'].shift(-1)
    data['Trade_Price'] = (data['High'] + data['Low']) / 2
    data.dropna(inplace=True)

    if data.empty:
        return {"error": "Not enough data after feature engineering. Please try a different company or check data availability."}

    features = ['Close', 'Volume', 'SMA_20', '52_Week_High', '52_Week_Low',
                '4_Week_High', '4_Week_Low', 'Prev_Close', 'Next_Open', 'Trade_Price']
    
    if len(data) < 252: # Ensure enough data for 52-week features
         return {"error": "Not enough historical data for robust prediction. At least 1 year of data is recommended."}


    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    num_features = scaled_data.shape[1] # Get actual number of features

    sequence_len = 200
    if len(scaled_data) < sequence_len + 1:
        return {"error": "Not enough data to create sequences for LSTM training."}

    X, y = create_sequences(scaled_data, sequence_len)
    
    # Ensure there's enough data for train/test split after sequence creation
    if len(X) < 2: # At least one sample for train, one for test
        return {"error": "Insufficient data to create meaningful training and test sets after sequence generation."}

    train_len = int(len(X) * 0.8)
    if train_len == 0: # Ensure train_len is at least 1
        train_len = 1
    
    # Ensure test_len is at least 1
    if len(X) - train_len == 0:
        return {"error": "Insufficient data for a test set. Please provide more historical data."}


    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return {"error": "Not enough data for training or testing after splitting. Adjust sequence length or get more data."}

    if not skip_training:
        # LSTM model
        model = Sequential()
        model.add(LSTM(units=512, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.01))
        model.add(LSTM(units=256, return_sequences=False))
        model.add(Dropout(0.01))
        model.add(Dense(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_absolute_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_schedule = LearningRateScheduler(lr_scheduler)

        # Train the model
        try:
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                batch_size=64,
                epochs=10, # Reduced epochs for quicker response in web app
                callbacks=[early_stopping, lr_schedule],
                verbose=2 # Suppress training output
            )
        except Exception as e:
            return {"error": f"Error during model training: {e}"}
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Reverse scaling
    train_pred_rescaled = rescale_predictions(train_pred, scaler, num_features)
    test_pred_rescaled = rescale_predictions(test_pred, scaler, num_features)
    y_train_rescaled = rescale_predictions(y_train.reshape(-1, 1), scaler, num_features)
    y_test_rescaled = rescale_predictions(y_test.reshape(-1, 1), scaler, num_features)

    # Currency conversion: USD to INR (Live)
    usd_to_inr = get_usd_to_inr()
    y_train_rescaled *= usd_to_inr
    y_test_rescaled *= usd_to_inr
    train_pred_rescaled *= usd_to_inr
    test_pred_rescaled *= usd_to_inr

    # --- Generate Plots ---
   # --- Generate Full Training + Testing Plot ---
    plt.figure(figsize=(14, 6))
    plt.plot(data.index[sequence_len:train_len + sequence_len], y_train_rescaled, label='Training Data', color='green')
    plt.plot(data.index[train_len + sequence_len:], y_test_rescaled, label='Actual Prices', color='red')
    plt.plot(data.index[sequence_len:train_len + sequence_len], train_pred_rescaled, label='Training Predictions', color='blue')
    plt.plot(data.index[train_len + sequence_len:], test_pred_rescaled, label='Test Predictions', color='orange')
    plt.title(f"{company_name} Stock Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Close Price (INR)')
    plt.legend()
    plots['full_plot'] = plot_to_base64(plt)

    # --- Setup for Dynamic Year/Month Plots ---
    test_data_start_idx_in_original = train_len + sequence_len

    if not data.index[test_data_start_idx_in_original:].empty:
        last_date_in_test_set = data.index[test_data_start_idx_in_original:].max()

        # --- Dynamic Last N Year Plots ---
        for years in range(1, 7):
            label = f"last_{years}_years_plot"
            cutoff_date = last_date_in_test_set - pd.DateOffset(years=years)
            relevant_dates = data.index[test_data_start_idx_in_original:][
                data.index[test_data_start_idx_in_original:] >= cutoff_date
            ]
            if not relevant_dates.empty:
                start_idx = data.index.get_loc(relevant_dates.min())
                rel_idx = start_idx - test_data_start_idx_in_original
                plt.figure(figsize=(14, 6))
                plt.plot(data.index[start_idx:], y_test_rescaled[rel_idx:], label='Actual Prices', color='red')
                plt.plot(data.index[start_idx:], test_pred_rescaled[rel_idx:], label='Test Predictions', color='orange')
                plt.title(f"{company_name} Actual vs Predicted (Last {years} Year{'s' if years > 1 else ''})")
                plt.xlabel('Date')
                plt.ylabel('Price (INR)')
                plt.legend()
                plots[label] = plot_to_base64(plt)
                data_key = f"last_{years}_years_data"
                plots[data_key] = {
                    "labels": data.index[start_idx:].strftime('%Y-%m-%d').tolist(),
                    "actualData": y_test_rescaled[rel_idx:].tolist(),
                    "predictedData": test_pred_rescaled[rel_idx:].tolist()
                }



            else:
                plots[label] = f"No data for last {years} year(s)."

        # --- Dynamic Last N Month Plots ---
        for months in range(1, 13):
            label = f"last_{months}_months_plot"
            cutoff_date = last_date_in_test_set - pd.DateOffset(months=months)
            relevant_dates = data.index[test_data_start_idx_in_original:][
                data.index[test_data_start_idx_in_original:] >= cutoff_date
            ]
            if not relevant_dates.empty:
                start_idx = data.index.get_loc(relevant_dates.min())
                rel_idx = start_idx - test_data_start_idx_in_original
                plt.figure(figsize=(14, 6))
                plt.plot(data.index[start_idx:], y_test_rescaled[rel_idx:], label='Actual Prices', color='red')
                plt.plot(data.index[start_idx:], test_pred_rescaled[rel_idx:], label='Test Predictions', color='orange')
                plt.title(f"{company_name} Actual vs Predicted (Last {months} Month{'s' if months > 1 else ''})")
                plt.xlabel('Date')
                plt.ylabel('Price (INR)')
                plt.legend()
                plots[label] = plot_to_base64(plt)
                data_key = f"last_{months}_months_data"
                plots[data_key] = {
                    "labels": data.index[start_idx:].strftime('%Y-%m-%d').tolist(),
                    "actualData": y_test_rescaled[rel_idx:].tolist(),
                    "predictedData": test_pred_rescaled[rel_idx:].tolist()
                }
            else:
                plots[label] = f"No data for last {months} month(s)."


    # --- Evaluation ---
    train_rmse = np.sqrt(mean_squared_error(y_train_rescaled, train_pred_rescaled))
    train_mae = mean_absolute_error(y_train_rescaled, train_pred_rescaled)
    test_rmse = np.sqrt(mean_squared_error(y_test_rescaled, test_pred_rescaled))
    test_mae = mean_absolute_error(y_test_rescaled, test_pred_rescaled)

    train_mae_percentage = (train_mae / np.mean(y_train_rescaled)) * 100 if np.mean(y_train_rescaled) != 0 else 0
    test_mae_percentage = (test_mae / np.mean(y_test_rescaled)) * 100 if np.mean(y_test_rescaled) != 0 else 0

    train_r2 = r2_score(y_train_rescaled, train_pred_rescaled)
    test_r2 = r2_score(y_test_rescaled, test_pred_rescaled)

    train_accuracy = 100 - train_mae_percentage
    test_accuracy = 100 - test_mae_percentage

    metrics = {
        "Training MAE %": f"{train_mae_percentage:.2f}%",
        "Testing MAE %": f"{test_mae_percentage:.2f}%",
        "Training R2 Score": f"{train_r2:.4f}",
        "Testing R2 Score": f"{test_r2:.4f}",
        "Training Accuracy": f"{train_accuracy:.2f}%",
        "Testing Accuracy": f"{test_accuracy:.2f}%",
        "Live USD to INR Rate": f"₹{usd_to_inr:.2f}"
    }

    # Store necessary data for future prediction
    global_prediction_data = {
        'model': model,
        'last_sequence': X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2]),
        'scaler': scaler,
        'original_df': data.copy(),
        'num_features': num_features,
        'symbol': stock_symbol
    }
    
    app.config['PREDICTION_DATA'] = global_prediction_data

    return {"plots": plots, "metrics": metrics, "company_name": company_name}


def predict_future_prices(model, last_sequence, scaler, original_df, num_features, future_type, n=None, start_date=None, end_date=None):
    if future_type == "year":
        steps = 252
        label = "Next 1 Year"
    elif future_type == "months":
        steps = int(n * 21)
        label = f"Next {n} Month(s)"
    elif future_type == "weeks":
        steps = int(n * 5)
        label = f"Next {n} Week(s)"
    elif future_type == "days":
        steps = n
        label = f"Next {n} Day(s)"
    elif future_type == "custom":
        # Generate all possible trading dates for the next year
        all_dates = [original_df.index[-1] + timedelta(days=i) for i in range(1, 366)]
        future_dates = [d for d in all_dates if d.weekday() < 5 and start_date <= d <= end_date]
        steps = len(future_dates)
        if steps == 0:
            return {"error": "No trading days found in the specified custom range.", "plot": None}
        label = f"Custom Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    else:
        return {"error": "Invalid future_type.", "plot": None}

    if steps == 0:
        return {"error": "No prediction steps generated. Check input parameters or date range.", "plot": None}

    current_sequence = last_sequence.copy()
    predictions = []

    for _ in range(steps):
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[1], current_sequence.shape[2]), verbose=0)
        predictions.append(next_pred[0, 0])
        # Update the sequence: shift left and add the new prediction for the 'Close' feature
        # Assuming 'Close' is the first feature (index 0)
        current_sequence = np.roll(current_sequence, -1, axis=1)
        # Ensure that the new prediction only replaces the 'Close' value at the last position of the sequence
        current_sequence[0, -1, 0] = next_pred[0, 0]
        # For other features, we are simply carrying over the last known values, which is a simplification
        # A more robust approach would involve predicting other features or using external data for them.

    pred_scaled = np.zeros((len(predictions), num_features))
    pred_scaled[:, 0] = predictions # Place predictions in the 'Close' column

    pred_scaled_original_scale = scaler.inverse_transform(pred_scaled)[:, 0]

    usd_to_inr = get_usd_to_inr()
    pred_scaled_inr = pred_scaled_original_scale * usd_to_inr

    if future_type != "custom":
        last_date = original_df.index[-1]
        temp_future_dates = []
        current_date = last_date
        while len(temp_future_dates) < steps:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5: # Monday to Friday
                temp_future_dates.append(current_date)
        future_dates = temp_future_dates[:steps]

    # Plot (Keeping this for compatibility, but frontend will use chart_data)
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, pred_scaled_inr, label='Predicted Price (INR)', color='blue')
    plt.title(f'Stock Price Forecast - {label} (in INR)')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    future_plot_base64 = plot_to_base64(plt)
    
    # Prepare predictions for display (e.g., last few days)
    future_table_data = []
    for i in range(min(len(future_dates), 10)): # Display last 10 predictions
        future_table_data.append({
            'date': future_dates[i].strftime('%Y-%m-%d'),
            'predicted_price': f"₹{pred_scaled_inr[i]:.2f}"
        })
    
    return {
        "plot": future_plot_base64, # Retained for backward compatibility if needed
        "table_data": future_table_data,
        "label": label,
        "chart_data": { # NEW: Add data for Chart.js
            "labels": [d.strftime('%Y-%m-%d') for d in future_dates],
            "data": pred_scaled_inr.tolist()
        }
    }
    

@app.route('/')
def index():
    return render_template('index.html')

# New endpoint to get stock quote and overview
@app.route('/get_stock_quote', methods=['POST'])
def get_stock_quote():
    company_name = request.json.get('companyName')
    if not company_name:
        return jsonify({"error": "Company name is required."}), 400

    listings_df = fetch_listing_status(api_key)
    if listings_df is None:
        return jsonify({"error": "Could not fetch stock listings. Please check API key or network."}), 500

    stock_symbol = get_stock_symbol(company_name, listings_df)
    if stock_symbol is None:
        return jsonify({"error": f"No stock symbol found for '{company_name}'. Please check the company name."}), 404

    try:
        ts = TimeSeries(key=api_key, output_format='json')
        fd = FundamentalData(key=api_key, output_format='json')

        # Get quote endpoint data
        quote_data, _ = ts.get_quote_endpoint(symbol=stock_symbol)

        # Get company overview
        overview_data, _ = fd.get_company_overview(symbol=stock_symbol)

        # Convert to INR if needed (for prices in quote_data)
        usd_to_inr_rate = get_usd_to_inr()
        if quote_data and "05. price" in quote_data:
            try:
                quote_data["05. price"] = f"₹{(float(quote_data['05. price']) * usd_to_inr_rate):.2f}"
            except ValueError:
                pass # Handle cases where price might not be a valid float
        if quote_data and "02. open" in quote_data:
            try:
                quote_data["02. open"] = f"₹{(float(quote_data['02. open']) * usd_to_inr_rate):.2f}"
            except ValueError:
                pass
        if quote_data and "03. high" in quote_data:
            try:
                quote_data["03. high"] = f"₹{(float(quote_data['03. high']) * usd_to_inr_rate):.2f}"
            except ValueError:
                pass
        if quote_data and "04. low" in quote_data:
            try:
                quote_data["04. low"] = f"₹{(float(quote_data['04. low']) * usd_to_inr_rate):.2f}"
            except ValueError:
                pass
        
        return jsonify({"symbol": stock_symbol, "quote": quote_data, "overview": overview_data})

    except Exception as e:
        print(f"Error fetching stock quote or overview: {e}")
        return jsonify({"error": f"Error fetching stock quote or overview for {stock_symbol}: {e}"}), 500

# New endpoint to get historical data for charts
@app.route('/get_historical_data', methods=['POST'])
def get_historical_data():
    symbol = request.json.get('symbol')
    timeframe = request.json.get('timeframe')

    if not symbol or not timeframe:
        return jsonify({"error": "Symbol and timeframe are required."}), 400

    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        if timeframe == 'daily':
            data, _ = ts.get_daily(symbol=symbol, outputsize='compact') # compact for last 100 days
        elif timeframe == 'weekly':
            data, _ = ts.get_weekly(symbol=symbol)
        elif timeframe == 'monthly':
            data, _ = ts.get_monthly(symbol=symbol)
        else:
            return jsonify({"error": "Invalid timeframe specified."}), 400

        if data.empty:
            return jsonify({"error": f"No historical data found for {symbol} ({timeframe})."}), 404

        data = data[::-1] # Reverse to get ascending dates
        data_filtered = data[['4. close']] # Only need close price for chart
        data_filtered.columns = ['Close']
        data_filtered.index = data_filtered.index.strftime('%Y-%m-%d') # Format index as string

        # Convert to INR
        usd_to_inr_rate = get_usd_to_inr()
        data_filtered['Close'] = data_filtered['Close'].astype(float) * usd_to_inr_rate
        
        # Prepare for JSON response
        chart_labels = data_filtered.index.tolist()
        chart_data = data_filtered['Close'].tolist()

        return jsonify({"labels": chart_labels, "data": chart_data})

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return jsonify({"error": f"Error fetching historical data for {symbol} ({timeframe}): {e}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    company_name = request.json.get('companyName')
    if not company_name:
        return jsonify({"error": "Company name is required."}), 400

    results = run_prediction(company_name)
    return jsonify(results)

@app.route('/forecast', methods=['POST'])
def forecast():
    forecast_options = request.json
    future_type = forecast_options.get('type')
    n = int(forecast_options.get('n')) if forecast_options.get('n') else None
    start_date_str = forecast_options.get('startDate')
    end_date_str = forecast_options.get('endDate')

    print("Received forecast request:")
    print("Type:", future_type, "| N:", n, "| Start:", start_date_str, "| End:", end_date_str)

    prediction_data = app.config.get('PREDICTION_DATA')
    if not prediction_data:
        return jsonify({"error": "Please run a stock prediction first."}), 400

    model = prediction_data['model']
    last_sequence = prediction_data['last_sequence']
    scaler = prediction_data['scaler']
    original_df = prediction_data['original_df']
    num_features = prediction_data['num_features']
    symbol = prediction_data['symbol']

    # Load forecast from cache if it exists
    cache_path = get_forecast_cache_path(symbol, str(forecast_options)) # Convert dict to string for hash
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print("✅ Loaded forecast from cache.")
            return jsonify(pickle.load(f))

    start_date = pd.to_datetime(start_date_str) if start_date_str else None
    end_date = pd.to_datetime(end_date_str) if end_date_str else None

    forecast_results = predict_future_prices(
        model, last_sequence, scaler,
        original_df, num_features,
        future_type, n, start_date, end_date
    )

    with open(cache_path, 'wb') as f:
        pickle.dump(forecast_results, f)

    return jsonify(forecast_results)

@app.route('/get_custom_history_plot', methods=['POST'])
def get_custom_history_plot():
    data = request.json
    start_date_str = data.get('startDate')
    end_date_str = data.get('endDate')
    symbol = data.get('symbol') # You'll need to pass the stock symbol from frontend

    if not start_date_str or not end_date_str or not symbol:
        return jsonify({"error": "Start date, end date, and symbol are required."}), 400

    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format. UseYYYY-MM-DD."}), 400

    # Retrieve the prediction data from app.config (set during /predict call)
    prediction_data = app.config.get('PREDICTION_DATA')
    if not prediction_data or prediction_data['symbol'] != symbol:
        return jsonify({"error": "Prediction data not available for this symbol. Please run a full prediction first."}), 400

    original_df = prediction_data['original_df']
    model = prediction_data['model']
    scaler = prediction_data['scaler']
    num_features = prediction_data['num_features']
    sequence_len = 200 # Assuming this is consistent

    try:
        features = ['Close', 'Volume', 'SMA_20', '52_Week_High', '52_Week_Low',
                    '4_Week_High', '4_Week_Low', 'Prev_Close', 'Next_Open', 'Trade_Price']
        scaled_data_full = scaler.fit_transform(original_df[features]) # Use the same scaler
        X_full, y_full = create_sequences(scaled_data_full, sequence_len)

        # Make predictions for the full historical period
        # This can be inefficient if done on every custom history plot request.
        # Ideally, `run_prediction` would store `y_actual_full` and `y_predicted_full` along with their dates.
        all_predictions_scaled = model.predict(X_full)

        # Rescale all predictions and actuals
        all_predictions_rescaled = rescale_predictions(all_predictions_scaled, scaler, num_features)
        y_actual_rescaled = rescale_predictions(y_full.reshape(-1, 1), scaler, num_features)

        usd_to_inr = get_usd_to_inr() # Get the latest conversion rate
        all_predictions_rescaled *= usd_to_inr
        y_actual_rescaled *= usd_to_inr

        # Adjust indices to match the original_df dates after sequence creation
        # The dates for X_full and y_full start from `sequence_len` index of original_df
        plot_dates = original_df.index[sequence_len:sequence_len + len(y_full)]

        # Filter data for the requested custom range
        mask = (plot_dates >= start_date) & (plot_dates <= end_date)

        # Ensure the filtered arrays are not empty
        if not mask.any():
            return jsonify({"error": "No data available for the selected custom date range."}), 404

        filtered_dates = plot_dates[mask]
        filtered_actuals = y_actual_rescaled[mask]
        filtered_predictions = all_predictions_rescaled[mask]

        if filtered_dates.empty:
            return jsonify({"error": "No data available for the selected custom date range."}), 404

        # Generate the plot
        plt.figure(figsize=(14, 6))
        plt.plot(filtered_dates, filtered_actuals, label='Actual Prices', color='red')
        plt.plot(filtered_dates, filtered_predictions, label='Predicted Prices', color='orange')
        plt.title(f"{symbol} Actual vs Predicted ({start_date_str} to {end_date_str})")
        plt.xlabel('Date')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return jsonify({
    "labels": filtered_dates.strftime('%Y-%m-%d').tolist(),
    "actualData": filtered_actuals.tolist(),
    "predictedData": filtered_predictions.tolist()
})


    except Exception as e:
        print(f"Error generating custom history plot: {e}")
        return jsonify({"error": f"Failed to generate custom history plot: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
