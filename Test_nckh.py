from flask import Flask, render_template, request, jsonify
import pymysql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Cấu hình ESP32 IP
ESP32_IP = "http://192.168.166.199"

# Lấy thông tin cơ sở dữ liệu từ biến môi trường
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASS', '')
DB_NAME = os.environ.get('DB_NAME', 'air_quality_db')

# Kết nối cơ sở dữ liệu
db = pymysql.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    db=DB_NAME,
    cursorclass=pymysql.cursors.DictCursor
)
cursor = db.cursor()

# Nhận dữ liệu từ ESP32
@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    air_quality = data.get('air_quality')

    query = "INSERT INTO air_quality (temperature, humidity, air_quality) VALUES (%s, %s, %s)"
    cursor.execute(query, (temperature, humidity, air_quality))
    db.commit()

    return jsonify({"message": "Data received"}), 200

# Cung cấp dữ liệu cho web
@app.route('/api/chart-data', methods=['GET'])
def chart_data():
    query = "SELECT timestamp, air_quality, temperature, humidity FROM air_quality ORDER BY timestamp ASC LIMIT 500"
    cursor.execute(query)
    result = cursor.fetchall()

    if not result:
        return jsonify({"message": "No data available"}), 400

    df = pd.DataFrame(result)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    future_steps = 10
    interval = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2]).seconds
    future_timestamps = [
        df['timestamp'].iloc[-1] + timedelta(seconds=interval * i) for i in range(1, future_steps + 1)
    ]

    air_quality_model = ExponentialSmoothing(df['air_quality'], trend='add', seasonal=None, damped_trend=True).fit()
    air_quality_predictions = air_quality_model.forecast(future_steps)

    temperature_model = ExponentialSmoothing(df['temperature'], trend='add', seasonal=None, damped_trend=True).fit()
    temperature_predictions = temperature_model.forecast(future_steps)

    humidity_model = ExponentialSmoothing(df['humidity'], trend='add', seasonal=None, damped_trend=True).fit()
    humidity_predictions = humidity_model.forecast(future_steps)

    temperature_mean = np.mean(df['temperature'])
    humidity_mean = np.mean(df['humidity'])
    air_quality_mean = np.mean(df['air_quality'])

    temperature_std = np.std(df['temperature'])
    humidity_std = np.std(df['humidity'])
    air_quality_std = np.std(df['air_quality'])

    correlation_matrix = df[['air_quality', 'temperature', 'humidity']].corr()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stats_query = """
    INSERT INTO air_quality_statistics (temperature_mean, humidity_mean, air_quality_mean, 
    temperature_std, humidity_std, air_quality_std, timestamp)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(stats_query, (temperature_mean, humidity_mean, air_quality_mean,
                                 temperature_std, humidity_std, air_quality_std, timestamp))
    db.commit()

    return jsonify({
        "timestamps": df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "AirQuality": df['air_quality'].tolist(),
        "Temperature": df['temperature'].tolist(),
        "Humidity": df['humidity'].tolist(),
        "FutureTimestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_timestamps],
        "AirQualityTrendPrediction": air_quality_predictions.tolist(),
        "TemperatureTrendPrediction": temperature_predictions.tolist(),
        "HumidityTrendPrediction": humidity_predictions.tolist(),
        "CorrelationMatrix": correlation_matrix.to_dict(),
        "Statistics": {
            "temperature_mean": temperature_mean,
            "humidity_mean": humidity_mean,
            "air_quality_mean": air_quality_mean,
            "temperature_std": temperature_std,
            "humidity_std": humidity_std,
            "air_quality_std": air_quality_std,
            "timestamp": timestamp
        }
    })

# Điều khiển thiết bị qua ESP32
@app.route('/api/control', methods=['POST'])
def control_device():
    data = request.get_json()
    device = data.get('device')
    action = data.get('action')

    if not device or not action:
        return jsonify({"message": "Invalid parameters"}), 400

    esp32_url = f"{ESP32_IP}/control"
    payload = {"device": device, "action": action}
    try:
        response = requests.post(esp32_url, json=payload)
        if response.status_code == 200:
            return jsonify({"message": f"{device.capitalize()} turned {action}"}), 200
        else:
            return jsonify({"message": "Failed to control device"}), 500
    except Exception as e:
        return jsonify({"message": str(e)}), 500

# Giao diện web
@app.route('/a')
def index():
    return render_template('test_nckh.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
