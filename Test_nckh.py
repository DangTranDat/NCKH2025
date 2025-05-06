from flask import Flask, render_template, request, jsonify
import MySQLdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Cấu hình ESP32 IP
ESP32_IP = "http://192.168.166.199"

# Cấu hình kết nối cơ sở dữ liệu MySQL
db = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="dat03092003",
    db="air_quality_db"
)
cursor = db.cursor()

# Endpoint nhận dữ liệu từ ESP32
@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    air_quality = data.get('air_quality')

    # Lưu dữ liệu vào cơ sở dữ liệu
    query = "INSERT INTO air_quality (temperature, humidity, air_quality) VALUES (%s, %s, %s)"
    cursor.execute(query, (temperature, humidity, air_quality))
    db.commit()
    
    return jsonify({"message": "Data received"}), 200

# Endpoint cung cấp dữ liệu cho giao diện web
@app.route('/api/chart-data', methods=['GET'])
def chart_data():
    # Lấy 500 bản ghi gần nhất từ cơ sở dữ liệu
    query = "SELECT timestamp, air_quality, temperature, humidity FROM air_quality ORDER BY timestamp ASC LIMIT 500"
    cursor.execute(query)
    result = cursor.fetchall()

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(result, columns=['timestamp', 'air_quality', 'temperature', 'humidity'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Dự đoán xu hướng tương lai bằng Holt-Winters cho air_quality, temperature, và humidity
    future_steps = 10  # Số bước dự đoán
    interval = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2]).seconds
    future_timestamps = [
        df['timestamp'].iloc[-1] + timedelta(seconds=interval * i) for i in range(1, future_steps + 1)
    ]
    
    # Holt-Winters Forecasting for air_quality
    air_quality_model = ExponentialSmoothing(df['air_quality'], trend='add', seasonal=None, damped_trend=True)
    air_quality_model_fit = air_quality_model.fit()
    air_quality_predictions = air_quality_model_fit.forecast(future_steps)

    # Holt-Winters Forecasting for temperature
    temperature_model = ExponentialSmoothing(df['temperature'], trend='add', seasonal=None, damped_trend=True)
    temperature_model_fit = temperature_model.fit()
    temperature_predictions = temperature_model_fit.forecast(future_steps)

    # Holt-Winters Forecasting for humidity
    humidity_model = ExponentialSmoothing(df['humidity'], trend='add', seasonal=None, damped_trend=True)
    humidity_model_fit = humidity_model.fit()
    humidity_predictions = humidity_model_fit.forecast(future_steps)

    # Tính toán thống kê cơ bản bằng NumPy
    temperature_mean = np.mean(df['temperature'])
    humidity_mean = np.mean(df['humidity'])
    air_quality_mean = np.mean(df['air_quality'])

    temperature_std = np.std(df['temperature'])
    humidity_std = np.std(df['humidity'])
    air_quality_std = np.std(df['air_quality'])

    # Tính toán tương quan giữa các biến (air_quality, temperature, humidity)
    correlation_matrix = df[['air_quality', 'temperature', 'humidity']].corr()

    # Lấy thời gian hiện tại
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Lưu thống kê vào cơ sở dữ liệu
    stats_query = """
    INSERT INTO air_quality_statistics (temperature_mean, humidity_mean, air_quality_mean, 
    temperature_std, humidity_std, air_quality_std, timestamp)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(stats_query, (temperature_mean, humidity_mean, air_quality_mean, temperature_std, humidity_std, air_quality_std, timestamp))
    db.commit()

    # Trả về dữ liệu JSON
    return jsonify({
        "timestamps": df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "AirQuality": df['air_quality'].tolist(),
        "Temperature": df['temperature'].tolist(),
        "Humidity": df['humidity'].tolist(),
        "FutureTimestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_timestamps],
        "AirQualityTrendPrediction": air_quality_predictions.tolist(),
        "TemperatureTrendPrediction": temperature_predictions.tolist(),
        "HumidityTrendPrediction": humidity_predictions.tolist(),
        "CorrelationMatrix": correlation_matrix.to_dict(),  # Chuyển ma trận tương quan thành dict
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

# Endpoint gửi lệnh điều khiển đến ESP32
@app.route('/api/control', methods=['POST'])
def control_device():
    data = request.get_json()
    device = data.get('device')  # "light" hoặc "fan"
    action = data.get('action')  # "on" hoặc "off"

    if not device or not action:
        return jsonify({"message": "Invalid parameters"}), 400

    # Gửi yêu cầu HTTP đến ESP32
    esp32_url = f"{ESP32_IP}/control"
    payload = {"device": device, "action": action}
    response = requests.post(esp32_url, json=payload)
    
    if response.status_code == 200:
        return jsonify({"message": f"{device.capitalize()} turned {action}"}), 200
    else:
        return jsonify({"message": "Failed to control device"}), 500


# Trang web hiển thị
@app.route('/a')
def index():
    return render_template('test_nckh.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    