from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import base64
import threading
import tempfile
import os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)  # Kích hoạt CORS cho tất cả các đường dẫn

# Dữ liệu huấn luyện và mô hình
X_train = [
    [7.3, 0.67, 0.02, 2.4, 0.083, 9.0, 52.0, 0.9968, 3.36, 0.61, 9.9],
    [7.0, 0.68, 0.06, 1.8, 0.075, 17.0, 26.0, 0.9976, 3.48, 0.57, 9.3],
    [8.3, 0.42, 0.62, 19.25, 0.040, 41.0, 172.0, 1.0002, 2.98, 0.67, 9.3],
    [6.8, 0.63, 0.12, 3.8, 0.099, 16.0, 126.0, 0.9969, 3.55, 0.63, 10.2],
    [6.9, 0.58, 0.01, 1.9, 0.080, 8.0, 22.0, 0.9968, 3.38, 0.52, 9.4],
    [8.1, 0.64, 0.24, 2.8, 0.094, 5.0, 13.0, 0.9972, 3.37, 0.65, 9.7],
    [7.5, 0.49, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.80, 10.5],
    [7.4, 0.34, 0.42, 1.1, 0.033, 17.0, 76.0, 0.9962, 3.2, 0.74, 10.8],
]

y_train = [5, 6, 4, 7, 5, 6, 7, 8]
 

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Định tuyến đến trang chủ hiển thị 
@app.route('/index')
def index():
    return render_template('index.html')

def save_plot(features, quality_prediction):
    # Tạo một đối tượng Figure với kích thước là 10x6 inches
    plt.figure(figsize=(10, 6))

    # Lưu biểu đồ xuống một tệp tạm thời với định dạng PNG
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name)

    # Đóng đối tượng Figure để giải phóng tài nguyên
    plt.close()

    # Trả về đường dẫn của tệp tạm thời
    return temp_file.name


# Hàm dự đoán chất lượng, được gọi khi nhận được yêu cầu POST từ trình duyệt
def predict_quality():
    # Kiểm tra xem có phải là phương thức POST hay không
    if request.method == 'POST':
        # Lấy giá trị từ form và chuyển đổi sang kiểu float
        features = [float(request.form['feature_' + str(i)]) for i in range(11)]

        # Dự đoán chất lượng sử dụng hàm predict (giả sử nó đã được định nghĩa)
        quality_prediction = predict(features)

        # Kiểm tra xem hiện tại có đang ở trong luồng chính hay không
        if threading.current_thread() != threading.main_thread():
            # Nếu không ở trong luồng chính, chuyển sang luồng chính trước khi vẽ biểu đồ
            app.logger.info("Not in the main thread. Switching to main thread.")
            with app.app_context():
                plot_prediction(features, quality_prediction)
        else:
            # Nếu đang ở trong luồng chính, vẽ biểu đồ bình thường
            plot_prediction(features, quality_prediction)
        
        # Vẽ biểu đồ và lưu xuống một tệp tạm thời
        plot_path = save_plot(features, quality_prediction)

        # Trả về template với các giá trị và đường dẫn của biểu đồ
        return render_template('index.html', quality_prediction=quality_prediction, features=features, plot_path=plot_path)

    # Nếu không phải là phương thức POST, trả về template mặc định
    return render_template('index.html')




# Đường dẫn /plot để nhận dữ liệu và trả về biểu đồ và dự đoán
@app.route('/plot', methods=['POST'])
def plot():
    features = [float(request.form['feature_' + str(i)]) for i in range(11)]
    quality_prediction = predict(features)

    # Vẽ biểu đồ
    img_data = plot_prediction(features, quality_prediction)
    img_str = base64.b64encode(img_data).decode('utf-8')

    return jsonify({'img': img_str, 'prediction': quality_prediction})

# Hàm dự đoán sử dụng mô hình
def predict(features):
    return model.predict([features])[0]

# Hàm vẽ biểu đồ
def plot_prediction(features, prediction):
    feature_names = ['Độ axit cố định', 'Độ bay hơi axit', 'Axit citric', 'Đường dư', 'Clorua',
                     'Lưu huỳnh đioxit tự do', 'Tổng lượng lưu huỳnh điôxít', 'Mật độ', 'Độ pH', 'Muối Sulfat', 'Cồn']

    # Màu sắc cho mỗi cột
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(feature_names, features, color=colors, alpha=0.7, label='Đặc trưng đầu vào')
    plt.axhline(y=prediction, color='r', linestyle='-', label='Chất lượng dự đoán')

    # Đặt màu cho cột dự đoán
    bars[feature_names.index('Độ pH')].set_color('r')

    plt.xlabel('Các Đặc Trưng')
    plt.ylabel('Giá Trị')
    plt.title('Dự Đoán Chất Lượng Rượu')
    plt.legend()
    plt.xticks(rotation=45, ha='right')  # Xoay tên cột để tránh trùng lắp
    plt.tight_layout()

    # Chuyển đổi biểu đồ thành ảnh
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close()

    return img_data.getvalue()



if __name__ == '__main__':
    app.run(threaded=True)

