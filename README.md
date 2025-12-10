# FINAL_PROJECT_DS_K23: HỆ THỐNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN

Dự án triển khai một **pipeline Học máy (Machine Learning Pipeline)** hoàn chỉnh theo kiến trúc **OOP (Lập trình hướng đối tượng)** để giải quyết bài toán **Hồi quy (Regression)** dự đoán giá căn hộ trên thị trường thứ cấp.

Pipeline bao gồm các module độc lập: Xử lý Dữ liệu Thô, Kỹ thuật Đặc trưng Chuyên biệt, Huấn luyện Model Đa mô hình (Random Forest, XGBoost, SVR, Ensemble) và Tối ưu hóa Siêu tham số an toàn (chống Data Leakage).

## Cấu Trúc Thư Mục Dự Án (Project Structure)

Cấu trúc thư mục được tổ chức theo chuẩn MDI (Modular Design for Intelligence) để đảm bảo tính module hóa, dễ bảo trì và khả năng tái lập.

``` text
FINAL_PROJECT_DS_K23/
│
├── main.py             # Lõi Điều Phối (Orchestrator): File chạy chính, điều khiển luồng dữ liệu giữa tất cả các module.
├── requirements.txt    # Quản lý Phụ thuộc: Danh sách các thư viện Python cần thiết.
│
├── config/             # Quản lý Cấu hình
│   └── config.yaml     # Chứa tất cả tham số (paths, hyperparameters, random seed).
│
├── data/               # Quản lý Dữ liệu
│   ├── raw/            # Dữ liệu Thô
│   │   └── rawdata.csv # Dữ liệu đầu vào ban đầu (đã có sẵn trong Repo).
│   └── processed/      # Dữ liệu Sạch
│       └── df_model_ready.csv, EDA_data.csv # Dữ liệu sau xử lý, sẵn sàng cho Model và Visualization.
│
├── logs/               # Ghi Nhật ký
│   └── full_pipeline.log # File log ghi lại chi tiết quá trình chạy.
│
├── models/             # Lưu trữ Kết quả & Mô hình
│   ├── saved_models/   # Mô hình Tuần tự hóa (.joblib: Voting_Ensemble, scaler...).
│   └── metrics/        # Kết quả Định lượng (model_results_summary.csv/.json: bảng tổng hợp RMSE, R2, MAE của các mô hình).
│
├── reports/            # Báo cáo và Hình ảnh
│   └── figures/        # Biểu đồ Tự động Sinh (so sánh hiệu năng, Residuals, Feature Importance).
│
└── src/                # SOURCE CODE CHÍNH (Modules OOP)
     ├── preprocessor.py  # Data Preprocessing: Chứa các Class xử lý Tiền xử lý, Imputation, Lọc ngoại lai và Feature Engineering.
     ├── model_trainer.py # Model Training & Tuning: Class quản lý toàn bộ quy trình huấn luyện, tối ưu tham số và đánh giá mô hình.
     ├── visualization.py # Trực quan hóa: Các hàm hỗ trợ vẽ biểu đồ EDA chuyên sâu và phân tích kết quả mô hình.
     └── utils.py         # Tiện ích Nền tảng: Các hàm thiết yếu (load_config, set_seed, setup_logger).
```

## Hướng Dẫn Cài Đặt và Chạy

Để chạy lại toàn bộ pipeline, cần môi trường Python (khuyến nghị Python 3.8+) và thực hiện các bước sau:

### Bước 1: Clone Repository

Tải source code về máy:

```bash
git clone https://github.com/uyen-huynh2808/predict-house-price
cd FINAL_PROJECT_DS_K23
```

### Bước 2: Chuẩn bị Môi trường (Environment Setup)
Tạo và kích hoạt môi trường ảo (Virtual Environment) để cô lập các thư viện:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (trên Windows)
.\venv\Scripts\activate
# Kích hoạt môi trường ảo (trên MacOS/Linux)
source venv/bin/activate
```

### Bước 3: Cài đặt Thư viện Phụ thuộc
Sử dụng file `requirements.txt` để cài đặt tất cả các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### Bước 4: Chạy Pipeline
Vì file dữ liệu thô (`rawdata.csv`) đã có sẵn trong thư mục `data/raw/`, chỉ cần chạy lệnh sau để khởi động toàn bộ quy trình:

```bash
python main.py --config config/config.yaml
```
> [Lưu ý] Sau khi chạy xong, kết quả đánh giá mô hình, biểu đồ và các file log sẽ được tạo tự động trong thư mục `models/` và `reports/figures/`.
