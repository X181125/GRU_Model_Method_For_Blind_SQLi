# Demo GRU Method - Blind SQLi Comparison

## Mô tả
Demo web đơn giản để so sánh hiệu suất tấn công Blind SQL Injection giữa:
- **GRU Neural Network Method**: Sử dụng mô hình GRU đã huấn luyện để sinh tên bảng/cột thông minh
- **SQLMap (Traditional)**: Phương pháp brute-force truyền thống với dictionary

## Dựa trên bài báo
"A SQL Blind Injection Method Based on Gated Recurrent Neural Network" - IEEE 2020

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install flask flask-cors requests tensorflow numpy colorama
```

### 2. Cài đặt SQLMap (tùy chọn, cho so sánh)

```bash
# Ubuntu/Debian
sudo apt install sqlmap

# Hoặc pip
pip install sqlmap
```

### 3. Đảm bảo model GRU đã được huấn luyện

Model nằm tại: `../GRU_Model_Method_For_Blind_SQLi/trained_models/`

Nếu chưa có model, chạy huấn luyện:
```bash
cd ../GRU_Model_Method_For_Blind_SQLi
python train_gru_model.py
```

## Chạy Demo

```bash
cd Demo_GRU_Method
python server.py
```

Truy cập: http://127.0.0.1:5000

## Tính năng

### 1. Vulnerable Endpoints
- `/login` - Boolean-based blind SQLi
- `/time` - Time-based blind SQLi  
- `/search` - LIKE-based injection

### 2. Attack APIs
- `POST /api/attack/gru/start` - Bắt đầu tấn công GRU
- `POST /api/attack/sqlmap/start` - Bắt đầu tấn công SQLMap
- `GET /api/attack/status` - Lấy trạng thái tấn công
- `GET /api/compare` - So sánh kết quả
- `GET /api/database/info` - Xem cấu trúc database thực

### 3. Demo Database
SQLite database với các bảng:
- `users` - Thông tin người dùng
- `admin` - Thông tin admin
- `products` - Sản phẩm
- `orders` - Đơn hàng
- `sessions` - Phiên đăng nhập

## So sánh phương pháp

| Tiêu chí | GRU Method | SQLMap |
|----------|------------|--------|
| Cách tiếp cận | Neural Network | Dictionary/Brute-force |
| Số requests | Ít hơn | Nhiều hơn |
| Độ chính xác | Cao với tên phổ biến | Phụ thuộc wordlist |
| Thời gian | Nhanh hơn | Chậm hơn |
| Linh hoạt | Sinh tên mới | Chỉ dùng có sẵn |

## Cảnh báo

⚠️ **Chỉ sử dụng cho mục đích học tập và nghiên cứu!**

Demo này được thiết kế để chạy trên localhost với database giả lập. Không sử dụng cho mục đích tấn công hệ thống thực.

## Cấu trúc thư mục

```
Demo_GRU_Method/
├── server.py           # Flask backend server
├── templates/
│   └── index.html      # Frontend demo page
├── static/             # Static files (CSS, JS)
├── vulnerable.db       # SQLite database (auto-created)
└── README.md           # This file
```

## License

MIT License - For educational purposes only.
