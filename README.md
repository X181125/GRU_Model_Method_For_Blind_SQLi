# GRU Model Method For Blind SQLi

Mã nguồn minh họa cách huấn luyện mô hình sinh chuỗi dựa trên GRU để gợi ý tên bảng/cột phổ biến, sau đó dùng chúng trong khai thác Blind SQL Injection dạng time-based. Logic và cơ sở lý thuyết tham chiếu trong `Base_paper/A SQL Blind Injection Method Based on Gated Recurrent Neural Network.pdf`.

> Chỉ sử dụng cho mục đích học tập/kiểm thử có phép. Không dùng lên hệ thống không được ủy quyền.

## Cấu trúc thư mục
- `data/`: danh sách tên bảng (`common-tables.txt`) và cột (`common-columns.txt`) phổ biến (nguồn sqlmap).
- `trained_models/`: chứa model GRU đã huấn luyện (`blind_sqli_gru_model.h5`).
- `train_gru_model.py`: huấn luyện model (ưu tiên GPU, fallback CPU).
- `generate_names.py`: sinh tên bảng/cột gợi ý từ model.
- `blind_sqli_exploit.py`: thử payload blind SQLi với tên sinh ra.
- `test_lib.py`: kiểm tra thiết bị GPU TensorFlow.
- `Base_paper/`: bản PDF bài báo tham khảo.

## Yêu cầu môi trường
- Python 3.10+
- TensorFlow 2.x (GPU khuyến khích), NumPy, SciPy, scikit-learn. Cài nhanh: `pip install -r requirements.txt`.
- Để chạy GPU: cài CUDA + cuDNN tương thích với phiên bản TensorFlow (xem bảng compatibility của TF).

## Thiết lập nhanh
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Kiểm tra GPU: `python test_lib.py`.

## Huấn luyện model (ưu tiên GPU)
`train_gru_model.py` tự chọn GPU qua `tf.distribute.MirroredStrategy`; nếu không có sẽ rơi về CPU. Các tham số chính: `SEQ_LENGTH=20`, `BATCH_SIZE=256`, `EPOCHS=50`, `VALIDATION_SPLIT=0.1`, GRU 2 tầng 256 units, dropout, ReduceLROnPlateau + EarlyStopping.

Chạy huấn luyện:
```bash
python train_gru_model.py
```
Model sau khi train lưu tại `trained_models/blind_sqli_gru_model.h5` (tạo thư mục nếu chưa có). Tập vocab được xây từ `data/` sau khi bỏ comment và dòng trống để khớp lúc suy luận.

## Sinh tên bảng/cột gợi ý
```bash
python generate_names.py
```
Sinh 10 chuỗi mặc định (tối đa 20 ký tự, dừng khi gặp ký tự xuống dòng). Có thể chỉnh `max_len` trong hàm `generate_name` nếu cần.

## Khai thác Blind SQLi mẫu
`blind_sqli_exploit.py` sinh tên và bắn payload MySQL time-based:
- Chỉnh mục tiêu trong file: `TARGET`, `PARAM`, `DELAY`.
- Payload dạng `1' OR IF(EXISTS(...), SLEEP(DELAY), 0)-- -` kiểm tra tên bảng trong `information_schema.tables`.
- Chạy:
```bash
python blind_sqli_exploit.py
```
Mặc định thử tối đa 20 tên, dừng khi thấy độ trễ >= `DELAY`. Nên URL-encode và thêm logic retry/median nếu áp dụng thực tế để giảm nhiễu.

## Gợi ý mở rộng
- Lưu/đọc vocab cùng model để cố định mapping char-index.
- Thử điều chỉnh `temperature` hoặc top-k khi sampling để cân bằng đa dạng/chất lượng tên sinh.
- Bổ sung checkpointing, logging (TensorBoard) để theo dõi quá trình train.
- Mở rộng payload cho các DBMS khác (PostgreSQL, MSSQL) và thêm enumerate cột sau khi tìm thấy bảng.

## Cảnh báo pháp lý
Mọi hoạt động tấn công phải có sự cho phép. Tác giả không chịu trách nhiệm nếu dùng sai mục đích.
