# GRU Method for Blind SQLi

Mô hình GRU sinh tên bảng/cột phổ biến để hỗ trợ khai thác blind SQLi (time-based). Ý tưởng lấy từ paper `Base_paper/A SQL Blind Injection Method Based on Gated Recurrent Neural Network.pdf`. Repo cung cấp toàn bộ pipeline: tiền xử lý danh sách tên, huấn luyện GRU ký tự, sinh payload time-based và kịch bản demo. Chỉ dùng cho mục đích học tập/kiểm thử được ủy quyền.

## Cấu trúc thư mục
- `data/`: danh sách tên bảng (`common-tables.txt`) và cột (`common-columns.txt`) trích từ sqlmap (giăng trong dòng comment bản quyền).
- `train_gru_model.py`: huấn luyện model GRU ký tự (seed, split, callbacks, lưu config/vocab/history).
- `trained_models/`: artefact sau train (`blind_sqli_gru_best.h5`, `blind_sqli_gru_final.keras`, `blind_sqli_gru_final.weights.h5`, `vocab.json`, `config.json`, `training_log.csv`, `training_history.png`, `final_results.json`). Ban đầu có thể trống nếu chưa train.
- `generate_names.py`: script ngắn nạp model cũ (`blind_sqli_gru_model.h5`) và sinh 10 tên thử nghiệm (không dùng vocab JSON mới).
- `blind_sqli_exploit.py`: nạp model tốt nhất + vocab/config, sinh tên bảng/cột và bắn payload time-based lên đích.
- `test_lib.py`: kiểm tra GPU TensorFlow.
- `Base_paper/`: bài báo gốc.
- `requirements.txt`: phụ thuộc (TF 2.19 + Keras 3.12).

## Môi trường
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Kiểm tra GPU: `python test_lib.py`.

> Ghi chú tương thích: model `.h5` được lưu với TensorFlow 2.19/Keras 3.12. Nếu dùng phiên bản thấp hơn (ví dụ 2.12) hoặc cao hơn, Keras có thể cần file `.keras`/`.weights.h5` và vocab/config đi kèm. Script exploit đã đọc đúng vocab/config để tránh sai lệch index.

## Dữ liệu và tiền xử lý (train_gru_model.py)
- Đọc `common-tables.txt` + `common-columns.txt`, bỏ dòng comment, lowercase, nối thêm END_TOKEN `*` vào cuối mỗi tên để model học dấu kết thúc.
- Xây dựng vocab ký tự theo tần suất (chỉ số thấp hơn = ký tự phổ biến hơn), chèn `PAD_TOKEN=<PAD>`, `UNK_TOKEN=<UNK>` đầu danh sách. Lưu mapping vào `trained_models/vocab.json`.
- Chuỗi huấn luyện: ghép các tên bằng newline, trích cửa sổ trượt độ dài `SEQ_LENGTH=5`, bỏ qua cửa sổ bị END_TOKEN để tránh học tiếp chuỗi qua tên khác.
- Tách tập: 80% train, 10% val, 10% test. Tạo `tf.data.Dataset` shuffle/prefetch.

## Kiến trúc & tham số
- Embedding 256 chiều (mask padding) → GRU 512 (return_sequences, dropout 0.3, recurrent_dropout 0.2, L2) → BatchNorm → GRU 512 → BatchNorm → Dense 512 ReLU → Dropout → BatchNorm → Dense 256 ReLU → Dropout → Dense vocab softmax.
- Tối ưu hóa: Adam lr 1e-3, clipnorm 1.0; cosine LR schedule + ReduceLROnPlateau backup; EarlyStopping (patience 20) + ModelCheckpoint lưu `trained_models/blind_sqli_gru_best.h5`; CSVLogger.
- Seed cố định 42; sử dụng `tf.distribute.MirroredStrategy` nếu có GPU, bật memory growth.
- Hyperparams chính: `BATCH_SIZE=64`, `EPOCHS=200`, `VALIDATION_SPLIT=0.1`, `TEST_SPLIT=0.1`, dropout 0.3, L2 1e-4.

## Huấn luyện
```bash
python train_gru_model.py
```
Trong quá trình train:
- Log ra kích thước vocab, chỉ số END token, thống kê mẫu (top ký tự phổ biến).
- Callbacks tự động lưu checkpoint tốt nhất (`blind_sqli_gru_best.h5`) và vẽ biểu đồ `training_history.png`.
- Đánh giá trên tập test, tính perplexity (exp cross-entropy) theo paper, thử generate mẫu với nhiều seed/temperature. Kết quả đã train sẵn: `test_loss` 2.2469, `test_accuracy` 0.4876, `top5` 0.8056, `perplexity` ~5.99 (`trained_models/final_results.json`).
- Lưu model cuối ở 2 dạng: `blind_sqli_gru_final.keras` và `blind_sqli_gru_final.weights.h5` + `config.json` (seq_length, dropout, vocab_size...).

## Sinh tên (demo thử)
- Cách chuẩn (dùng vocab/config đồng nhất với model): dùng các hàm `generate_text` hoặc `test_generation` trong `train_gru_model.py` sau khi train xong, hoặc dùng `blind_sqli_exploit.py` (đã đọc vocab.json).
- Script ngắn sẵn có `generate_names.py` (sử dụng file `trained_models/blind_sqli_gru_model.h5` cũ, vocab từ data/). Chạy: `python generate_names.py`. Script này không đọc vocab.json, chỉ phù hợp khi model được train cùng vocab mặc định.

## Khai thác time-based (blind_sqli_exploit.py)
- Nạp `trained_models/blind_sqli_gru_best.h5` + `vocab.json` + `config.json` để đảm bảo mapping ký tự đúng (END_TOKEN index, PAD/UNK). Có kiểm tra file tồn tại trước khi chạy.
- Sinh tên bảng/cột bằng temperature sampling + top-k, seed phổ biến (`us, ad, pa, id, na...`), dùng chuỗi đệm độ dài tối đa 20 ký tự, dùng END_TOKEN `*` làm ký tự dừng.
- Payload mẫu (SQLite demo):
  - V1: `UNION SELECT 1,'<table>','p','r' FROM sqlite_master WHERE name='<table>' -- -`
  - V2: `CASE WHEN (SELECT name FROM sqlite_master WHERE type='table' AND name='<table>') ...`
  - V3 (đang dùng): `admin' OR (SELECT COUNT(*) FROM <table>) > 0 -- -`
  - Payload cột: `admin' OR (SELECT <column> FROM <table> LIMIT 1) IS NOT NULL -- -`
- Gửi request GET đến `TARGET` (mặc định `http://127.0.0.1:5000/time`, param `username`) và đo thời gian. Nếu thời gian >= `DELAY_THRESHOLD=4s` thì coi là có bảng/cột. Script in log dạng `[idx/total] <name> -> 4.12s  DELAY` và tổng kết danh sách nghi ngờ.
- Flow mặc định: sinh mẫu tên bảng, hiển thị sanity check, cho bấm Enter để bắt đầu fuzz; nếu tìm thấy bảng thì hỏi tiếp tục fuzz cột.

Chạy khai thác:
```bash
python blind_sqli_exploit.py
```
Thay đổi `TARGET`, `PARAM`, `DELAY_THRESHOLD` cho phù hợp đích lab.

## Tệp sinh ra sau train
- `trained_models/blind_sqli_gru_best.h5`: checkpoint val_loss tốt nhất, dùng cho exploit.
- `trained_models/blind_sqli_gru_final.keras` + `blind_sqli_gru_final.weights.h5`: snapshot cuối.
- `trained_models/vocab.json`: mapping ký tự → chỉ số (có END token index).
- `trained_models/config.json`: siêu tham số (seq_length, dropout, vocab_size...).
- `trained_models/training_log.csv`,