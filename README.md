# GRU Method for Blind SQLi

Huong dan huan luyen va su dung mo hinh GRU sinh ten bang/cot pho bien de ho tro khai thac Blind SQLi (time-based). Co so ly thuyet xem trong `Base_paper/A SQL Blind Injection Method Based on Gated Recurrent Neural Network.pdf`. Chi dung cho muc dich hoc tap/kiem thu duoc uy quyen.

## Cau truc thu muc
- `data/`: danh sach ten bang (`common-tables.txt`) va cot (`common-columns.txt`) tu sqlmap (co dong comment ban quyen).
- `train_gru_model.py`: huan luyen GRU ky tu tu danh sach tren (uu tien GPU, co warning neu sai version TF).
- `trained_models/`: model da luu (`blind_sqli_gru_model.h5`). Ban dau co the chua ton tai hoac la model cu.
- `generate_names.py`: nap model, sinh ten bang/cot de tham khao.
- `blind_sqli_exploit.py`: dung model de sinh payload blind SQLi (mau trich xuat bang) va co fallback huan luyen nhanh neu khong load duoc `.h5` do lech Keras.
- `test_lib.py`: kiem tra GPU TensorFlow.
- `Base_paper/`: bai bao tham khao.
- `requirements.txt`: phu thuoc (khuyen nghi TF 2.12 de tuong thich file `.h5`).

## Cai dat nhanh
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Kiem tra GPU: `python test_lib.py`.

### Luu y ve phien ban TensorFlow/Keras
Model `.h5` tu TF/Keras 2.12 co the loi khi load bang Keras 3. De dam bao tuong thich:
- Nen dung `tensorflow==2.12.0` (khong cai them goi `keras` doc lap).
- Luu model voi `include_optimizer=False` (da cau hinh trong script). Neu van loi, load se fallback huan luyen nhanh trong `blind_sqli_exploit.py`, nhung chat luong ten sinh co the thap.

## Huan luyen GRU
`train_gru_model.py` tao vocab ky tu tu danh sach bang/cot (bo dong comment), tao chuoi ky tu cach nhau bang newline va truot cua so 20 ky tu de du doan ky tu tiep theo.

Tham so chinh:
- `SEQ_LENGTH=20`, `BATCH_SIZE=256`, `EPOCHS=50`, `VALIDATION_SPLIT=0.1`.
- Kien truc: Embedding 128 -> GRU 256 (return_sequences, dropout) -> GRU 256 (dropout) -> Dense 256 ReLU -> Dense vocab softmax.
- GPU: tu dong dung `tf.distribute.MirroredStrategy` neu phat hien GPU, set memory growth; fallback CPU.

Chay huan luyen:
```bash
python train_gru_model.py
```
Model duoc luu tai `trained_models/blind_sqli_gru_model.h5`. Neu train tren Colab, dam bao dung TF 2.12 roi tai file ve thay the file cu.

## Sinh ten bang/cot
```bash
python generate_names.py
```
- Nap model `.h5`, doc vocab tu `data/` (bo comment), sampling voi phan pho xac suat cua model, stop khi gap `\n` hoac qua `max_len` (mac dinh 20). In 10 ten.
- Co the chinh `max_len` trong ham `generate_name` neu can.

## Khai thac mau (time-based) voi GRU
`blind_sqli_exploit.py`:
- Muc tieu mac dinh: lab Flask `/time` o `http://127.0.0.1:5000/time` tham so `username`, delay ~4s (matching TIME_DELAY trong lab). Chinh `TARGET`, `PARAM`, `DELAY` neu can.
- Sinh ten bang qua GRU, tao payload UNION vao SQLite demo de tao ket qua dung -> server sleep -> do thoi gian.
- Co temperature sampling, URL-encode payload.
- Neu load model `.h5` that bai do lech phien ban, script se huan luyen nhanh (5 epoch) tu dataset va luu de dung ngay (chat luong thap hon).

Chay:
```bash
python blind_sqli_exploit.py
```
Quan sat thoi gian tra ve, neu >= `DELAY` coi la hit.

## Tich hop voi blind_sqli_demo
- Repo `blind_sqli_demo` (cung cap) co route `/gru` mo phong so sanh SQLMap baseline vs GRU (doc model tu thu muc nay). Dam bao file `trained_models/blind_sqli_gru_model.h5` ton tai de demo GRU.
- Cac route khac la cac lab blind SQLi (boolean/error/timing/oob) dung chung login surface; xem README trong thu muc `blind_sqli_demo`.

## Phap ly
Chi khai thac tren he thong/lab duoc phep. Tac gia khong chiu trach nhiem neu dung sai muc dich.
