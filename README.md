# GRU Method for Blind SQLi

Mo hinh GRU sinh ten bang/cot pho bien de ho tro khai thac blind SQLi (time-based). Y tuong lay tu paper `Base_paper/A SQL Blind Injection Method Based on Gated Recurrent Neural Network.pdf`. Repo cung cap toan bo pipeline: tien xu ly danh sach ten, huan luyen GRU ky tu, sinh payload time-based va kich ban demo. Chi dung cho muc dich hoc tap/kiem thu duoc uy quyen.

## Cau truc thu muc
- `data/`: danh sach ten bang (`common-tables.txt`) va cot (`common-columns.txt`) trich tu sqlmap (giang trong dong comment ban quyen).
- `train_gru_model.py`: huan luyen model GRU ky tu (seed, split, callbacks, luu config/vocab/history).
- `trained_models/`: artefact sau train (`blind_sqli_gru_best.h5`, `blind_sqli_gru_final.keras`, `blind_sqli_gru_final.weights.h5`, `vocab.json`, `config.json`, `training_log.csv`, `training_history.png`, `final_results.json`). Ban dau co the trong neu chua train.
- `generate_names.py`: script ngan nap model cu (`blind_sqli_gru_model.h5`) va sinh 10 ten thu nghiem (khong dung vocab JSON moi).
- `blind_sqli_exploit.py`: nap model tot nhat + vocab/config, sinh ten bang/cot va bắn payload time-based len dich.
- `test_lib.py`: kiem tra GPU TensorFlow.
- `Base_paper/`: bai bao goc.
- `requirements.txt`: phu thuoc (TF 2.19 + Keras 3.12).

## Moi truong
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Kiem tra GPU: `python test_lib.py`.

> Ghi chu tuong thich: model `.h5` duoc luu voi TensorFlow 2.19/Keras 3.12. Neu dung phien ban thap hon (vi du 2.12) hoac cao hon, Keras co the can file `.keras`/`.weights.h5` va vocab/config di kem. Script exploit da doc dung vocab/config de tranh sai lech index.

## Du lieu va tien xu ly (train_gru_model.py)
- Doc `common-tables.txt` + `common-columns.txt`, bo dong comment, lowercase, noi them END_TOKEN `*` vao cuoi moi ten de model hoc dau ket thuc.
- Xay dung vocab ky tu theo tan suat (chi so thap hon = ky tu pho bien hon), chen `PAD_TOKEN=<PAD>`, `UNK_TOKEN=<UNK>` dau danh sach. Luu mapping vao `trained_models/vocab.json`.
- Chuoi huan luyen: ghep cac ten bang newline, trich cua so truot do dai `SEQ_LENGTH=5`, bo qua cua so bi END_TOKEN de tranh hoc tiep chuoi qua ten khac.
- Tach tap: 80% train, 10% val, 10% test. Tao `tf.data.Dataset` shuffle/prefetch.

## Kien truc & tham so
- Embedding 256 chieu (mask padding) -> GRU 512 (return_sequences, dropout 0.3, recurrent_dropout 0.2, L2) -> BatchNorm -> GRU 512 -> BatchNorm -> Dense 512 ReLU -> Dropout -> BatchNorm -> Dense 256 ReLU -> Dropout -> Dense vocab softmax.
- Toi uu hoa: Adam lr 1e-3, clipnorm 1.0; cosine LR schedule + ReduceLROnPlateau backup; EarlyStopping (patience 20) + ModelCheckpoint luu `trained_models/blind_sqli_gru_best.h5`; CSVLogger.
- Seed co dinh 42; su dung `tf.distribute.MirroredStrategy` neu co GPU, bat memory growth.
- Hyperparams chinh: `BATCH_SIZE=64`, `EPOCHS=200`, `VALIDATION_SPLIT=0.1`, `TEST_SPLIT=0.1`, dropout 0.3, L2 1e-4.

## Huan luyen
```bash
python train_gru_model.py
```
Trong qua trinh train:
- Log ra kich thuoc vocab, chi so END token, thong ke mau (top ky tu pho bien).
- Callbacks tu dong luu checkpoint tot nhat (`blind_sqli_gru_best.h5`) va ve bieu do `training_history.png`.
- Danh gia tren tap test, tinh perplexity (exp cross-entropy) theo paper, thu generate mau voi nhieu seed/temperature. Ket qua da train san: `test_loss` 2.2469, `test_accuracy` 0.4876, `top5` 0.8056, `perplexity` ~5.99 (`trained_models/final_results.json`).
- Luu model cuoi o 2 dang: `blind_sqli_gru_final.keras` va `blind_sqli_gru_final.weights.h5` + `config.json` (seq_length, dropout, vocab_size...).

## Sinh ten (demo thu)
- Cach chuan (dung vocab/config dong nhat voi model): dung cac ham `generate_text` hoac `test_generation` trong `train_gru_model.py` sau khi train xong, hoac dung `blind_sqli_exploit.py` (da doc vocab.json).
- Script ngan san co `generate_names.py` (su dung file `trained_models/blind_sqli_gru_model.h5` cu, vocab tu data/). Chay: `python generate_names.py`. Script nay khong doc vocab.json, chi phu hop khi model duoc train cung vocab mac dinh.

## Khai thac time-based (blind_sqli_exploit.py)
- Nap `trained_models/blind_sqli_gru_best.h5` + `vocab.json` + `config.json` de dam bao mapping ky tu dung (END_TOKEN index, PAD/UNK). Co kiem tra file ton tai truoc khi chay.
- Sinh ten bang/cot bang temperature sampling + top-k, seed pho bien (`us, ad, pa, id, na...`), dung chuoi dem do dai toi da 20 ky tu, dung END_TOKEN `*` lam ky tu dung.
- Payload mau (SQLite demo):
  - V1: `UNION SELECT 1,'<table>','p','r' FROM sqlite_master WHERE name='<table>' -- -`
  - V2: `CASE WHEN (SELECT name FROM sqlite_master WHERE type='table' AND name='<table>') ...`
  - V3 (dang dung): `admin' OR (SELECT COUNT(*) FROM <table>) > 0 -- -`
  - Payload cot: `admin' OR (SELECT <column> FROM <table> LIMIT 1) IS NOT NULL -- -`
- Gui request GET den `TARGET` (mac dinh `http://127.0.0.1:5000/time`, param `username`) va do thoi gian. Neu thoi gian >= `DELAY_THRESHOLD=4s` thi coi la co bang/cot. Script in log dang `[idx/total] <name> -> 4.12s  DELAY` va tong ket danh sach nghi ngo.
- Flow mac dinh: sinh mau ten bang, hien thi sanity check, cho bam Enter de bat dau fuzz; neu tim thay bang thi hoi tiep tuc fuzz cot.

Chay khai thac:
```bash
python blind_sqli_exploit.py
```
Thay doi `TARGET`, `PARAM`, `DELAY_THRESHOLD` cho phu hop dich lab.

## Tep sinh ra sau train
- `trained_models/blind_sqli_gru_best.h5`: checkpoint val_loss tot nhat, dung cho exploit.
- `trained_models/blind_sqli_gru_final.keras` + `blind_sqli_gru_final.weights.h5`: snapshot cuoi.
- `trained_models/vocab.json`: mapping ky tu -> chi so (co END token index).
- `trained_models/config.json`: sieu tham so (seq_length, dropout, vocab_size...).
- `trained_models/training_log.csv`, `training_history.png`: log lich su train.
- `trained_models/final_results.json`: so lieu test + perplexity.

## Luu y bao mat/phap ly
Chi thu nghiem tren dich vu/lab duoc uy quyen. Tac gia khong chiu trach nhiem neu su dung sai muc dich.***
