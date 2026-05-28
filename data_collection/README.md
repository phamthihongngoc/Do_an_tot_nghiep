# Gesture Data Collection ESP32 + MPU6050

Du an thu du lieu gia toc va van toc goc tu ESP32 + MPU6050/GY-521 o tan so 100 Hz.
Giao dien chinh hien tai la Streamlit.

## 1. Cai moi truong

```powershell
cd G:\DOAN2\data_collection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Chay giao dien Streamlit

Cach nhanh tren Windows:

```text
G:\DOAN2\data_collection\run_streamlit.bat
```

Hoac chay bang PowerShell:

```powershell
cd G:\DOAN2\data_collection
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Mo trinh duyet tai:

```text
http://localhost:8501
```

Neu Chrome bao `ERR_CONNECTION_REFUSED`, nghia la Streamlit server chua chay hoac terminal chay Streamlit da bi dong.

## 3. Quy trinh thu bang Streamlit

1. Chon `Data source`: `ESP32 Serial`.
2. Chon dung `Serial port`, vi du `COM5`.
3. Bam `Connect`, sau do co the bam `Ping` de kiem tra.
4. Nhap `Subject ID`, vi du `S01`.
5. Chon nhan `G1-G15` hoac `N1-N5`.
6. Dat `Collection duration` khoang 3-5 giay.
7. Dat `Number of repeats` khoang 30-50 lan cho moi nhan cua moi nguoi.
8. Giu bat `Save each trial to data/raw`.
9. Bam `Start Collection`.

Moi lan lap se tu luu thanh mot file CSV trong:

```text
data/raw/<Subject ID>/<Gesture ID>/
```

Dong thoi metadata duoc ghi vao:

```text
data/manifest.csv
```

Nut `Download CSV` chi dung de tai trial moi nhat khi can kiem tra nhanh.
Nut `Download ZIP` tai toan bo batch vua thu, giu dung cau truc thu muc `data/raw/S01/G1/`, `data/raw/S01/G2/`, ...

Moi file CSV mau chi gom cac cot dung cho huan luyen:

```text
time,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,label
```

Trong do `time` la thoi gian tinh bang giay tu luc bat dau trial, va `label` la ma nhan nhu `G1`, `G2`, `N1`.

## 4. Bo nhan hien tai

| Cu chi | Ten nhan |
| --- | --- |
| G1 | star_first |
| G2 | select_wrist_rotate |
| G3 | tv_favorite_chanel |
| G4 | tv_swtich_source |
| G5 | tv_channel_up |
| G6 | tv_channel_down |
| G7 | tv_voice_search |
| G8 | speaker_volume_up |
| G9 | speaker_volume_down |
| G10 | light_on |
| G11 | light_off |
| G12 | curtain_close |
| G13 | curtain_open |
| G14 | stop_palm |
| G15 | emergency_reset |
| N1 | noise_walking |
| N2 | noise_watch |
| N3 | noise_typing |
| N4 | noise_drinking |
| N5 | noise_scratch |

Theo anh ban gui: 15 nhan cu chi tay va 5 nhan nhieu.

## 5. Cong cu dong lenh du phong

Liet ke cong Serial:

```powershell
python .\collector\collect_serial.py --list-ports
```

Liet ke nhan:

```powershell
python .\collector\collect_serial.py --list-labels
```

Thu mot nhan:

```powershell
python .\collector\collect_serial.py --port COM5 --subject S01 --label G1 --repeats 10 --duration 4
```

Thu toan bo nhan `G1-G15` va `N1-N5`:

```powershell
python .\collector\collect_serial.py --port COM5 --subject S01 --label ALL --repeats 10 --duration 4
```

## 6. Goi y so luong mau

Cho do an, cau hinh nen dung:

- 15 nguoi tham gia: `S01-S15`.
- 20 nhan: `G1-G15` va `N1-N5`.
- Moi nhan moi nguoi: 30-50 lan lap.
- Moi lan lap: 3-5 giay.

Tong so mau neu thu 50 lan lap:

```text
15 nguoi x 20 nhan x 50 lan = 15,000 file mau
```

Khi chia train/test, nen chia theo nguoi de danh gia kha nang tong quat hoa:

```text
Train: 11 nguoi
Validation: 2 nguoi
Test: 2 nguoi
```
