# Smart Home Gesture Demo 50Hz

Demo dieu khien nha thong minh bang cu chi IMU. Demo mac dinh dung bo model
50Hz da train trong `G:\DOAN2\training_50hz_clean\checkpoints`.
Web demo doc du lieu truc tiep tu ESP32 + MPU6050 qua Serial, hien bieu do
gia toc/con quay hoi chuyen va doc thong bao bang text-to-speech khi lenh
dieu khien duoc chap nhan.

## Cau truc

```text
demo/
  predict.py            # CLI/module du doan tu CSV hoac mang mau IMU
  backend/app.py        # FastAPI server, REST API, WebSocket live Serial
  web/index.html        # Giao dien web: phong 2D ben trai, bieu do IMU ben phai
  requirements.txt
```

## Cai dat

```powershell
G:\DOAN2\.venv\Scripts\python.exe -m pip install -r G:\DOAN2\demo\requirements.txt
```

## Chay backend va web

```powershell
G:\DOAN2\demo\run_demo.ps1
```

Hoac chay truc tiep:

```powershell
G:\DOAN2\.venv\Scripts\python.exe -m uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000
```

Script se in URL dang dung, vi du `http://127.0.0.1:8000/`. Neu cong 8000
dang ban, script tu chuyen sang 8001, 8002, ...
Script tu mo URL bang Coc Coc neu tim thay `browser.exe`. Neu may cai Coc Coc o
vi tri khac, dat bien moi truong truoc khi chay:

```powershell
$env:DEMO_BROWSER_PATH = "C:\Program Files\CocCoc\Browser\Application\browser.exe"
G:\DOAN2\demo\run_demo.ps1
```

Mac dinh backend tu tim Transformer moi nhat trong:

```text
G:\DOAN2\training_50hz_clean\checkpoints\transformer_100epoch
```

## Chon model khac

```powershell
$env:DEMO_MODEL = "lstm"
$env:DEMO_RUN_DIR = "G:\DOAN2\training_50hz_clean\checkpoints\lstm_100epoch\lstm_20260518_222258"
G:\DOAN2\.venv\Scripts\python.exe -m uvicorn demo.backend.app:app --port 8000
```

Model Transformer:

```powershell
$env:DEMO_MODEL = "transformer"
$env:DEMO_RUN_DIR = "G:\DOAN2\training_50hz_clean\checkpoints\transformer_100epoch\transformer_20260519_050409"
G:\DOAN2\.venv\Scripts\python.exe -m uvicorn demo.backend.app:app --port 8000
```

Model Random Forest chi hien san sang trong menu khi run co file
`model.joblib`. Cac run RF cu chi co `meta.json`/confusion matrix nen can train
lai bang `training\random_forest.py` hoac `training_50hz_clean\src\random_forest.py`
sau ban cap nhat nay.

## Cau hinh 50Hz

Demo mac dinh:

```powershell
$env:DEMO_TARGET_LEN = "201"
$env:DEMO_SAMPLE_RATE_HZ = "50"
$env:DEMO_WINDOW_SECONDS = "4.0"
$env:DEMO_PREDICT_INTERVAL_SECONDS = "3.5"
$env:DEMO_ACTION_COOLDOWN_SECONDS = "3.5"
$env:DEMO_CONFIDENCE_THRESHOLD = "0.50"
$env:DEMO_MIN_GYRO_RMS_DPS = "6.0"
$env:DEMO_MIN_ACCEL_DYNAMIC_G = "0.035"
$env:DEMO_CONTEXT_G2_OVER_STOP_MIN_CONFIDENCE = "0.005"
$env:DEMO_GESTURE_CAPTURE_SECONDS = "3.5"
$env:DEMO_GESTURE_RECOVERY_SECONDS = "0.8"
$env:DEMO_STOP_CONFIRM_SECONDS = "6.0"
$env:DEMO_STOP_CONFIDENCE_THRESHOLD = "0.995"
```

Live WebSocket phat hien chuyen dong tu gia toc/con quay, thu mot doan cu chi
khoang 3-4 giay, sau do moi dua vao model va chi phat voice khi action duoc
chap nhan. Co che nay tranh viec predict/voice lap lien tuc moi 1 giay.
Cu chi G14 can duoc thuc hien 2 lan lien tiep trong 6 giay moi tat he thong,
de tranh model nhan nham G14 lam dung demo giua chung.
Khi he thong dang active, neu model dua G14 len top-1 nhung G2 co trong top-k
thi backend uu tien G2 de tranh nham thao tac chon thiet bi thanh tat he thong.

Model duoc train voi 201 diem/4 giay. Neu ESP32 van stream 100Hz, backend se
lay cua so 4 giay roi resample ve 201 diem truoc khi dua vao model.

## REST API

| Method | Path                 | Mo ta |
|--------|----------------------|-------|
| GET    | `/health`            | Thong tin model, run_dir, target_len |
| GET    | `/labels`            | Danh sach nhan va metadata |
| GET    | `/devices`           | Trang thai thiet bi gia lap va lich su |
| GET    | `/serial/ports`      | Liet ke cong COM kha dung |
| POST   | `/predict`           | JSON `{ "samples": [[ax,ay,az,gx,gy,gz], ...] }` |
| POST   | `/predict/csv`       | Upload mot trial CSV |
| POST   | `/predict/serial`    | Doc Serial 4 giay va du doan |
| WS     | `/ws/serial`         | Stream IMU realtime, predict dinh ky, cap nhat nha 2D |
| POST   | `/devices/{device}/{op}` | Test dieu khien thiet bi thu cong |

Moi action duoc chap nhan co them truong `message`; web dung truong nay de
hien toast va doc TTS, vi du den bat/tat, rem mo/dong, chon thiet bi, bat/tat
che do dieu khien.

## Logic kich ban

Mapping demo bam theo `data_collection\labels.json` va bang thong bao yeu cau:

```text
G1        -> HE THONG DA DUOC MO, mac dinh target TV
G2        -> Chon target tiep theo: TV -> Loa -> Den -> Rem -> TV
G3        -> DOI KENH
G4        -> CHON NGUON HDMI/TV
G5        -> TANG KENH
G6        -> GIAM KENH
G7        -> MO TIM KIEM BANG GIONG NOI TREN YOUTUBE
G8        -> TANG AM LUONG LOA
G9        -> GIAM AM LUONG LOA
G10       -> DA BAT DEN
G11       -> DA TAT DEN
G12       -> REM DA DONG
G13       -> REM DA MO
G14       -> DA TAT HE THONG
G15       -> RESET LAI HE THONG
N1-N5 -> Noise, khong dieu khien thiet bi
```

Khi chua Start, cac lenh dieu khien G3-G13 bi bo qua. Sau Start, G1 mac dinh
chon TV; moi lan G2 se xoay target theo thu tu TV -> Loa -> Den -> Rem -> TV.
Lenh khac muc tieu dang chon se bi tu choi voi `unexpected_sequence`.

## Test nhanh bang CSV 50Hz

```powershell
G:\DOAN2\.venv\Scripts\python.exe G:\DOAN2\demo\predict.py `
  --run-dir G:\DOAN2\training_50hz_clean\checkpoints\cnn_100epoch\cnn_20260518_215132 `
  --model cnn `
  --csv G:\DOAN2\data_collection\data\processed_50hz\S01\G1\20260514_084740_S01_G1_T001.csv
```

## Demo voi ESP32/MPU6050 qua Serial

1. Nap firmware `data_collection\firmware\esp32_mpu6050_logger\esp32_mpu6050_logger.ino`.
2. Dong Arduino Serial Monitor/Serial Plotter.
3. Chay backend.
4. Mo web demo, nhap cong COM, vi du `COM5`, bam `Connect`.
5. Web se hien 2 bieu do realtime: accelerometer va gyroscope.
6. Moi 4 giay backend cat cua so mau, dua qua model da train, roi cap nhat nha 2D.

Goi API truc tiep:

```powershell
$body = @{
  port = "COM5"
  baud = 115200
  duration_s = 4.0
  calibrate = $false
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict/serial `
  -ContentType "application/json" -Body $body
```

Input Serial dung format firmware hien co:

```text
DATA,millis_ms,sample_index,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_c
```
