# Hand Gesture Recognition for Smart Home Control
### Using Wrist-Worn IMU Sensor (MPU6050 + ESP32)

> **Đồ án tốt nghiệp** — Trường Đại học Đại Nam  
> Sinh viên: **Phạm Thị Hồng Ngọc** (MSV: 1671020226)  
> GVHD: *(Tên giảng viên hướng dẫn)*

---

## Tổng quan hệ thống

Hệ thống nhận dạng **15 cử chỉ tay** (G1–G15) và **5 chuyển động nhiễu** (N1–N5) từ cảm biến IMU đeo cổ tay, điều khiển thiết bị nhà thông minh theo thời gian thực.

```
Cảm biến IMU  ──►  Tiền xử lý  ──►  Mô hình DL  ──►  Điều khiển Smart Home
(MPU6050+ESP32)    (50Hz, 4s)       (20 lớp)        (TV / Loa / Đèn / Rèm)
```

---

## Bộ dữ liệu — SHG_Dataset

| Thông số | Giá trị |
|---|---|
| Tổng số mẫu | 10,644 trials |
| Số người tham gia | 15 subjects (S01–S15) |
| Nhãn cử chỉ | 15 gestures (G1–G15) |
| Nhãn nhiễu | 5 noise actions (N1–N5) |
| Tần số mẫu | 50 Hz (resample từ ~100 Hz) |
| Độ dài mỗi mẫu | 4 giây = 201 timesteps |
| Kích thước input | (201, 6) — 3 trục gia tốc + 3 trục con quay |
| Tập huấn luyện | 7,882 (S01–S11) |
| Tập kiểm tra | 1,432 (S12–S13) / 1,330 (S14–S15) |

**Danh sách cử chỉ:**

| Ký hiệu | Mô tả | Ký hiệu | Mô tả |
|---|---|---|---|
| G1 | START — Nắm tay | G9 | Giảm âm lượng (V xuống) |
| G2 | SELECT — Xoay cổ tay | G10 | Bật đèn (vòng tròn CCW) |
| G3 | Kênh yêu thích TV | G11 | Tắt đèn (vòng tròn CW) |
| G4 | Đổi nguồn TV | G12 | Đóng rèm (cung phải) |
| G5 | Kênh tiếp theo | G13 | Mở rèm (cung trái) |
| G6 | Kênh trước đó | G14 | STOP — Lòng bàn tay mở |
| G7 | Tìm kiếm giọng nói (S) | G15 | RESET — Hình chữ Z |
| G8 | Tăng âm lượng (V lên) | N1–N5 | Đi bộ / Xem đồng hồ / Gõ phím / Uống nước / Gãi đầu |

---

## Kiến trúc các mô hình

![IEEE Architecture Diagram](images/model_architectures.png)

### Kết quả so sánh (Test set: S14–S15)

| Mô hình | Accuracy | F1-score | Params | Inference |
|---|---|---|---|---|
| CNN1D | 96.79% | 96.75% | 56K | 1.21 ms |
| CNN-BiLSTM | 99.64% | 99.62% | 641K | 3.35 ms |
| **Transformer** ★ | **99.95%** | **99.94%** | 153K | 2.67 ms |
| GAFormer | 99.62% | 99.64% | 20.5M | 4.11 ms |
| Random Forest | 100% | 100% | — | — |

> ★ **Transformer** được chọn cho hệ thống demo — cân bằng tốt nhất giữa độ chính xác, tốc độ và kích thước mô hình.

---

## Cấu trúc dự án

```
Do_an/
├── data_collection/          # Thu thập dữ liệu từ cảm biến
│   ├── collector/
│   │   └── collect_serial.py # Đọc serial từ ESP32
│   ├── streamlit_app.py      # Giao diện thu thập dữ liệu
│   └── resample_to_50hz.py   # Resample về 50 Hz
│
├── train_4_model/            # Huấn luyện CNN1D, CNN-BiLSTM, Transformer
│   ├── training/
│   │   ├── models.py         # Định nghĩa 3 mô hình DL
│   │   ├── dataset.py        # SHGDataset + DataLoader
│   │   ├── augmentation.py   # Jitter/Scale/TimeWarp/MagnitudeWarp/ChannelDropout
│   │   ├── trainer.py        # Vòng lặp huấn luyện + Early Stopping
│   │   ├── models_rf.py      # Random Forest (Late Fusion)
│   │   └── features.py       # Trích xuất đặc trưng thống kê
│   ├── demo_models/          # Các model đã train (*.pt)
│   └── figures/              # Biểu đồ kết quả huấn luyện
│
├── gaformer/                 # GAFormer — GADF + CoAtNet-light
│   └── src/
│       ├── model.py          # GAFormer architecture
│       ├── gadf.py           # Gramian Angular Difference Field transform
│       ├── train.py          # Training script
│       ├── baselines.py      # GASF/GADF + ResNet50/CoAtNet-0 baselines
│       └── dataset.py        # Dataset cho GAFormer
│
├── demo/                     # Demo điều khiển Smart Home
│   ├── backend/
│   │   └── app.py            # FastAPI + WebSocket server
│   └── predict.py            # Inference pipeline + FSM
│
├── scripts/                  # Utility scripts
│   ├── train_dl.py           # Train tất cả DL models
│   ├── train_rf_latefusion.py# Train ML models
│   ├── evaluate.py           # Đánh giá tổng hợp
│   ├── preprocess.py         # Tiền xử lý dataset
│   └── export_best.py        # Export model tốt nhất
│
├── visualization/            # Công cụ vẽ biểu đồ
├── images/                   # Hình ảnh, sơ đồ kiến trúc
├── config.yaml               # Cấu hình toàn hệ thống
├── requirements_training.txt # Dependencies
└── colab_training.ipynb      # Notebook training trên Google Colab
```

---

## Cài đặt

### Yêu cầu

- Python 3.10+
- PyTorch 2.x
- CUDA (tùy chọn, để tăng tốc training)

### Cài đặt dependencies

```bash
git clone https://github.com/phamthihongngoc/Do_an_tot_nghiep.git
cd Do_an_tot_nghiep

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements_training.txt
```

---

## Sử dụng

### 0. Tải bộ dữ liệu SHG_Dataset

> Dataset (~163 MB) không được lưu trực tiếp trên GitHub do giới hạn dung lượng.  
> Tải về tại: **[Google Drive — SHG_Dataset.zip](https://drive.google.com/)**  
> Giải nén vào thư mục: `data_collection/data/SHG_Dataset/`

```
data_collection/data/
└── SHG_Dataset/
    ├── S01/  G1/ G2/ ... G15/ N1/ ... N5/
    ├── S02/
    ...
    └── S15/
```

### 1. Thu thập dữ liệu

```bash
# Chạy giao diện thu thập dữ liệu (Streamlit)
streamlit run data_collection/streamlit_app.py
```

Kết nối ESP32 qua cổng COM, chọn cử chỉ và ghi dữ liệu. Dữ liệu raw lưu dưới dạng CSV.

```bash
# Resample dữ liệu về 50 Hz
python data_collection/resample_to_50hz.py
```

### 2. Huấn luyện mô hình

```bash
# Train CNN1D, CNN-BiLSTM, Transformer
python scripts/train_dl.py --config config.yaml

# Train Random Forest, SVM, XGBoost
python scripts/train_rf_latefusion.py --config config.yaml

# Train GAFormer
python gaformer/src/train.py
```

**Hoặc dùng Google Colab:**
Mở `colab_training.ipynb` trên [Google Colab](https://colab.research.google.com/).

### 3. Đánh giá

```bash
python scripts/evaluate.py --model transformer --checkpoint train_4_model/demo_models/transformer_20260525_202104/best.pt
```

### 4. Chạy Demo Smart Home

```bash
# Backend FastAPI
python demo/backend/app.py

# Mở trình duyệt: http://localhost:8000
```

Kết nối ESP32, thực hiện cử chỉ → hệ thống nhận dạng real-time và điều khiển thiết bị mô phỏng.

![Demo UI](images/Giao_dien_mo_phong_dieu_khien_thiet_bi.png)

---

## Finite State Machine (FSM)

Hệ thống sử dụng FSM 4 trạng thái để điều khiển luồng nhận dạng:

```
IDLE ──(G1)──► ACTIVE ──(G14)──► STOP_PENDING ──(timeout 6s)──► IDLE
                  │                                    │
                  └──(G15 RESET)──► EMERGENCY ─────────┘
```

| Trạng thái | Mô tả |
|---|---|
| IDLE | Chờ cử chỉ kích hoạt G1 (nắm tay) |
| ACTIVE | Nhận dạng và điều khiển thiết bị |
| STOP_PENDING | Nhận G14, chờ xác nhận dừng |
| EMERGENCY | G15 RESET, khởi động lại hệ thống |

---

## Phần cứng

![Sơ đồ phần cứng](images/So_do_khoi_thiet_bi_deo_tay.png)

| Thành phần | Mô tả |
|---|---|
| **MPU6050** | Cảm biến IMU 6 trục (3-axis accel + 3-axis gyro) |
| **ESP32-WROOM-32** | Vi điều khiển WiFi/BT, đọc I2C từ MPU6050 |
| Giao tiếp | I2C (SDA/SCL), tần số ~100 Hz → resample 50 Hz |

---

## Tăng cường dữ liệu

| Kỹ thuật | Mô tả |
|---|---|
| **Jitter** | Thêm nhiễu Gaussian ngẫu nhiên |
| **Scale** | Nhân biên độ với hệ số ngẫu nhiên |
| **TimeWarp** | Co/giãn trục thời gian cục bộ |
| **MagnitudeWarp** | Biến dạng biên độ cục bộ theo spline |
| **ChannelDropout** | Bỏ ngẫu nhiên 1 kênh cảm biến |

---

## Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Label Smoothing | ε = 0.05 |
| Early Stopping | patience = 15 |
| Batch size | 32 |
| Augmentation prob | 0.5 (mỗi kỹ thuật) |

---

## Kết quả trực quan

| Confusion Matrix | t-SNE |
|---|---|
| ![Confusion Matrix](train_4_model/figures/fig_confusion_transformer.png) | ![t-SNE](train_4_model/figures/fig_tsne_transformer.png) |

---

## Tác giả

**Phạm Thị Hồng Ngọc**  
Sinh viên Đại học Đại Nam — MSV: 1671020226  
Email: phamnogc887@gmail.com  
GitHub: [phamthihongngoc](https://github.com/phamthihongngoc)

---

## Giấy phép

Dự án phục vụ mục đích học thuật — Đồ án tốt nghiệp Đại học Đại Nam, 2026.
