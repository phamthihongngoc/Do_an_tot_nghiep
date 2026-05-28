# Huấn luyện GAFormer trên tập dữ liệu GesHome*
# Chạy từ thư mục gốc dự án: .\gaformer\scripts\train_gaformer.ps1

$PYTHON    = "G:\DOAN2\.venv\Scripts\python.exe"
$SCRIPT    = "G:\DOAN2\gaformer\src\train.py"
$MANIFEST  = "G:\DOAN2\data_collection\data\SHG_Dataset\manifest_shg.csv"
$ROOT      = "G:\DOAN2\data_collection\data\SHG_Dataset"
$OUT_DIR   = "G:\DOAN2\gaformer\checkpoints"

& $PYTHON $SCRIPT `
    --manifest      $MANIFEST `
    --root          $ROOT `
    --results-dir   $OUT_DIR `
    --epochs        30 `
    --batch-size    16 `
    --lr            1e-4 `
    --dropout       0.1 `
    --img-size      224 `
    --smooth-window 5 `
    --seed          42
