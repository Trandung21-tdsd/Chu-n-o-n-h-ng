# Train 4 model trên máy Linux — GPU AMD RX570

Thư mục này dùng để train **ResNet-50, Swin-T, MobileNetV2, DenseNet-121** (chẩn đoán ảnh BAGLS-VF, 3 nhóm bệnh) trên máy
Linux có **GPU AMD RX570**, đồng thời ghi **telemetry GPU 1 Hz** (nhiệt độ, công suất, xung nhịp, % tải) cho TATO.
Code train là pipeline của dự án (chia theo video, early stopping, chọn model theo validation, benchmark CSV, xuất TATO).

> **Vì sao phải dùng Docker?** RX570 là kiến trúc Polaris (**gfx803**). ROCm và PyTorch chính thức đã **bỏ hỗ trợ
> gfx803 từ 2021**, nên `pip install torch` (kể cả bản ROCm) **không** chạy được trên RX570. Script cài đặt sẽ biên dịch
> PyTorch 2.6 + ROCm 6.4 riêng cho gfx803 theo dự án cộng đồng
> [robertrosenbusch/gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm), trong Docker. Máy Linux **không cần
> cài ROCm**, chỉ cần driver `amdgpu` (có sẵn trong kernel) và Docker. Đây là bản build cộng đồng, không phải AMD hỗ trợ
> chính thức — vì vậy có bước `kiem_tra_gpu.py` để xác nhận GPU tính **đúng** trước khi train.

---

## 1. File trong thư mục

| File | Việc |
|---|---|
| `train_may_linux.sh` | **Script chính.** Menu chọn 4 model (hoặc chạy cả 4); tự chạy bên trong Docker RX570; bật telemetry ngầm |
| `train.py` | Gọi pipeline của dự án cho 1 model (được `train_may_linux.sh` gọi) |
| `cai_moi_truong_rx570.sh` | **Cài môi trường 1 lần**: kiểm tra máy, build PyTorch gfx803 + image `throat-ai-rx570`, kiểm tra GPU |
| `Dockerfile.rx570`, `requirements_rx570.txt` | Image train = PyTorch gfx803 + thư viện của dự án (giữ nguyên bản torch gfx803) |
| `chay_trong_docker.sh` | Chạy 1 lệnh bất kỳ trong container (GPU + thư mục dự án gắn vào `/workspace`) |
| `kiem_tra_gpu.py` | Kiểm tra GPU tính đúng (so với CPU) + train thử 3 bước mỗi model |
| `run_train_with_telemetry.py`, `telemetry_logger_standalone.py` | Telemetry 1 Hz chạy song song khi train. **GPU AMD đọc qua sysfs của driver amdgpu**; GPU NVIDIA đọc qua NVML |
| `dong_goi_tu_mac.sh` | Chạy **trên máy Mac**: đóng gói code + dữ liệu (~3 GB) để copy sang Linux |
| `requirements.txt` | Thư viện của telemetry (`nvidia-ml-py` chỉ cần cho NVIDIA, `psutil`) |

---

## 2. Yêu cầu máy Linux

- **GPU**: RX570 (cũng chạy với RX470/480/580/590 — cùng gfx803). Bản **8 GB** khuyến nghị; bản 4 GB vẫn chạy (tự giảm batch).
- **Hệ điều hành**: Ubuntu 22.04 hoặc 24.04 (distro khác cũng được nếu có Docker).
- **Kernel**: tránh **6.13 và 6.14** (gfx803 bị SegFault). Theo dự án gfx803_rocm, đã chạy tốt: 6.2, 6.6, 6.8 (Ubuntu 24.04 gốc),
  6.9–6.11, ≥ 6.12.21, ≥ 6.15.8. Xem kernel: `uname -r`.
- **Mainboard** hỗ trợ **PCIe atomics** (đa số máy từ khoảng 2015 trở đi; ROCm cần).
- **RAM ≥ 16 GB**, **trống ≥ 50 GB**, Internet; **1–3 giờ** để biên dịch PyTorch (chỉ lần đầu).

---

## 3. Các bước

### Bước 1 — Trên máy Mac: đóng gói
```bash
cd "<thư mục dự án>"
python -m throat_ai.prepare --task bagls_3class --overwrite   # phân chia mới (đã loại video trùng cảnh quay với test)
./tools_may_train/dong_goi_tu_mac.sh                          # -> ~/Downloads/throat_ai_linux_<ngày>.tar.gz
```
Gói gồm: code, `configs/`, thư mục này, ảnh BAGLS-VF train/test, **chỉ file `.meta`** của archive-2 (nhãn bệnh — không
cần 17 GB ảnh gốc), `data/prepared/bagls_3class` (để máy Linux dùng **đúng phân chia** của máy Mac → kết quả 2 máy so
sánh được), trọng số pretrained đã tải (nếu có).

### Bước 2 — Trên máy Linux: giải nén
```bash
mkdir -p ~/throat_ai && tar -xzf throat_ai_linux_<ngày>.tar.gz -C ~/throat_ai
cd ~/throat_ai/tools_may_train
chmod +x *.sh
```
**Hoặc lấy code bằng git** (dữ liệu không có trên GitHub vì nằm trong `.gitignore`): trên Mac tạo gói chỉ dữ liệu
`./tools_may_train/dong_goi_tu_mac.sh --chi-du-lieu`, rồi trên Linux:
```bash
git clone https://github.com/Trandung21-tdsd/Chu-n-o-n-h-ng.git ~/throat_ai
tar -xzf throat_ai_linux_<ngày>.tar.gz -C ~/throat_ai        # thêm data/ + data/prepared vào thư mục clone
```
Đừng bỏ qua `data/prepared/bagls_3class`: thiếu nó, `train.py` sẽ tự chia lại train/val trên Linux và kết quả không
còn so sánh được với máy Mac.

### Bước 3 — Cài môi trường (1 lần)
```bash
sudo apt install -y docker.io git pciutils
sudo usermod -aG docker,video,render $USER      # rồi ĐĂNG XUẤT / khởi động lại máy
./cai_moi_truong_rx570.sh --check                # chỉ kiểm tra máy (GPU, driver, kernel, quyền, ổ đĩa)
./cai_moi_truong_rx570.sh                        # build (1–3 giờ) + kiểm tra GPU + train thử 4 model
```
Kết thúc phải thấy `[✔] Phép tính trên GPU đúng (cả chiều xuôi lẫn lan truyền ngược).` và 4 dòng
`[✔] ResNet-50 / Swin-T / MobileNetV2 / DenseNet-121`. Phần kiểm tra so GPU với CPU cho: nhân ma trận, tích chập,
BatchNorm và **gradient** (lan truyền ngược) — vì image chỉ biên dịch lại PyTorch/rocBLAS cho gfx803, còn MIOpen
(tích chập, BatchNorm) là bản ROCm 6.4 gốc.
Kiểm tra lại bất cứ lúc nào: `./chay_trong_docker.sh python3 kiem_tra_gpu.py`

Chạy thử nhanh toàn bộ luồng (1 epoch, ~vài phút) trước khi train thật:
```bash
TRAIN_EXTRA_ARGS="--epochs 1 --out ../outputs/thu_nghiem_rx570" ./train_may_linux.sh 3
```
Tuỳ chọn thêm (bộ kiểm thử của dự án, chạy trên CPU trong container): `./chay_trong_docker.sh python3 -m pytest -q ../tests`

### Bước 4 — Train
```bash
./train_may_linux.sh          # menu
./train_may_linux.sh 1        # ResNet-50
./train_may_linux.sh 5        # cả 4 model lần lượt (mỗi model 1 file telemetry riêng)
./train_may_linux.sh 1 16 FP32 2   # model, batch, precision, accumulation
```
| Số | Model | Mặc định (RX570 8 GB) |
|---|---|---|
| 1 | ResNet-50 | batch 16 × accumulation 2, FP32 |
| 2 | Swin-T | batch 16 × 2, FP32 |
| 3 | MobileNetV2 | batch 32 × 1, FP32 |
| 4 | DenseNet-121 | batch 16 × 2, FP32 |

- Batch hiệu dụng luôn **32** (= batch × accumulation) để so sánh được với ma trận chính trên máy Mac.
- **FP32** vì Polaris không tăng tốc FP16; muốn thử AMP: `./train_may_linux.sh 1 16 AMP 2`.
- RX570 **4 GB**: `train.py` tự giảm batch còn 8 (tăng accumulation) cho ResNet-50/Swin-T/DenseNet-121.
- Lần đầu mỗi model, MIOpen (thư viện tích chập của AMD) **biên dịch kernel** nên epoch đầu chậm; cache lưu ở
  `<dự án>/.cache/miopen` nên các lần sau nhanh hơn.
- Bị ngắt (mất điện, Ctrl+C): chạy lại đúng lệnh — model đã train xong được bỏ qua.
- `TRAIN_EXTRA_ARGS="..."` chuyển thêm tham số cho `train.py` / `throat_ai.benchmark` (vd. `--epochs 20 --seeds 42 43`).

---

## 4. Kết quả

| Ở đâu | Nội dung |
|---|---|
| `outputs/bagls_3class_linux_rx570/benchmark_results.csv` | Mỗi lần train 1 dòng (val/test macro-F1, epoch, thời gian, mốc `training_start_unix` / `finished_unix`) |
| `outputs/bagls_3class_linux_rx570/comparison.md`, `charts/` | Bảng so sánh + biểu đồ (gồm train loss vs val loss) |
| `outputs/bagls_3class_linux_rx570/tato/` | Dữ liệu TATO (static_profiling_dataset, online_observations...) |
| `tools_may_train/telemetry_logs/telemetry_<thời_gian>.csv` | Telemetry GPU 1 Hz: `timestamp`, nhiệt độ, công suất W, xung nhịp MHz, % tải, VRAM |

Ghép telemetry với từng lần train theo thời gian: `timestamp` (telemetry) ↔ `training_start_unix` / `finished_unix`
(benchmark_results.csv) và `timestamp_unix` từng epoch trong `runs/<run_id>/history.csv`.

**Telemetry trên GPU AMD:** đọc từ `/sys/class/drm/card*/device` của driver `amdgpu` — nhiệt độ (`temp1_input`), công suất
(`power1_average`), xung nhịp (`freq1_input` hoặc `pp_dpm_sclk`), % tải (`gpu_busy_percent`), VRAM. amdgpu **không báo cờ
throttle** như NVIDIA → cột `gpu_throttled` để **trống** (không phải 0); có thể suy ra throttle từ xung nhịp tụt khi
nhiệt độ cao.

---

## 5. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `torch.cuda.is_available() = False` | Chạy trong Docker (`./chay_trong_docker.sh ...`); user thuộc nhóm `video`, `render` (đăng xuất/đăng nhập lại); có `/dev/kfd` |
| Không có `/dev/kfd`, `lsmod` không thấy `amdgpu` | Polaris đang dùng driver `radeon` cũ: thêm `radeon.si_support=0 radeon.cik_support=0 amdgpu.si_support=1 amdgpu.cik_support=1` vào `GRUB_CMDLINE_LINUX_DEFAULT` trong `/etc/default/grub`, `sudo update-grub`, khởi động lại |
| SegFault / treo khi dùng GPU | Kernel 6.13 / 6.14 → đổi kernel (xem mục 2) |
| `kiem_tra_gpu.py` báo **kết quả sai** | Không train (kết quả sẽ vô nghĩa). Đổi kernel, build lại image; xem issue của [gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm/issues) |
| Hết bộ nhớ (`out_of_memory` trong benchmark_results.csv) | Giảm batch, tăng accumulation: `./train_may_linux.sh 2 8 FP32 4` |
| Build PyTorch bị kill (hết RAM) | Thêm swap 8–16 GB, hoặc sửa `MAX_JOBS` trong `~/gfx803_rocm/Dockerfile_rocm64_base` xuống 4–6 |
| Không dùng được GPU | Dự phòng CPU (rất chậm): `./cai_moi_truong_rx570.sh --cpu` |
| Telemetry trống cột nhiệt độ/công suất | Chạy `ls /sys/class/drm/card*/device/hwmon/` — nếu không có hwmon thì driver không xuất cảm biến |
