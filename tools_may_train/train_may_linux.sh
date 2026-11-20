#!/bin/bash
# ==========================================================================
#  TATO — Script gộp Train AI và Ghi Telemetry GPU trên máy Linux
#  Hỗ trợ 4 kiến trúc mục tiêu: ResNet-50, Swin-T, MobileNetV2, DenseNet-121
#  GPU AMD RX570 (gfx803): tự chạy bên trong Docker throat-ai-rx570 (cài bằng ./cai_moi_truong_rx570.sh).
#
#  ./train_may_linux.sh                    menu chọn model
#  ./train_may_linux.sh 1                  ResNet-50 với cấu hình mặc định
#  ./train_may_linux.sh 5                  chạy lần lượt cả 4 model
#  ./train_may_linux.sh 1 16 FP32 2        model, batch, precision, accumulation (batch hiệu dụng = 16 x 2)
#  NO_DOCKER=1 ./train_may_linux.sh 3      chạy thẳng bằng python3 của máy (NVIDIA/CPU, không dùng Docker)
#  TRAIN_EXTRA_ARGS="--epochs 1" ./train_may_linux.sh 3     thêm tham số cho train.py (vd. chạy thử 1 epoch)
# ==========================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Ngoài Docker + có image RX570 -> chạy lại chính script này bên trong container (GPU AMD cần PyTorch gfx803).
if [ -z "${THROAT_AI_IN_DOCKER:-}" ] && [ -z "${NO_DOCKER:-}" ]; then
    if command -v docker >/dev/null 2>&1 && docker image inspect "${THROAT_AI_IMAGE:-throat-ai-rx570:latest}" >/dev/null 2>&1; then
        exec "$HERE/chay_trong_docker.sh" ./train_may_linux.sh "$@"
    fi
    echo "[!] Chưa có image Docker throat-ai-rx570 -> chạy bằng python3 của máy (GPU AMD RX570 sẽ KHÔNG được dùng)."
    echo "    Cài môi trường RX570: ./cai_moi_truong_rx570.sh"
fi

CHOICE="${1:-}"

if [ -z "$CHOICE" ]; then
    echo "=================================================================="
    echo "   TATO — CHỌN MÔ HÌNH HUẤN LUYỆN (Linux Training Node, GPU AMD RX570)"
    echo "=================================================================="
    echo "  1) ResNet-50    (CNN tiêu chuẩn, 25.6M params, 4.12 GFLOPs)"
    echo "  2) Swin-T       (Vision Transformer, 28.3M params, 4.50 GFLOPs)"
    echo "  3) MobileNetV2  (Mô hình nhẹ, 3.5M params, 0.30 GFLOPs)"
    echo "  4) DenseNet-121 (Dense connections, 8.0M params, 2.88 GFLOPs)"
    echo "  5) Chạy lần lượt cả 4 model"
    echo "=================================================================="
    read -p "Nhập số [1-5] (Mặc định: 1): " CHOICE_INPUT
    CHOICE="${CHOICE_INPUT:-1}"
fi

# Mặc định cho RX570 8 GB: FP32 (Polaris không tăng tốc FP16), batch hiệu dụng 32 = batch x accumulation.
# Bản 4 GB: train.py tự giảm batch xuống 8 (tăng accumulation) cho 3 model nặng.
config_for() {
    case "$1" in
        1|resnet50|ResNet-50|resnet)                        MODEL_NAME="ResNet-50";    DEF_BATCH=16; DEF_ACCUM=2 ;;
        2|swin|Swin-T|swin_t|swin-tiny)                     MODEL_NAME="Swin-T";       DEF_BATCH=16; DEF_ACCUM=2 ;;
        3|mobilenet|MobileNetV2|mobilenetv2|mobilenet_v2)   MODEL_NAME="MobileNetV2";  DEF_BATCH=32; DEF_ACCUM=1 ;;
        4|densenet|DenseNet-121|densenet121|densenet_121)   MODEL_NAME="DenseNet-121"; DEF_BATCH=16; DEF_ACCUM=2 ;;
        *)                                                  MODEL_NAME="$1";           DEF_BATCH=16; DEF_ACCUM=2 ;;
    esac
}

run_one() {
    config_for "$1"
    BATCH_SIZE="${2:-$DEF_BATCH}"
    PRECISION="${3:-FP32}"
    ACCUM="${4:-$DEF_ACCUM}"
    echo ""
    echo "[*] Mô hình đã chọn : $MODEL_NAME"
    echo "[*] Batch Size       : $BATCH_SIZE x accumulation $ACCUM"
    echo "[*] Precision        : $PRECISION"
    echo ""

    # LỆNH TRAIN: pipeline của dự án (train.py cùng thư mục) — kết quả ở outputs/bagls_3class_linux_rx570/
    LENH_TRAIN="python3 '$HERE/train.py' --model '$MODEL_NAME' --batch_size $BATCH_SIZE --accumulation $ACCUM --precision $PRECISION ${TRAIN_EXTRA_ARGS:-}"

    # File telemetry_*.csv ghi vào tools_may_train/telemetry_logs/; lệnh train chạy từ tools_may_train/
    # (đường dẫn tương đối trong TRAIN_EXTRA_ARGS, vd. --out ../outputs/x, tính từ tools_may_train/).
    mkdir -p "$HERE/telemetry_logs"
    (cd "$HERE/telemetry_logs" && python3 "$HERE/run_train_with_telemetry.py" \
        --cmd "cd '$HERE' && $LENH_TRAIN" \
        --model "$MODEL_NAME" \
        --batch_size "$BATCH_SIZE" \
        --precision "$PRECISION")
}

if [ "$CHOICE" = "5" ] || [ "$CHOICE" = "all" ]; then
    STATUS=0
    for m in 1 2 3 4; do
        run_one "$m" || { echo "[!] Model $m lỗi — chạy tiếp model sau."; STATUS=1; }
    done
    exit $STATUS
fi
run_one "$CHOICE" "${2:-}" "${3:-}" "${4:-}"
