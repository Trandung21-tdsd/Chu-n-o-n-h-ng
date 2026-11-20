#!/usr/bin/env bash
# ==========================================================================
#  Cài PyTorch cho GPU AMD RX570 (gfx803) KHÔNG cần Docker, KHÔNG cần sudo.
#
#  Dùng ROCm 5.7.x ĐÃ CÀI SẴN trên máy (/opt/rocm-5.7.*) — bản ROCm cuối còn chạy được Polaris/gfx803.
#  Biên dịch PyTorch 2.3 + TorchVision 0.18 cho gfx803 theo cách của dự án cộng đồng
#  https://github.com/robertrosenbusch/gfx803_rocm57_pt23 (bản đó chạy trong Docker; script này làm y vậy trên máy).
#  Mọi thứ nằm trong thư mục dự án (nên để trên ổ lớn, vd. /data):
#      .rx570-venv/    môi trường Python có PyTorch gfx803 (+ thư viện của dự án)
#      .build-rx570/   mã nguồn + bản build (~20–30 GB, xoá được sau khi cài xong)
#
#  Cách dùng (chạy trong tmux — biên dịch 2–4 giờ):
#    ./cai_khong_docker_rx570.sh --check     chỉ kiểm tra máy
#    ./cai_khong_docker_rx570.sh             kiểm tra + biên dịch + cài + kiểm tra GPU
#  Sau đó ./train_may_linux.sh tự dùng môi trường .rx570-venv (không cần Docker).
# ==========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
ENV_DIR="${RX570_ENV:-$PROJECT/.rx570-venv}"
BUILD_DIR="${RX570_BUILD:-$PROJECT/.build-rx570}"
ROCM="${ROCM_PATH:-$(ls -d /opt/rocm-5.7* 2>/dev/null | sort -V | tail -1 || true)}"
TORCH_BRANCH="release/2.3"
VISION_BRANCH="release/0.18"
MODE="${1:-}"

ok()   { echo "  [✔] $*"; }
warn() { echo "  [!] $*"; }
fail() { echo "  [✖] $*"; FAILED=1; }
FAILED=0

echo "== 1. Kiểm tra máy (không cần sudo)"
# ROCm 5.7
if [ -n "$ROCM" ] && [ -d "$ROCM" ]; then
    ok "ROCm: $ROCM (phiên bản $(cat "$ROCM/.info/version" 2>/dev/null || echo '?'))"
else
    fail "Không thấy ROCm 5.7 ở /opt/rocm-5.7* (đặt ROCM_PATH=<thư mục ROCm> nếu cài chỗ khác)"
    ROCM=/nonexistent
fi
[ -x "$ROCM/bin/hipcc" ] && ok "hipcc có" || fail "Thiếu $ROCM/bin/hipcc (ROCm cài thiếu gói phát triển HIP — nhờ quản lý máy cài 'rocm-hip-sdk')"
if [ -x "$ROCM/bin/rocminfo" ]; then
    ROCMINFO_OUT="$("$ROCM/bin/rocminfo" 2>&1 || true)"   # lưu ra biến: grep -q + pipefail sẽ báo nhầm
    if grep -q gfx803 <<<"$ROCMINFO_OUT"; then ok "ROCm nhìn thấy GPU gfx803 (RX570)"; else
        fail "rocminfo không thấy gfx803 — user có trong nhóm render/video chưa? (id -nG), /dev/kfd có quyền chưa?"
    fi
fi
# Thư viện ROCm mà PyTorch cần khi biên dịch
MISSING=""
for lib in librocblas libhipblas libMIOpen libhipsparse libhipfft librocfft librocrand libhiprand libhipsolver librocsolver; do
    ls "$ROCM"/lib/${lib}.so* >/dev/null 2>&1 || MISSING="$MISSING $lib"
done
for inc in rocprim hipcub thrust; do
    [ -d "$ROCM/include/$inc" ] || MISSING="$MISSING include/$inc"
done
[ -z "$MISSING" ] && ok "Đủ thư viện ROCm cho PyTorch" || fail "Thiếu thư viện ROCm:$MISSING (nhờ quản lý máy cài 'rocm-libs' / 'rocm-hip-sdk' 5.7)"
ls "$ROCM"/lib/libhipblaslt.so* >/dev/null 2>&1 && ok "hipBLASLt có" || warn "Không có hipBLASLt (không bắt buộc với gfx803 — sẽ tắt)"
if ls "$ROCM"/lib/rocblas/library/*gfx803* >/dev/null 2>&1; then ok "rocBLAS có kernel gfx803"; else
    warn "rocBLAS không có file kernel riêng cho gfx803 — phép nhân ma trận dùng đường dự phòng; kiem_tra_gpu.py sẽ xác nhận đúng/sai"
fi
# Trình biên dịch + công cụ
for tool in gcc g++ git curl make; do
    command -v "$tool" >/dev/null 2>&1 && ok "$tool: $($tool --version 2>/dev/null | head -1 | cut -c1-60)" || fail "Thiếu $tool"
done
# Python có header (Python.h) để biên dịch
PY_SYS="$(command -v python3.12 || command -v python3 || true)"
if [ -n "$PY_SYS" ] && "$PY_SYS" -c "import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'],'Python.h')) else 1)" 2>/dev/null; then
    PY_BUILD="$PY_SYS"; ok "Python để biên dịch: $PY_BUILD ($($PY_SYS --version 2>&1)) có Python.h"
else
    PY_BUILD=""; warn "Python hệ thống thiếu Python.h (gói python3-dev) -> sẽ tải Python 3.12 bản độc lập bằng uv (không cần sudo)"
fi
# RAM / CPU / ổ đĩa
RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
CPUS="$(nproc)"
JOBS="${MAX_JOBS:-$(( RAM_GB / 3 < CPUS ? RAM_GB / 3 : CPUS ))}"
[ "$JOBS" -ge 1 ] || JOBS=1
ok "RAM ${RAM_GB} GB, ${CPUS} CPU -> biên dịch song song MAX_JOBS=$JOBS (đặt MAX_JOBS=... để đổi)"
[ "$RAM_GB" -ge 12 ] || warn "RAM ít (< 12 GB): biên dịch sẽ chậm; nếu bị 'Killed' hãy chạy lại với MAX_JOBS=2"
mkdir -p "$BUILD_DIR"
FREE_GB="$(df -BG --output=avail "$BUILD_DIR" | tail -1 | tr -dc '0-9')"
[ "$FREE_GB" -ge 40 ] && ok "Trống ${FREE_GB} GB tại $(df --output=target "$BUILD_DIR" | tail -1)" \
    || fail "Chỉ trống ${FREE_GB} GB tại $BUILD_DIR (cần >= 40 GB) — đặt dự án trên /data"
# Dữ liệu
[ -f "$PROJECT/data/prepared/bagls_3class/manifest.csv" ] && ok "Có dữ liệu + phân chia (data/prepared/bagls_3class)" \
    || warn "Chưa có data/prepared/bagls_3class — giải nén gói dữ liệu từ Mac trước khi train"

if [ "$FAILED" != 0 ]; then echo; echo "Còn lỗi [✖] ở trên — sửa xong rồi chạy lại."; exit 1; fi
if [ "$MODE" = "--check" ]; then echo; echo "Máy đạt yêu cầu. Chạy tiếp (trong tmux): ./cai_khong_docker_rx570.sh"; exit 0; fi

echo; echo "== 2. Môi trường Python: $ENV_DIR"
if [ ! -x "$ENV_DIR/bin/python" ]; then
    if [ -z "$PY_BUILD" ]; then
        command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        uv python install 3.12
        PY_BUILD="$(uv python find 3.12)"
    fi
    "$PY_BUILD" -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
# PyTorch 2.3 cần numpy 1.x; CMake 4 không đọc được vài thư viện con cũ của PyTorch 2.3
cat > "$BUILD_DIR/constraints.txt" <<'EOF'
numpy<2
cmake>=3.27,<4
setuptools<75
EOF
pip install -q --upgrade pip wheel
pip install -q -c "$BUILD_DIR/constraints.txt" setuptools cmake ninja numpy pyyaml typing_extensions sympy filelock \
    networkx jinja2 fsspec packaging astunparse requests

export ROCM_PATH="$ROCM" HIP_PATH="$ROCM" ROCM_HOME="$ROCM"
export PATH="$ROCM/bin:$PATH" CMAKE_PREFIX_PATH="$ROCM${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PYTORCH_ROCM_ARCH=gfx803 ROCM_ARCH=gfx803 HSA_OVERRIDE_GFX_VERSION=8.0.3 ROC_ENABLE_PRE_VEGA=1
export USE_ROCM=1 USE_CUDA=0 USE_NINJA=1 FORCE_CUDA=1 MAX_JOBS="$JOBS"
# Tắt phần không cần cho train 1 GPU gfx803 (ít lỗi + biên dịch nhanh hơn)
export BUILD_TEST=0 USE_DISTRIBUTED=0 USE_KINETO=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_ROCM_KERNEL_ASSERT=0

echo; echo "== 3. Biên dịch PyTorch ($TORCH_BRANCH) cho gfx803 — 2–4 giờ"
if python -c "import torch, sys; sys.exit(0 if '$TORCH_BRANCH'.split('/')[1] in torch.__version__ and torch.version.hip else 1)" 2>/dev/null; then
    ok "Đã có PyTorch ROCm trong môi trường — bỏ qua (xoá $ENV_DIR để cài lại)"
else
    [ -d "$BUILD_DIR/pytorch/.git" ] || git clone --recursive --depth 1 -b "$TORCH_BRANCH" \
        https://github.com/pytorch/pytorch.git "$BUILD_DIR/pytorch"
    cd "$BUILD_DIR/pytorch"
    pip install -q -c "$BUILD_DIR/constraints.txt" -r requirements.txt
    [ -f .rx570_hipified ] || { python tools/amd_build/build_amd.py >/dev/null && touch .rx570_hipified; }
    python setup.py bdist_wheel
    pip install --force-reinstall --no-deps dist/torch-*.whl
fi

echo; echo "== 4. Biên dịch TorchVision ($VISION_BRANCH)"
if python -c "import torchvision" 2>/dev/null; then ok "Đã có TorchVision — bỏ qua"; else
    [ -d "$BUILD_DIR/vision/.git" ] || git clone --depth 1 -b "$VISION_BRANCH" https://github.com/pytorch/vision.git "$BUILD_DIR/vision"
    cd "$BUILD_DIR/vision"
    python setup.py bdist_wheel
    pip install --no-deps dist/torchvision-*.whl
fi

echo; echo "== 5. Thư viện của dự án (giữ nguyên torch/torchvision vừa biên dịch)"
pip list --format=freeze | grep -iE '^(torch|torchvision|numpy)==' > "$BUILD_DIR/locked.txt"
pip install -q -c "$BUILD_DIR/locked.txt" -r "$HERE/requirements_rx570.txt"

# Biến môi trường ROCm tự đặt mỗi khi kích hoạt .rx570-venv
if ! grep -q "RX570 (gfx803)" "$ENV_DIR/bin/activate"; then
    cat >> "$ENV_DIR/bin/activate" <<EOF

# --- RX570 (gfx803) + ROCm 5.7 (thêm bởi cai_khong_docker_rx570.sh) ---
export ROCM_PATH="$ROCM"
export PATH="$ROCM/bin:\$PATH"
export LD_LIBRARY_PATH="$ROCM/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export HSA_OVERRIDE_GFX_VERSION=8.0.3 ROC_ENABLE_PRE_VEGA=1 TORCH_BLAS_PREFER_HIPBLASLT=0 MIOPEN_LOG_LEVEL=3
EOF
fi
deactivate
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo; echo "== 6. Kiểm tra GPU + train thử 4 model"
cd "$HERE"
python kiem_tra_gpu.py
echo; echo "XONG. Train: cd $HERE && ./train_may_linux.sh   (tự dùng $ENV_DIR, không cần Docker)"
echo "Có thể xoá $BUILD_DIR để lấy lại ổ đĩa (giữ lại nếu muốn biên dịch lại nhanh hơn)."
