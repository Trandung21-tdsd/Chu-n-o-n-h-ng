#!/usr/bin/env bash
# ==========================================================================
#  Cài môi trường train cho GPU AMD RX570 (Polaris / gfx803) trên Linux — làm 1 lần.
#
#  Vì sao phải dùng Docker: ROCm và PyTorch chính thức đã BỎ hỗ trợ gfx803 (RX470/480/570/580/590) từ 2021,
#  `pip install torch` bản ROCm KHÔNG chạy được RX570. Script này build PyTorch 2.6 + ROCm 6.4 biên dịch riêng
#  cho gfx803 theo dự án cộng đồng https://github.com/robertrosenbusch/gfx803_rocm, rồi cài thêm thư viện của
#  dự án lên trên (image throat-ai-rx570). Máy chủ KHÔNG cần cài ROCm, chỉ cần driver amdgpu (có sẵn trong kernel) + Docker.
#
#  Cần: Ubuntu 22.04/24.04 (hoặc distro tương tự), RAM >= 16 GB, trống >= 50 GB, mạng Internet,
#       1–3 giờ để biên dịch (chỉ lần đầu).
#
#  Cách dùng:
#    ./cai_moi_truong_rx570.sh            # kiểm tra máy + build image + chạy kiểm tra GPU
#    ./cai_moi_truong_rx570.sh --check    # chỉ kiểm tra máy, không build
#    ./cai_moi_truong_rx570.sh --cpu      # KHÔNG dùng GPU: tạo venv PyTorch CPU (dự phòng, train rất chậm)
# ==========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
GFX803_DIR="${GFX803_DIR:-$HOME/gfx803_rocm}"
BASE_TAG="rocm6_gfx803_base:6.4"
PYTORCH_TAG="rocm64_gfx803_pytorch:2.6"
IMAGE="${THROAT_AI_IMAGE:-throat-ai-rx570:latest}"
MODE="${1:-}"

ok()   { echo "  [✔] $*"; }
warn() { echo "  [!] $*"; }
fail() { echo "  [✖] $*"; FAILED=1; }
FAILED=0

# --------------------------------------------------------------------------- dự phòng: CPU
if [ "$MODE" = "--cpu" ]; then
    echo "== Tạo môi trường CPU (không dùng GPU) tại $PROJECT/.venv-cpu"
    python3 -m venv "$PROJECT/.venv-cpu"
    "$PROJECT/.venv-cpu/bin/pip" install --upgrade pip
    "$PROJECT/.venv-cpu/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    "$PROJECT/.venv-cpu/bin/pip" install -r "$HERE/requirements_rx570.txt" -r "$HERE/requirements.txt"
    echo "Xong. Dùng: source $PROJECT/.venv-cpu/bin/activate && NO_DOCKER=1 THROAT_AI_ALLOW_CPU=1 ./train_may_linux.sh 3"
    exit 0
fi

echo "== 1. Kiểm tra máy"
# GPU
if command -v lspci >/dev/null 2>&1; then
    GPU_LINE="$(lspci -nn | grep -iE 'vga|display|3d' | grep -i '1002:' || true)"
    if grep -qiE 'ellesmere|polaris|rx 4[78]0|rx 5[789]0|67df|67ef' <<<"$GPU_LINE"; then
        ok "GPU AMD Polaris: $(echo "$GPU_LINE" | head -1 | cut -c1-110)"
    elif [ -n "$GPU_LINE" ]; then
        warn "GPU AMD nhưng không phải Polaris/RX570: $(echo "$GPU_LINE" | head -1 | cut -c1-110) — image gfx803 có thể không hợp."
    else
        fail "Không thấy GPU AMD (lspci). Kiểm tra card đã cắm / driver."
    fi
else
    warn "Không có lspci (cài pciutils) — bỏ qua kiểm tra tên GPU."
fi
# Driver amdgpu + thiết bị ROCm
# /sys/module/amdgpu có cả khi amdgpu là module lẫn khi build sẵn trong kernel (lsmod có thể không có với user thường)
DRIVER_IN_USE="$(lspci -k 2>/dev/null | grep -iA3 -E 'vga|display|3d' | grep -i 'driver in use' | head -1 | sed 's/.*: //' || true)"
LSMOD_OUT="$(lsmod 2>/dev/null || true)"
if [ -d /sys/module/amdgpu ] || grep -q '^amdgpu' <<<"$LSMOD_OUT" || [ "$DRIVER_IN_USE" = "amdgpu" ]; then
    ok "Driver amdgpu đang chạy${DRIVER_IN_USE:+ (GPU dùng driver: $DRIVER_IN_USE)}"
elif [ "$DRIVER_IN_USE" = "radeon" ] || [ -d /sys/module/radeon ]; then
    fail "GPU đang dùng driver 'radeon' cũ, không phải amdgpu. Thêm vào GRUB_CMDLINE_LINUX_DEFAULT
         'radeon.si_support=0 radeon.cik_support=0 amdgpu.si_support=1 amdgpu.cik_support=1' rồi sudo update-grub && reboot"
else
    fail "Không thấy driver amdgpu (lspci -k: '${DRIVER_IN_USE:-không rõ}'). Máy có đúng là có card AMD không? Xem: lspci -k | grep -iA3 vga"
fi
[ -e /dev/kfd ] && ok "/dev/kfd có" || fail "Thiếu /dev/kfd (ROCm không truy cập được GPU)"
[ -d /dev/dri ] && ok "/dev/dri có" || fail "Thiếu /dev/dri"
# Kernel (theo bảng thử nghiệm của gfx803_rocm: 6.13 và 6.14 bị SegFault)
KERNEL="$(uname -r)"
KMAJMIN="$(echo "$KERNEL" | cut -d. -f1-2)"
case "$KMAJMIN" in
    6.13|6.14) fail "Kernel $KERNEL: gfx803 bị SegFault với 6.13/6.14 — dùng kernel 6.8 (Ubuntu 24.04 gốc), 6.6, 6.11 hoặc >= 6.15.8" ;;
    *) ok "Kernel $KERNEL" ;;
esac
# PCIe atomics (ROCm cần)
if ls /sys/class/kfd/kfd/topology/nodes/*/io_links/0/properties >/dev/null 2>&1; then
    ok "Thông tin PCIe của GPU (ROCm cần mainboard hỗ trợ PCIe atomics; nếu kiem_tra_gpu.py báo lỗi, xem dòng này):"
    grep -h flags /sys/class/kfd/kfd/topology/nodes/*/io_links/0/properties 2>/dev/null | sed 's/^/        /' || true
fi
# Nhóm quyền
ME="$(id -un)"
for g in video render; do
    if grep -qx "$g" <<<"$(id -nG "$ME" | tr ' ' '\n')"; then ok "User thuộc nhóm $g"; else
        fail "User chưa thuộc nhóm $g -> sudo usermod -aG video,render $ME  rồi ĐĂNG XUẤT/khởi động lại"
    fi
done
# Docker
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then ok "Docker chạy được"; else
    fail "Chưa có Docker hoặc user chưa có quyền -> sudo apt install -y docker.io git && sudo usermod -aG docker $ME  (đăng xuất/đăng nhập lại)"
fi
# RAM / ổ đĩa
RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
[ "$RAM_GB" -ge 15 ] && ok "RAM ${RAM_GB} GB" || warn "RAM ${RAM_GB} GB < 16 GB: biên dịch PyTorch có thể hết bộ nhớ (giảm MAX_JOBS, thêm swap)"
DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
FREE_GB="$(df -BG --output=avail "$(dirname "$DOCKER_ROOT")" 2>/dev/null | tail -1 | tr -dc '0-9')"
[ -n "$FREE_GB" ] && { [ "$FREE_GB" -ge 50 ] && ok "Trống ${FREE_GB} GB cho Docker" || warn "Chỉ trống ${FREE_GB} GB (< 50 GB) cho Docker"; }
# Dữ liệu
for d in data/BAGLS-VF_train data/BAGLS-VF_test data/archive-2; do
    [ -d "$PROJECT/$d" ] && ok "Có $d" || warn "Chưa có $PROJECT/$d (copy dữ liệu sang trước khi train — xem README)"
done
[ -f "$PROJECT/data/prepared/bagls_3class/manifest.csv" ] && ok "Có phân chia dữ liệu từ máy Mac (data/prepared/bagls_3class)" \
    || warn "Chưa có data/prepared/bagls_3class — train.py sẽ tự chia lại trên máy này (khác máy Mac)"

if [ "$FAILED" != 0 ]; then
    echo; echo "Còn lỗi ở trên — sửa xong rồi chạy lại script này."; exit 1
fi
[ "$MODE" = "--check" ] && { echo; echo "Máy đạt yêu cầu."; exit 0; }

echo; echo "== 2. Build PyTorch cho gfx803 (lần đầu 1–3 giờ; đã có image thì bỏ qua)"
if [ ! -d "$GFX803_DIR/.git" ]; then
    git clone https://github.com/robertrosenbusch/gfx803_rocm.git "$GFX803_DIR"
fi
if docker image inspect "$BASE_TAG" >/dev/null 2>&1; then ok "Đã có $BASE_TAG"; else
    (cd "$GFX803_DIR" && docker build -f Dockerfile_rocm64_base . -t "$BASE_TAG")
fi
if docker image inspect "$PYTORCH_TAG" >/dev/null 2>&1; then ok "Đã có $PYTORCH_TAG"; else
    (cd "$GFX803_DIR" && docker build -f Dockerfile_rocm64_pytorch . -t "$PYTORCH_TAG")
fi

echo; echo "== 3. Cài thư viện của dự án lên image ($IMAGE)"
docker build --build-arg BASE_IMAGE="$PYTORCH_TAG" -f "$HERE/Dockerfile.rx570" -t "$IMAGE" "$HERE"

echo; echo "== 4. Kiểm tra GPU + train thử 4 model"
"$HERE/chay_trong_docker.sh" python3 kiem_tra_gpu.py

chmod +x "$HERE"/*.sh
echo; echo "XONG. Train: cd $HERE && ./train_may_linux.sh"
