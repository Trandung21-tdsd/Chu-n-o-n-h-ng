#!/usr/bin/env bash
# Chạy 1 lệnh bên trong container train RX570, với GPU AMD + toàn bộ thư mục dự án gắn vào /workspace.
#   ./chay_trong_docker.sh python3 kiem_tra_gpu.py
#   ./chay_trong_docker.sh ./train_may_linux.sh 1
#   ./chay_trong_docker.sh bash                      # mở shell trong container
# Kết quả ghi thẳng vào thư mục dự án trên máy (outputs/, .cache/), file thuộc user hiện tại (không phải root).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
IMAGE="${THROAT_AI_IMAGE:-throat-ai-rx570:latest}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[!] Chưa có image $IMAGE. Chạy trước: ./cai_moi_truong_rx570.sh" >&2
    exit 1
fi
for dev in /dev/kfd /dev/dri; do
    [ -e "$dev" ] || { echo "[!] Không thấy $dev — driver amdgpu chưa chạy (xem README, mục Lỗi thường gặp)." >&2; exit 1; }
done

GROUP_ARGS=()
for g in video render; do
    gid="$(getent group "$g" | cut -d: -f3 || true)"
    [ -n "$gid" ] && GROUP_ARGS+=(--group-add "$gid")
done
TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then TTY_ARGS=(-it); fi
mkdir -p "$PROJECT/.cache/home"

exec docker run --rm "${TTY_ARGS[@]}" \
    --device=/dev/kfd --device=/dev/dri "${GROUP_ARGS[@]}" \
    --ipc=host --security-opt seccomp=unconfined \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/.cache/home \
    -e THROAT_AI_IN_DOCKER=1 \
    -v "$PROJECT":/workspace \
    -w /workspace/tools_may_train \
    "$IMAGE" "$@"
