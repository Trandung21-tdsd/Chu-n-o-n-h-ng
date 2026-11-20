#!/usr/bin/env bash
# ==========================================================================
#  Chạy TRÊN MÁY MAC: đóng gói code + dữ liệu cần để train trên máy Linux thành 1 file .tar.gz (~3 GB).
#  archive-2 (17 GB) chỉ lấy file .meta (nhãn bệnh) — không cần ảnh/video gốc.
#  Kèm data/prepared/bagls_3class để máy Linux dùng ĐÚNG phân chia train/val/test của máy Mac.
#
#    ./tools_may_train/dong_goi_tu_mac.sh                      # -> ~/Downloads/throat_ai_linux_<ngày>.tar.gz
#    ./tools_may_train/dong_goi_tu_mac.sh /Volumes/USB/goi.tar.gz
#    ./tools_may_train/dong_goi_tu_mac.sh --chi-du-lieu         # chỉ dữ liệu (code lấy bằng git clone trên Linux)
#
#  Trên Linux:  mkdir -p ~/throat_ai && tar -xzf throat_ai_linux_<ngày>.tar.gz -C ~/throat_ai
# ==========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
ONLY_DATA=0
if [ "${1:-}" = "--chi-du-lieu" ]; then ONLY_DATA=1; shift; fi
OUT="${1:-$HOME/Downloads/throat_ai_linux_$(date +%Y%m%d).tar.gz}"
cd "$PROJECT"

for d in data/BAGLS-VF_train data/BAGLS-VF_test data/archive-2; do
    [ -d "$d" ] || { echo "[!] Thiếu $d"; exit 1; }
done
if ! grep -q '"same_scene_video_pairs"' data/prepared/bagls_3class/prepare_summary.json 2>/dev/null; then
    echo "[!] data/prepared/bagls_3class chưa có hoặc là bản CŨ (trước khi loại video trùng cảnh quay với test)."
    echo "    Chạy trước:  python -m throat_ai.prepare --task bagls_3class --overwrite"
    exit 1
fi

# Trọng số pretrained ImageNet đã tải trên Mac -> máy Linux khỏi tải lại (train.py đặt TORCH_HOME=<dự án>/.cache/torch)
mkdir -p .cache/torch/hub/checkpoints
for pattern in resnet50- swin_t- mobilenet_v2- densenet121-; do
    for f in "$HOME"/.cache/torch/hub/checkpoints/${pattern}*.pth; do
        [ -f "$f" ] && cp -n "$f" .cache/torch/hub/checkpoints/ 2>/dev/null || true
    done
done

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
{
    [ "$ONLY_DATA" = 1 ] || printf '%s\n' src/throat_ai configs tools_may_train pyproject.toml requirements.txt README.md
    printf '%s\n' data/BAGLS-VF_train data/BAGLS-VF_test data/prepared/bagls_3class .cache/torch
    find data/archive-2 -name '*.meta'
} > "$LIST"

echo "Đang đóng gói $(wc -l < "$LIST" | tr -d ' ') mục -> $OUT (vài phút)..."
COPYFILE_DISABLE=1 tar -czf "$OUT" --no-xattrs --no-mac-metadata \
    --exclude '__pycache__' --exclude '.DS_Store' --exclude '._*' \
    --exclude 'tools_may_train/telemetry_logs' \
    -T "$LIST"
echo "Xong: $OUT ($(du -h "$OUT" | cut -f1))"
echo "Copy sang máy Linux rồi:  mkdir -p ~/throat_ai && tar -xzf $(basename "$OUT") -C ~/throat_ai"
