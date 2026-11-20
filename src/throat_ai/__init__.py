"""throat_ai — pipeline nghiên cứu phân loại ảnh nội soi họng / thanh quản.

Các bước: prepare (audit + duplicate + chia tập) -> benchmark (nhiều model) ->
evaluate (chọn theo validation, test 1 lần) -> predict (chẩn đoán ảnh mới) -> tato_export.

Không phải thiết bị y tế. Xem README.md.
"""
import os as _os

# Phải đặt trước khi torch được import: op chưa hỗ trợ trên Apple MPS sẽ chạy dự phòng bằng CPU.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "2.0.0"
