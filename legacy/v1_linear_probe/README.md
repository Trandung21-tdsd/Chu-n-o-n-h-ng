# Pipeline cũ (v1, 16/11/2026) — chỉ để tham khảo

Phiên bản cũ dùng encoder ImageNet **đóng băng + linear head** (ResNet-18, MobileNetV3-Small) và có giao diện Streamlit (`app.py` đã bị xóa theo yêu cầu). Kết quả test ở mức bệnh nhân đạt macro-F1 khoảng 0,16–0,19 (xem `reports/RESULTS.md` ở thư mục gốc).

Pipeline mới (`src/throat_ai/`) thay thế toàn bộ các file ở đây: fine-tune toàn bộ mạng, 9+ kiến trúc, audit file nhãn, 3 chính sách duplicate, telemetry và xuất dữ liệu cho TATO.

Các file ở đây **không còn chạy được** cùng package mới (ví dụ chúng import `throat_ai.model`). Có thể xóa thư mục này nếu không cần nữa.

| File | Trước đây ở |
|---|---|
| `README_v1.md` | `README.md` |
| `src/*.py` | `src/throat_ai/` |
| `tests/test_pipeline.py` | `tests/` |
| `scripts/*.py` | `scripts/` |
| `streamlit_config/config.toml` | `.streamlit/` |
| `requirements-lock.txt` | thư mục gốc |
