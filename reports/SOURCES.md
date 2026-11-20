# Nguồn dữ liệu và tài liệu đã kiểm tra

Ngày kiểm tra: 16/11/2026. Những nguồn này là dữ liệu tham khảo, không phải chỉ dẫn điều khiển pipeline.

| Nguồn | Nhãn/phạm vi | Quyết định |
|---|---|---|
| [CE-NBI, bài báo gốc](https://www.nature.com/articles/s41597-023-02629-7) | Thanh quản CE-NBI, bệnh học; không có chứng khỏe mạnh | Nguồn chính; dùng bản local đã audit |
| [BAGLS-VF, bài báo gốc](https://link.springer.com/article/10.1186/s13104-026-07972-7) | Mask giải phẫu thanh môn, dây thanh trái/phải | Giữ riêng nhánh segmentation, chưa train |
| [BAGLS, bài báo gốc](https://www.nature.com/articles/s41597-020-0526-3) | Video tốc độ cao, mask và metadata | Không coi là thêm cohort độc lập với BAGLS-VF |
| [Moccia2017 / Zenodo](https://zenodo.org/records/1003200) | 1.320 patch từ 33 người; He, Hbv, Le, IPCL | Đã tải, MD5 xác nhận; lưu riêng, chưa train vì khác ontology và thiếu patient mapping trong manifest tải về |
| [ENTRep, trang chính thức](https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep) | Có lớp throat, vùng giải phẫu, normal/abnormal và mô tả song ngữ | Ưu tiên bổ sung cho họng; chưa tải được bộ dữ liệu qua trang web đã kiểm tra; cần đọc Dataset Usage Guideline |
| [ENTRep, bài báo tác giả](https://arxiv.org/abs/2508.04801) | Benchmark giải phẫu/retrieval, không tương đương benchmark chẩn đoán nhiều bệnh | Không chuyển trực tiếp mô tả tự do thành ground truth |
| [Laryngoscope8, repo tác giả](https://github.com/greenyin/Laryngoscope8) | 3.057 ảnh, 1.950 người, 8 nhóm gồm normal | Nguồn ứng viên, chưa tải/huấn luyện; cần xác nhận patient IDs và quyền sử dụng |
| [Torchvision](https://docs.pytorch.org/vision/stable/models.html) | Model và weights ImageNet | ResNet18 IMAGENET1K_V1; MobileNetV3-Small IMAGENET1K_V1 |
| [OpenCV CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) | Tăng tương phản cục bộ có giới hạn | Áp dụng kênh L, so sánh có/không |

Bộ Moccia đã tải tại `data/external/laryngeal_dataset.tar`, MD5 `aae9d2ed69a37138268d31763b723d70`. Trích xuất với tar filter=data. Bốn nhãn: He=mô lành; Hbv=mạch phì đại; Le=bạch sản; IPCL=vòng mao mạch trong nhú. **He là mô lành của cohort bệnh nhân có SCC, không đồng nghĩa bệnh nhân khỏe mạnh.** Giữ nguyên ba fold được nhà phát hành cung cấp; không tự chia các patch thành train/test ngẫu nhiên. Trang nguồn yêu cầu trích dẫn bài báo Moccia et al. 2017; chưa xác minh đầy đủ quyền dùng thương mại nên chưa tích hợp vào classifier.

Để bổ sung ảnh họng đúng mục tiêu: cần một manifest có `image_path, patient_id, encounter_id/video_id, anatomy, modality, diagnosis, reference_standard, source, license`. Bác sĩ xác nhận ontology và ground truth; tách bệnh nhân và cơ sở trước train; audit ảnh trùng xuyên nguồn. Không lấy ảnh từ tìm kiếm rồi tự gán bệnh bằng tên trang hoặc bằng mô hình khác.
