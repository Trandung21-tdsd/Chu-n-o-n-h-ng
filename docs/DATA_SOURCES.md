# Nguồn ảnh nội soi họng / thanh quản có nhãn bệnh

Tra cứu ngày 19/11/2026. Ưu tiên vùng **họng – thanh quản**. Trước khi dùng, kiểm tra lại điều khoản trên trang gốc. Không lấy ảnh từ công cụ tìm kiếm rồi tự gán bệnh theo tên trang: nhãn phải do bác sĩ hoặc giải phẫu bệnh xác nhận.

## Tóm tắt

| Bộ dữ liệu | Vùng / loại ảnh | Nhãn | Quy mô | Truy cập | Dùng với pipeline |
|---|---|---|---|---|---|
| **CE-NBI** ([bài báo](https://www.nature.com/articles/s41597-023-02629-7)) | Thanh quản, nội soi tiếp xúc + NBI | Mô bệnh học (16 loại → 10 nhóm), lành/ác | 11.144 ảnh, 210 bệnh nhân | Mở, **đang có trên máy** | task `cenbi_10class`, `cenbi_binary` |
| **Laryngoscope8** ([GitHub](https://github.com/greenyin/Laryngoscope8)) | Thanh quản, nội soi thường | 8 lớp: Normal, Reinke's edema, Glottic cancer, Granuloma, Leukoplakia, Cyst, Nodules, Polyps | 3.057 ảnh, 1.950 người | Link Baidu (mã `soda`) được báo đã hết hạn ([issue #16](https://github.com/greenyin/Laryngoscope8/issues/16)); cần liên hệ tác giả; chưa rõ license | task `laryngoscope8` (imagefolder) |
| **ENTRep** ([trang challenge](https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep), [arXiv 2508.04801](https://arxiv.org/abs/2508.04801)) | Tai – mũi – **họng** – dây thanh, BV Thống Nhất TP.HCM | Vùng giải phẫu (Ear L/R, Nose L/R, **Throat**, VC-open, VC-closed), bình thường/bất thường, mô tả song ngữ | Bộ public + private | Link Google Drive + *Dataset Usage Guideline* trên trang challenge | Đổi annotation sang CSV → adapter `csv`. Là nhãn **giải phẫu/bất thường**, không phải chẩn đoán bệnh cụ thể |
| **Pharyngitis – Yoo 2020** ([Mendeley](https://data.mendeley.com/datasets/8ynyhnj2kz/2), [bài báo](https://pmc.ncbi.nlm.nih.gov/articles/PMC7440230)) | Họng miệng, **ảnh điện thoại** | Viêm họng / bình thường | 362 ảnh (v2) | CC BY 4.0 | task `pharyngitis_smartphone`. Ảnh lấy từ web, không có mã bệnh nhân |
| **Pharyngitis – Shojaei 2025** (Scientific Data, DOI [10.1038/s41597-025-05780-5](https://doi.org/10.1038/s41597-025-05780-5)) | Họng miệng, ảnh điện thoại | Vi khuẩn / không vi khuẩn (4–9 bác sĩ chẩn đoán), 20 triệu chứng | 742 bệnh nhân | Xem mục *Data Records* của bài báo | adapter `csv` (nhãn = đồng thuận bác sĩ, `patient_id` = mã bệnh nhân) |
| Moccia 2017 ([Zenodo](https://zenodo.org/records/1003200)) | Thanh quản NBI (patch mô) | He / Hbv / Le / IPCL | 1.320 patch, 33 người | Đã tải ở `data/external/` (pipeline cũ) | Bài toán mô/mạch máu, khác ontology bệnh |
| VoFoCD ([bài báo](https://pmc.ncbi.nlm.nih.gov/articles/PMC11612113/)) | Thanh quản, BV Chợ Rẫy | Không phát hiện / bình thường / u lành / u ác | 1.724 ảnh | **Không công khai** | — |
| Ung thư vòm họng (NPC) | Vòm mũi họng | — | Nhiều bộ lớn | **Không công khai** (dữ liệu bệnh viện) | — |
| BAGLS / BAGLS-VF | Thanh môn (video tốc độ cao) | Mask phân đoạn | — | Mở, đã có trên máy | Phân đoạn, không phải chẩn đoán |

Tổng quan về dữ liệu nội soi thanh quản và lý do khan hiếm: [Artificial intelligence for laryngoscopy in vocal fold diseases: a review of dataset, technology, and ethics](https://link.springer.com/article/10.1007/s10994-024-06602-2).

## Khuyến nghị

1. **Thanh quản (sát với dữ liệu hiện có):** xin Laryngoscope8 từ tác giả. Đây là nguồn công khai hiếm hoi có **lớp bình thường** và các bệnh dây thanh hay gặp, nhưng là ảnh ánh sáng trắng, nên cần train như **task riêng** (`laryngoscope8`), không trộn với CE-NBI.
2. **Họng miệng (viêm họng):** chỉ có ảnh điện thoại công khai (Yoo 2020, Shojaei 2025). Phù hợp cho bài toán "ảnh chụp họng → viêm họng", **không phải ảnh nội soi**. Bộ Shojaei có mã bệnh nhân và nhiều bác sĩ đánh giá, đáng tin hơn.
3. **ENTRep:** tốt để train một model **nhận diện vùng giải phẫu** (đây có phải ảnh họng/thanh quản không), dùng làm "cổng" trước model bệnh. Không suy ra bệnh từ mô tả tự do.
4. **Dữ liệu bệnh viện:** nên chuẩn bị file `labels.csv` với các cột `image_path, label, patient_id` (thêm `encounter_id`, `modality`, `anatomy`, `reference_standard` nếu có), rồi dùng task `custom_csv_example`.

## Sắp xếp dữ liệu sau khi tải

```text
data/external/Laryngoscope8/<tên_lớp>/<ảnh>.jpg          -> --task laryngoscope8
data/external/pharyngitis_yoo2020/<lớp>/<ảnh>.jpg         -> --task pharyngitis_smartphone
data/external/hospital_throat/labels.csv + ảnh            -> --task custom_csv_example
```

Nếu tên thư mục lớp khác với `display_names` trong `configs/tasks.json`, sửa lại `display_names` hoặc thêm `label_map`. Nếu bộ dữ liệu có sẵn thư mục `train/test`, đặt `class_level: 1`; pipeline luôn tự chia lại theo bệnh nhân/nhóm trùng. Khi không có mã bệnh nhân, chỉ gom được ảnh trùng/gần trùng; ảnh khác nhau của cùng một người vẫn có thể rơi vào hai tập (rủi ro leakage cần ghi rõ trong báo cáo).
