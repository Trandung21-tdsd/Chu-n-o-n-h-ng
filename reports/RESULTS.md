# Kết quả benchmark đã chạy

Đã hoàn thành 8 runs, seed 42, frozen ImageNet backbone + linear head. Model được chọn bằng validation patient macro-F1, không theo test.

| Model | CLAHE | Bỏ trùng train | Val patient F1 | Test patient F1 | Test patient accuracy | Best epoch / epochs |
|---|---|---|---|---|---|---|
| resnet18 | False | False | 0.2846 | 0.1933 | 0.2903 | 1 / 11 |
| resnet18 | False | True | 0.2751 | 0.1933 | 0.2903 | 1 / 11 |
| resnet18 | True | False | 0.2980 | 0.1758 | 0.2903 | 1 / 11 |
| resnet18 | True | True | 0.2776 | 0.1867 | 0.3226 | 1 / 11 |
| mobilenet_v3_small | False | False | 0.3497 | 0.1616 | 0.2903 | 1 / 11 |
| mobilenet_v3_small | False | True | 0.3603 | 0.1833 | 0.3226 | 6 / 16 |
| mobilenet_v3_small | True | False | 0.3930 | 0.1904 | 0.3548 | 3 / 13 |
| mobilenet_v3_small | True | True | 0.3930 | 0.1904 | 0.3548 | 3 / 13 |

Lựa chọn: MobileNetV3-Small + CLAHE, giữ duplicate train (hòa macro-F1 validation với cấu hình bỏ duplicate, chọn run xuất hiện trước theo thứ tự cố định). Chỉ 4 bản trùng cùng nhãn còn lại trong train; chênh lệch này quá nhỏ để suy ra xử lý duplicate không quan trọng. 54 ảnh trùng có nhãn xung đột luôn bị cách ly trong cả hai nhánh.

Test: 31 bệnh nhân / 1.578 ảnh. Cấu hình chọn đạt 11/31 bệnh nhân đúng (35,48%); image accuracy 28,45%, image macro-F1 0,1992. Kết quả còn thấp; một số lớp không được nhận diện đúng. Không đủ cơ sở dùng chẩn đoán lâm sàng.

Các kiểm tra đã thực hiện: 5 unit tests; kiểm tra cô lập patient/group/hash và mọi cặp pHash; đối chiếu checkpoint tải lại với dự đoán benchmark; từ chối modality ngoài phạm vi; Streamlit AppTest không có exception.

Giới hạn: một seed, 31 bệnh nhân test, chưa fine-tuning backbone, chưa external validation hoặc confidence calibration. Test này đã được xem; mọi phát triển tiếp sau là exploratory trên split này. Cần một tập external độc lập trước kết luận cuối.

Bước tiếp theo để cải thiện: xác minh nhãn với bác sĩ và xem các cặp gần trùng; thu thập đúng ảnh họng và ảnh bình thường với bệnh nhân độc lập; fine-tune CNN theo patient-group cross-validation trên train/validation; kiểm thử domain khác trên external test. PDF kỹ năng khám không bù được thiếu nhãn ảnh.
