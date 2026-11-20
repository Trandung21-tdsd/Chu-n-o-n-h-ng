"""Dấu hiệu nội soi hạ họng – thanh quản theo bài giảng "Kĩ năng khám Tai Mũi Họng" (Bộ môn TMH – ĐH Y Dược
TP.HCM, sinh viên Y5): tr.27 phân vùng họng, tr.33–36 khám & hình soi hạ họng + thanh quản bình thường,
tr.37 các bệnh lý thường gặp (hạt dây thanh, u hạt mấu thanh, u nhú thanh quản, ung thư thanh quản).
Mô tả của các bệnh không có trong bài giảng (polyp, nang, phù Reinke, liệt...) ghi rõ là "y văn bổ sung".

Đây là mô tả THAM KHẢO để đối chiếu với ảnh. Model chỉ học nhóm bệnh từ ảnh, KHÔNG tự phát hiện từng dấu hiệu.
"""
from __future__ import annotations

PDF = "bài giảng Kĩ năng khám TMH"

# ------------------------------------------------------------------------------ theo nhóm (nhãn model dự đoán)
GROUP_SIGNS: dict[str, list[str]] = {
    "Normal": [
        f"[{PDF}, tr.35] Mốc giải phẫu cân đối: nắp thanh môn, băng thanh thất (dây giả), dây thanh âm (dây thật), "
        "thanh thất, sụn phễu (củ chêm, củ sừng), xoang lê hai bên.",
        f"[{PDF}, tr.35] Hít vào: hai dây thanh mở, thanh môn hình tam giác; phát âm: hai dây thanh khép kín đều.",
        "Dây thanh trắng ngà, bờ tự do thẳng, nhẵn; không thấy khối sùi, nốt hay mảng trắng.",
    ],
    "Organic_lesion": [
        f"[{PDF}, tr.37] Có tổn thương nhìn thấy trên dây thanh, ví dụ: hạt dây thanh (nốt nhỏ đối xứng hai bên), "
        "u hạt mấu thanh (khối ở mấu thanh âm phía sau), u nhú thanh quản (sùi dạng chùm), ung thư thanh quản "
        "(sùi loét, bờ không đều).",
        "Bờ tự do dây thanh không còn thẳng nhẵn; khi phát âm hai dây thanh khép không kín quanh tổn thương "
        "(khe hở hình đồng hồ cát).",
    ],
    "Functional": [
        "Không thấy khối tổn thương rõ trên niêm mạc dây thanh (khác nhóm tổn thương thực thể).",
        "Bất thường về đóng/mở: thanh môn khép không kín khi phát âm (khe hở phía sau, hình thoi, cong), "
        "hoặc hai dây thanh di động không đều/một bên giảm di động (liệt).",
        "Rối loạn giọng do căng cơ: vùng trên thanh môn (băng thanh thất, sụn phễu) co thắt, che bớt dây thanh.",
    ],
    "Disorder": [
        "Có bất thường thanh quản: hoặc thấy tổn thương trên dây thanh (hạt, polyp, nang, u nhú, u hạt, ung thư — "
        f"{PDF}, tr.37), hoặc dây thanh khép/di động bất thường dù không thấy khối.",
    ],
}

# ------------------------------------------------------------------------------ theo bệnh chi tiết (nhãn thật)
DETAIL_SIGNS: dict[str, str] = {
    "normal": f"[{PDF}, tr.35] Thanh quản bình thường: dây thanh trắng, khép kín khi phát âm, mở hình tam giác khi hít vào.",
    "nodules": f"[{PDF}, tr.37] Hạt dây thanh: hai nốt nhỏ đối xứng ở chỗ nối 1/3 trước – 1/3 giữa hai dây thanh.",
    "granuloma": f"[{PDF}, tr.37] U hạt mấu thanh: khối tròn, nhẵn ở mấu thanh âm sụn phễu (1/3 sau dây thanh).",
    "papilloma": f"[{PDF}, tr.37] U nhú thanh quản: tổn thương sùi dạng chùm/nhú, có thể lan rộng.",
    "carcinoma": f"[{PDF}, tr.37] Ung thư thanh quản: khối sùi/loét bờ không đều, có thể giảm di động dây thanh.",
    "polyp": "(Y văn bổ sung) Polyp dây thanh: khối nhẵn một bên, có cuống hoặc không, ở bờ tự do.",
    "cyst": "(Y văn bổ sung) Nang dây thanh: khối tròn dưới niêm mạc, một bên, bề mặt nhẵn.",
    "reinke_edema": "(Y văn bổ sung) Phù Reinke: dây thanh phù mọng, trong mờ, thường hai bên.",
    "leukoplakia": "(Y văn bổ sung) Bạch sản: mảng trắng bám trên bề mặt dây thanh.",
    "laryngitis": "(Y văn bổ sung) Viêm thanh quản: niêm mạc dây thanh đỏ, phù nề, có thể có xuất tiết.",
    "scar_sulcus": "(Y văn bổ sung) Sẹo/rãnh dây thanh: bờ tự do lõm hoặc cứng, rung kém, khép không kín.",
    "mucosal_lesion": "(Y văn bổ sung) Sưng nề/bất thường niêm mạc khu trú ở đoạn giữa dây thanh.",
    "muscle_tension_dysphonia": "(Y văn bổ sung) Rối loạn giọng do căng cơ: co thắt trên thanh môn, khe hở phía sau khi phát âm.",
    "glottic_insufficiency": "(Y văn bổ sung) Đóng thanh môn không kín: còn khe hở (phía sau/trước/hình thoi) khi phát âm.",
    "paresis": "(Y văn bổ sung) Liệt/yếu dây thanh: một bên giảm hoặc mất di động, khép không kín.",
    "atrophy": "(Y văn bổ sung) Teo dây thanh: bờ tự do cong, khe hở hình thoi khi phát âm.",
    "spasmodic_dysphonia": "(Y văn bổ sung) Loạn trương lực thanh quản: dây thanh khép quá chặt/ngắt quãng khi phát âm.",
    "functional_other": "(Y văn bổ sung) Rối loạn chức năng: rung/khép dây thanh bất thường, không thấy khối tổn thương.",
}


def signs_for(label: str) -> list[str]:
    return GROUP_SIGNS.get(label, ["(Chưa có mô tả dấu hiệu cho nhóm này — thêm vào src/throat_ai/knowledge.py.)"])


def detail_sign(detail: str | None) -> str | None:
    return DETAIL_SIGNS.get(detail or "")
