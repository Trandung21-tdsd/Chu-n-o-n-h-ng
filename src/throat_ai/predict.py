"""BƯỚC 4 — Chẩn đoán (gợi ý) cho ảnh mới bằng model đã chọn theo validation.

Ví dụ:
    python -m throat_ai.predict anh_noi_soi.jpg
    python -m throat_ai.predict anh1.jpg anh2.png thu_muc_anh/ --out ket_qua.csv
    python -m throat_ai.predict anh.jpg --task bagls_binary                    # dùng model của task khác
    python -m throat_ai.predict anh.jpg --checkpoint outputs/bagls_3class/runs/<run_id>/best.pt
    python -m throat_ai.predict thu_muc_1_video/ --same-video     # gộp mọi frame (cùng 1 video) -> dấu hiệu thanh môn

Nếu đã chạy seg_benchmark (outputs/bagls_seg), mỗi ảnh có thêm phần phân đoạn: 2 dây thanh + thanh môn + số đo.

Dùng trong Python:
    from throat_ai.predict import Diagnoser
    result = Diagnoser.from_benchmark("outputs/bagls_3class").diagnose("anh.jpg")

Kết quả chỉ là GỢI Ý nghiên cứu cho bác sĩ, không phải chẩn đoán y khoa.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ImageDataset, close_loader, make_loader
from .inference import load_checkpoint, predict_loader
from .metrics import apply_temperature
from .models import display_name, pick_device
from .preprocessing import PreprocessConfig, build_transform, load_rgb
from .sources import IMAGE_EXTENSIONS
from .utils import DEFAULT_OUTPUT_DIR, read_json, resolve_cli_path, setup_console, write_csv, write_json

DISCLAIMER = ("Kết quả do mô hình nghiên cứu tạo ra, chưa được thẩm định lâm sàng; chỉ dùng tham khảo, "
              "không thay thế chẩn đoán của bác sĩ và kết quả giải phẫu bệnh.")


class Diagnoser:
    def __init__(self, checkpoint, device: str = "auto", temperature: float | None = None,
                 selection: dict | None = None) -> None:
        self.device = pick_device(device)
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self.checkpoint = str(checkpoint)
        self.classes: list[str] = list(self.meta["classes"])
        self.names: dict = self.meta.get("display_names") or {}
        self.preprocess = PreprocessConfig.from_dict(self.meta["preprocess"])
        self.temperature = float(temperature) if temperature else 1.0
        self.selection = selection or {}

    @classmethod
    def from_benchmark(cls, out_dir, device: str = "auto", use_temperature: bool = True) -> "Diagnoser":
        selection_path = Path(out_dir) / "selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(f"Chưa có {selection_path}. Chạy benchmark/evaluate trước.")
        selection = read_json(selection_path)
        temperature = selection.get("temperature") if use_temperature else None
        return cls(selection["checkpoint"], device, temperature, selection)

    def predict_files(self, paths: list[str], batch_size: int = 16, workers: int = 0) -> np.ndarray:
        dataset = ImageDataset(paths, [0] * len(paths), self.preprocess, build_transform(train=False))
        loader = make_loader(dataset, batch_size, shuffle=False, workers=workers, seed=0,
                             pin_memory=self.device.type == "cuda")
        try:
            probs = predict_loader(self.model, loader, self.device)
        finally:
            close_loader(loader)
        return apply_temperature(probs, self.temperature) if self.temperature != 1.0 else probs

    def _quality_warnings(self, path: str) -> tuple[str | None, list[str]]:
        try:
            image = load_rgb(path)
        except Exception as exc:
            return f"Không đọc được ảnh: {type(exc).__name__}: {exc}", []
        warnings = []
        if min(image.size) < 64:
            return f"Ảnh quá nhỏ ({image.size[0]}x{image.size[1]}).", []
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        if gray.mean() < 15:
            warnings.append("Ảnh rất tối — kết quả có thể không đáng tin.")
        if gray.std() < 5:
            warnings.append("Ảnh gần như đồng màu (mờ/che khuất?) — kết quả có thể không đáng tin.")
        return None, warnings

    def diagnose_many(self, paths: list[str], top_k: int = 3, threshold: float = 0.5, batch_size: int = 16) -> list[dict]:
        results, readable, extra = [], [], {}
        for path in paths:
            error, warnings = self._quality_warnings(path)
            if error:
                results.append({"image": path, "status": "error", "message": error})
            else:
                readable.append(path)
                extra[path] = warnings
        probs = self.predict_files(readable, batch_size) if readable else np.zeros((0, len(self.classes)))
        model_info = {
            "checkpoint": self.checkpoint, "run_id": self.meta.get("run_id"), "architecture": display_name(self.meta["model_name"]),
            "task": self.meta.get("task"), "modality": self.meta.get("modality"), "anatomy": self.meta.get("anatomy"),
            "clahe": self.preprocess.clahe, "resolution": self.preprocess.resolution, "temperature": self.temperature,
            "val_score": self.selection.get("val_score", self.meta.get("val_score")),
            "test_macro_f1": (self.selection.get("test_metrics") or {}).get("test_macro_f1"),
        }
        for path, row in zip(readable, probs):
            order = np.argsort(row)[::-1]
            top = [{"label": self.classes[i], "name_vi": self.names.get(self.classes[i], self.classes[i]),
                    "probability": round(float(row[i]), 4)} for i in order[:top_k]]
            warnings = list(extra[path])
            confident = top[0]["probability"] >= threshold
            if not confident:
                warnings.append(f"Độ tin cậy thấp (< {threshold:.0%}) — nên để bác sĩ xem lại.")
            if self.meta.get("modality"):
                warnings.append(f"Model chỉ học ảnh loại '{self.meta['modality']}' ({self.meta.get('anatomy', '')}); "
                                "ảnh khác loại (vd. nội soi ánh sáng trắng, ảnh chụp họng bằng điện thoại) vẫn cho ra "
                                "một nhãn nhưng nhãn đó không có giá trị.")
            results.append({
                "image": path, "status": "ok" if confident else "uncertain",
                "prediction": top[0], "top_k": top,
                "probabilities": {c: round(float(p), 6) for c, p in zip(self.classes, row)},
                "warnings": warnings, "model": model_info, "disclaimer": DISCLAIMER,
            })
        order = {p: i for i, p in enumerate(paths)}
        return sorted(results, key=lambda r: order.get(r["image"], 0))

    def diagnose(self, path, top_k: int = 3, threshold: float = 0.5) -> dict:
        return self.diagnose_many([str(path)], top_k, threshold)[0]


def collect_images(inputs: list[str]) -> list[str]:
    paths = []
    for item in inputs:
        path = resolve_cli_path(item)
        if path.is_dir():
            paths += [str(p) for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS
                      and not p.name.startswith((".", "._"))]
        elif path.exists():
            paths.append(str(path))
        else:
            raise FileNotFoundError(f"Không thấy {item}")
    if not paths:
        raise FileNotFoundError("Không có ảnh nào để chẩn đoán.")
    return paths


def _default_benchmark_dir(task: str = "bagls_3class") -> Path:
    """Thư mục kết quả của ĐÚNG task đang dùng (không tự lấy model của task khác/đã bỏ)."""
    out_dir = DEFAULT_OUTPUT_DIR / task
    if not (out_dir / "selection.json").exists():
        raise FileNotFoundError(
            f"Chưa có model được chọn cho task '{task}' ({out_dir / 'selection.json'}).\n"
            f"Nếu benchmark đã có run hoàn thành (vd. bị dừng giữa chừng), chọn model từ các run đó bằng:\n"
            f"    python -m throat_ai.evaluate --task {task}")
    return out_dir


def check_task_allowed(diagnoser: "Diagnoser", strict: bool = True) -> None:
    """Chặn model của task không còn trong configs/tasks.json (vd. model CE-NBI cũ)."""
    from .utils import load_tasks

    task = diagnoser.meta.get("task")
    allowed = load_tasks()
    if task not in allowed:
        message = (f"Model này thuộc task '{task}' — không còn được dùng (task hiện có: {', '.join(allowed)}). "
                   f"Checkpoint: {diagnoser.checkpoint}")
        if strict:
            raise ValueError(message)
        print(f"! CẢNH BÁO: {message}")


def main(argv=None) -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="file ảnh hoặc thư mục ảnh")
    parser.add_argument("--task", default="bagls_3class", help="task trong configs/tasks.json (mặc định bagls_3class)")
    parser.add_argument("--model", help="thư mục benchmark có selection.json (mặc định: outputs/<task>)")
    parser.add_argument("--checkpoint", help="đường dẫn trực tiếp tới best.pt (bỏ qua --model)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5, help="dưới ngưỡng này => 'uncertain'")
    parser.add_argument("--no-temperature", action="store_true", help="không dùng hiệu chuẩn nhiệt độ")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", help="lưu kết quả: .json hoặc .csv")
    parser.add_argument("--no-seg", action="store_true", help="không dùng model phân đoạn (outputs/bagls_seg)")
    parser.add_argument("--same-video", action="store_true",
                        help="các ảnh là frame của CÙNG 1 video -> tính dấu hiệu mức video từ phân đoạn")
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold phải trong [0, 1]")

    if args.checkpoint:
        diagnoser = Diagnoser(resolve_cli_path(args.checkpoint), args.device)
    else:
        out_dir = resolve_cli_path(args.model) if args.model else _default_benchmark_dir(args.task)
        diagnoser = Diagnoser.from_benchmark(out_dir, args.device, use_temperature=not args.no_temperature)
    check_task_allowed(diagnoser, strict=not (args.model or args.checkpoint))
    results = diagnoser.diagnose_many(collect_images(args.images), args.top_k, args.threshold, args.batch_size)

    video = None
    if not args.no_seg:
        from .seg_predict import SegAnalyzer

        seg = SegAnalyzer.from_benchmark(device=args.device)
        ok = [r for r in results if r["status"] != "error"]
        if seg is not None and ok:
            masks = seg.predict([r["image"] for r in ok])
            for r, mask in zip(ok, masks):
                analysis = seg.analyse_image(r["image"], mask)
                r["segmentation"] = {"features": analysis["features"], "description": analysis["description"],
                                     "dice_vs_ground_truth": analysis["dice_vs_ground_truth"], "model": seg.info}
            if args.same_video:
                video = seg.analyse_video([r["image"] for r in ok], masks)
    info = next((r["model"] for r in results if "model" in r), {})
    print(f"Model: {info.get('architecture')} | task: {info.get('task')} | ảnh loại: {info.get('modality')}")
    for result in results:
        if result["status"] == "error":
            print(f"\n✖ {result['image']}: {result['message']}")
            continue
        print(f"\n{'✔' if result['status'] == 'ok' else '?'} {result['image']}")
        for rank, item in enumerate(result["top_k"], start=1):
            print(f"   {rank}. {item['name_vi']} ({item['label']}): {item['probability']:.1%}")
        for line in (result.get("segmentation") or {}).get("description", []):
            print(f"   [phân đoạn] {line}")
        for warning in result["warnings"]:
            print(f"   ! {warning}")
    if video:
        print(f"\nDấu hiệu mức video ({video['n_frames']} frame, so với thanh quản bình thường của train):")
        for sign in video["signs"]:
            print(f"   • {sign}")
        if video["sign_classifier_prediction"]:
            p = video["sign_classifier_proba"][video["sign_classifier_prediction"]]
            print(f"   ⇒ Nhóm bệnh gợi ý từ số đo phân đoạn: {video['sign_classifier_prediction']} ({p:.0%})")
        results.append({"image": "<video>", "status": "video_summary", "video": video})
    print(f"\n{DISCLAIMER}")

    if args.out:
        out = resolve_cli_path(args.out)
        if out.suffix.lower() == ".csv":
            rows = []
            for r in results:
                row = {"image": r["image"], "status": r["status"]}
                if r["status"] == "video_summary":
                    row["warnings"] = " | ".join(r["video"]["signs"])
                elif r["status"] != "error":
                    row.update(prediction=r["prediction"]["label"], prediction_vi=r["prediction"]["name_vi"],
                               probability=r["prediction"]["probability"], warnings=" | ".join(r["warnings"]),
                               **{f"p_{k}": v for k, v in r["probabilities"].items()})
                else:
                    row["warnings"] = r["message"]
                rows.append(row)
            write_csv(pd.DataFrame(rows), out)
        else:
            write_json(out, results if len(results) > 1 else results[0])
        print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
