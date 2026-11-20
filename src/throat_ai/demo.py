"""Lấy NGẪU NHIÊN 1 (hoặc vài) ảnh trong dữ liệu -> PHÂN ĐOẠN 2 dây thanh + thanh môn, đo dấu hiệu, dự đoán nhóm bệnh.

Mặc định lấy từ tập TEST (model chưa từng thấy khi train/chọn model), và so với nhãn thật + mask thật.
Dùng 2 model (model nào đã train thì dùng model đó):
    - model phân loại  (outputs/bagls_3class): nhóm bệnh từ ảnh + Grad-CAM
    - model phân đoạn  (outputs/bagls_seg)    : mask dây thanh trái/phải + thanh môn -> số đo -> dấu hiệu;
      gộp các frame CÙNG VIDEO của ảnh đó -> khép kín? rung đều? khe đồng hồ cát? -> nhóm bệnh gợi ý

Ví dụ:
    python -m throat_ai.demo                                   # 1 ảnh ngẫu nhiên, model đã chọn mới nhất
    python -m throat_ai.demo --n 5 --seed 7                    # 5 ảnh, lặp lại được
    python -m throat_ai.demo --label Organic_lesion            # chỉ lấy ảnh có nhãn thật là tổn thương thực thể
    python -m throat_ai.demo --image duong_dan/anh.jpg         # 1 ảnh cụ thể (không có nhãn thật)
    python -m throat_ai.demo --task bagls_binary --split val

    python -m throat_ai.demo --no-seg                          # chỉ model phân loại

Kết quả: outputs/<task>/demo/<thời_gian>_<ảnh>.png (ảnh + mask thật/dự đoán + Grad-CAM + xác suất + dấu hiệu) và .json.
Bản đồ chú ý = Grad-CAM: vùng ảnh ảnh hưởng nhiều nhất tới dự đoán. Đó KHÔNG phải vùng tổn thương đã được xác nhận.
"""
from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_manifest
from .bagls import DETAIL_VI
from .knowledge import detail_sign, signs_for
from .metrics import apply_temperature
from .predict import DISCLAIMER, Diagnoser, _default_benchmark_dir, check_task_allowed
from .preprocessing import build_transform, load_rgb, preprocess_image
from .seg_data import load_seg_config, overlay
from .seg_predict import SegAnalyzer, default_seg_dir
from .glottis import consistency
from .utils import DEFAULT_PREPARED_DIR, iso_now, resolve_cli_path, setup_console, timestamp, write_json


def _target_layer(model):
    """Lớp đặc trưng không gian cuối cùng để tính Grad-CAM (None nếu kiến trúc không hỗ trợ, vd. ViT)."""
    for name in ("layer4", "features", "trunk_output", "conv5"):
        module = getattr(model, name, None)
        if module is not None:
            return module, type(model).__name__.startswith("Swin")  # Swin trả về (N, H, W, C)
    return None, False


def grad_cam(model, tensor: torch.Tensor, class_index: int) -> np.ndarray | None:
    layer, channels_last = _target_layer(model)
    if layer is None:
        return None
    store = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        store["activation"] = output

    handle = layer.register_forward_hook(hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor)
            logits[0, class_index].backward()
        activation = store["activation"]
        gradient = activation.grad
        if channels_last:
            activation, gradient = activation.permute(0, 3, 1, 2), gradient.permute(0, 3, 1, 2)
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().float().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    except Exception:
        return None
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def _attention_summary(cam: np.ndarray | None, image: np.ndarray) -> str:
    if cam is None:
        return "Kiến trúc này chưa hỗ trợ bản đồ chú ý."
    content = image.max(axis=2) > 8  # bỏ viền đen đệm
    hot = (cam >= 0.5) & content
    share = hot.sum() / max(content.sum(), 1)
    ys, xs = np.nonzero(hot)
    if len(xs) == 0:
        return "Không có vùng nổi bật rõ."
    h, w = cam.shape
    vertical = ["phía trên", "giữa", "phía dưới"][min(2, int(ys.mean() / h * 3))]
    horizontal = ["bên trái", "chính giữa", "bên phải"][min(2, int(xs.mean() / w * 3))]
    return f"Model dựa chủ yếu vào vùng {vertical} – {horizontal} của ảnh (~{share:.0%} diện tích ảnh)."


def analyse(diagnoser: Diagnoser, path: str, true_label: str | None, top_k: int = 3, true_detail: str | None = None) -> dict:
    image = preprocess_image(load_rgb(path), diagnoser.preprocess)
    tensor = build_transform(train=False)(image).as_subclass(torch.Tensor).unsqueeze(0).to(diagnoser.device)
    with torch.inference_mode():
        probs = torch.softmax(diagnoser.model(tensor).float(), dim=1)[0].cpu().numpy().astype(np.float64)
    probs = apply_temperature(probs[None], diagnoser.temperature)[0] if diagnoser.temperature != 1.0 else probs
    order = np.argsort(probs)[::-1]
    predicted = diagnoser.classes[order[0]]
    cam = grad_cam(diagnoser.model, tensor, int(order[0]))
    if cam is not None:
        cam = cam * (image.max(axis=2) > 8)  # không tô màu viền đen đệm (không phải nội dung ảnh)
    names = diagnoser.names
    return {
        "image": path,
        "true_label": true_label,
        "true_label_vi": names.get(true_label, true_label) if true_label else None,
        "prediction": predicted,
        "prediction_vi": names.get(predicted, predicted),
        "probability": float(probs[order[0]]),
        "correct": (predicted == true_label) if true_label else None,
        "true_detail": true_detail or None,
        "true_detail_vi": DETAIL_VI.get(true_detail or "", true_detail) if true_detail else None,
        "true_detail_sign": detail_sign(true_detail),
        "top_k": [{"label": diagnoser.classes[i], "name_vi": names.get(diagnoser.classes[i], diagnoser.classes[i]),
                   "probability": round(float(probs[i]), 4)} for i in order[:top_k]],
        "signs_of_predicted_disease": signs_for(predicted),
        "attention": _attention_summary(cam, image),
        "model": {"architecture": diagnoser.meta.get("model_name"), "task": diagnoser.meta.get("task"),
                  "modality": diagnoser.meta.get("modality"), "clahe": diagnoser.preprocess.clahe,
                  "run_id": diagnoser.meta.get("run_id")},
        "disclaimer": DISCLAIMER,
        "_image": image,
        "_cam": cam,
    }


GROUP_VI = {"Normal": "Bình thường", "Functional": "Rối loạn chức năng", "Organic_lesion": "Tổn thương thực thể",
            "Disorder": "Có bệnh lý"}


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def render(result: dict, path: Path) -> Path | None:
    try:
        from .charts import SERIES, TEXT, TEXT_2, _pyplot

        plt = _pyplot()
    except ImportError:
        return None
    seg, clf = result.get("segmentation"), result.get("classifier")
    panels = []
    if seg:
        panels.append(("Ảnh nội soi", seg["_image"], None))
        if seg["_truth"] is not None:
            panels.append(("Mask thật (gán tay)", overlay(seg["_image"], seg["_truth"]), None))
        dice = seg.get("dice_vs_ground_truth")
        panels.append(("Mask model dự đoán" + (f" (Dice {dice['mean']:.2f})" if dice else ""),
                       overlay(seg["_image"], seg["_pred"]), None))
    if clf:
        if not seg:
            panels.append(("Ảnh đưa vào model" + (" (CLAHE)" if clf["model"]["clahe"] else ""), clf["_image"], None))
        panels.append(("Vùng model phân loại dựa vào (Grad-CAM)", clf["_image"], clf["_cam"]))
    bars = []
    if clf:
        bars.append(("Model phân loại (ảnh)", clf["top_k"], clf["prediction"]))
    video = (seg or {}).get("video")
    if video and video.get("sign_classifier_proba"):
        proba = video["sign_classifier_proba"]
        items = [{"label": k, "name_vi": GROUP_VI.get(k, k), "probability": v}
                 for k, v in sorted(proba.items(), key=lambda kv: -kv[1])]
        bars.append((f"Từ số đo phân đoạn ({video['n_frames']} frame cùng video)", items, items[0]["label"]))
    columns = len(panels) + (1 if bars else 0)
    wrapped = [textwrap.fill(line, 175, subsequent_indent="   ") for line in result["lines"][:16]]
    text_lines = sum(w.count("\n") + 1 for w in wrapped) + 2
    text_height = 0.6 + 0.19 * text_lines
    fig = plt.figure(figsize=(3.6 * columns + 1, 4.4 + text_height))
    grid = fig.add_gridspec(2, columns, height_ratios=[4.0, text_height],
                            width_ratios=[1] * len(panels) + ([1.3] if bars else []))
    for i, (title, image, cam) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(image)
        if cam is not None:
            ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")
    if bars:
        sub = grid[0, -1].subgridspec(len(bars), 1, hspace=0.9)
        for j, (title, items, chosen) in enumerate(bars):
            ax = fig.add_subplot(sub[j, 0])
            items = items[::-1]
            values = [it["probability"] for it in items]
            ax.barh(range(len(items)), values, color=[SERIES[0] if it["label"] == chosen else "#b9b8b2" for it in items])
            for y, v in enumerate(values):
                ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=8, color=TEXT)
            ax.set_yticks(range(len(items)), [_short(GROUP_VI.get(it["label"], it["name_vi"]), 24) for it in items], fontsize=8)
            ax.set_xlim(0, 1.2)
            ax.set_title(title, fontsize=9)
            ax.grid(axis="y", visible=False)
    if seg:
        fig.text(0.01, text_height / (4.4 + text_height) + 0.01, "■ xanh lá: dây thanh bên trái ảnh   ■ xanh dương: dây thanh bên phải ảnh   ■ đỏ: thanh môn",
                 fontsize=8.5, color=TEXT_2)
    ax_text = fig.add_subplot(grid[1, :])
    ax_text.axis("off")
    header = result["header"]
    ax_text.text(0, 1, header, fontsize=11.5, fontweight="bold", va="top",
                 color=TEXT if result.get("correct") in (True, None) else SERIES[7])
    ax_text.text(0, 1 - 1.6 / text_lines, "\n".join(wrapped), fontsize=8.6, va="top", color=TEXT_2)
    fig.suptitle(Path(result["image"]).name, fontsize=10, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------ chọn ảnh
def load_pool(seg: SegAnalyzer | None, diagnoser: Diagnoser | None, prepared_dir: Path):
    """Danh sách ảnh để bốc ngẫu nhiên: manifest phân đoạn (có video_id, mask) hoặc manifest phân loại."""
    import pandas as pd

    seg_manifest = prepared_dir / load_seg_config()["task"] / "manifest.csv"
    if seg is not None and seg_manifest.exists():
        frame = pd.read_csv(seg_manifest, dtype=str, keep_default_na=False)
        return frame[["image_path", "label", "histology", "split", "video_id"]]
    if diagnoser is None:
        raise FileNotFoundError(f"Không thấy {seg_manifest}. Chạy seg_benchmark trước hoặc dùng --image.")
    manifest = prepared_dir / diagnoser.meta["task"] / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Không thấy {manifest}. Chạy prepare trước hoặc dùng --image.")
    full = load_manifest(manifest)
    full = full[full["canonical"]]
    return full.rename(columns={"patient_id": "video_id"})[["image_path", "label", "histology", "split", "video_id"]]


def pick_images(pool, split: str, n: int, seed: int | None, label: str | None) -> list[dict]:
    frame = pool[pool["split"] == split]
    if label:
        frame = frame[frame["label"] == label]
    if frame.empty:
        raise ValueError(f"Không có ảnh {'nhãn ' + repr(label) + ' ' if label else ''}trong tập {split}")
    rng = random.Random(seed)
    rows = rng.sample(range(len(frame)), min(n, len(frame)))
    return [frame.iloc[i].to_dict() for i in rows]


# ------------------------------------------------------------------------------ phân tích 1 ảnh
def analyse_item(item: dict, diagnoser: Diagnoser | None, seg: SegAnalyzer | None, pool) -> dict:
    path, label = item["image_path"], item.get("label") or None
    if diagnoser and label and label not in diagnoser.classes and "Disorder" in diagnoser.classes:
        label = "Disorder"  # task nhị phân: Functional/Organic_lesion -> Có bệnh lý
    detail = item.get("histology") or None
    names = diagnoser.names if diagnoser else {}
    result = {"image": path, "true_label": label, "true_label_vi": names.get(label, GROUP_VI.get(label, label)) if label else None,
              "true_detail": detail, "true_detail_vi": DETAIL_VI.get(detail or "", detail) if detail else None,
              "true_detail_sign": detail_sign(detail), "video_id": item.get("video_id"), "disclaimer": DISCLAIMER}
    lines: list[str] = []
    if diagnoser:
        clf = analyse(diagnoser, path, label, true_detail=detail)
        result["classifier"] = clf
        result.update(prediction=clf["prediction"], prediction_vi=clf["prediction_vi"], probability=clf["probability"],
                      correct=clf["correct"], top_k=clf["top_k"], signs_of_predicted_disease=clf["signs_of_predicted_disease"],
                      attention=clf["attention"])
        lines.append(f"[Model phân loại] Dự đoán: {clf['prediction_vi']} ({clf['probability']:.0%}). {clf['attention']}")
        lines += ["   • " + s for s in clf["signs_of_predicted_disease"][:2]]
    if seg:
        s = seg.analyse_image(path)
        result["segmentation"] = s
        lines.append("[Phân đoạn – frame này] " + " ".join(s["description"]))
        if s["dice_vs_ground_truth"]:
            d = s["dice_vs_ground_truth"]
            lines.append(f"   Độ khớp với mask gán tay (Dice): dây thanh trái {d['left_fold']:.2f}, phải {d['right_fold']:.2f}, "
                         f"thanh môn {d['glottis']:.2f}")
        video_id = item.get("video_id")
        if pool is not None and video_id:
            frames = pool[(pool["video_id"] == video_id) & (pool["split"] == item.get("split"))]["image_path"].tolist()
            if len(frames) >= 3:
                video = seg.analyse_video(frames)
                s["video"] = video
                lines.append(f"[Phân đoạn – cả video {video_id}, {video['n_frames']} frame] Dấu hiệu so với thanh quản "
                             "bình thường (train):")
                lines += ["   • " + t for t in video["signs"][:5]] or ["   • (chưa có khoảng tham chiếu — chạy lại seg_benchmark)"]
                if video["sign_classifier_prediction"]:
                    p = video["sign_classifier_proba"][video["sign_classifier_prediction"]]
                    lines.append(f"   ⇒ Nhóm bệnh gợi ý từ số đo phân đoạn: {GROUP_VI.get(video['sign_classifier_prediction'])} ({p:.0%})")
                    if not diagnoser:
                        result.update(prediction=video["sign_classifier_prediction"],
                                      prediction_vi=GROUP_VI.get(video["sign_classifier_prediction"]),
                                      probability=p,
                                      correct=(video["sign_classifier_prediction"] == label) if label else None)
                note = consistency(result.get("prediction") if diagnoser else None,
                                   {"support": video["support"], "leaning": video["rule_leaning"]})
                if note:
                    lines.append("   " + note)
    if result.get("true_detail_sign"):
        lines.append(f"[Bệnh thật theo metadata] {result['true_detail_sign']}")
    if label and result.get("prediction"):
        verdict = "ĐÚNG" if result["correct"] else "SAI"
        result["header"] = (f"Nhãn thật: {_short(GROUP_VI.get(label, label), 40)}"
                            + (f" ({result['true_detail_vi']})" if result.get("true_detail_vi") else "")
                            + f"  |  Dự đoán: {_short(GROUP_VI.get(result['prediction'], result['prediction_vi']), 40)}  →  {verdict}")
    elif result.get("prediction"):
        result["header"] = f"Dự đoán: {result['prediction_vi']} ({result['probability']:.0%})"
    else:
        result["header"] = "Phân đoạn thanh môn" + (f" — nhãn thật: {GROUP_VI.get(label, label)}" if label else "")
    result["lines"] = lines
    return result


def _public(result: dict) -> dict:
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not str(k).startswith("_")}
        return value

    return clean(result)


def main(argv=None) -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="bagls_3class", help="task phân loại trong configs/tasks.json")
    parser.add_argument("--model", help="thư mục benchmark phân loại có selection.json (mặc định: outputs/<task>)")
    parser.add_argument("--seg-model", help="thư mục benchmark phân đoạn (mặc định: outputs/bagls_seg)")
    parser.add_argument("--no-seg", action="store_true", help="không dùng model phân đoạn")
    parser.add_argument("--no-classifier", action="store_true", help="không dùng model phân loại")
    parser.add_argument("--image", help="phân tích 1 ảnh cụ thể thay vì lấy ngẫu nhiên")
    parser.add_argument("--n", type=int, default=1, help="số ảnh ngẫu nhiên")
    parser.add_argument("--seed", type=int, help="cố định lựa chọn ngẫu nhiên")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--label", help="chỉ lấy ảnh có nhãn thật này (Normal / Functional / Organic_lesion)")
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    diagnoser, missing = None, []
    out_dir = None
    if not args.no_classifier:
        try:
            out_dir = resolve_cli_path(args.model) if args.model else _default_benchmark_dir(args.task)
            diagnoser = Diagnoser.from_benchmark(out_dir, args.device)
            check_task_allowed(diagnoser, strict=not args.model)
        except FileNotFoundError as exc:
            if args.model:
                raise
            missing.append(str(exc))
            out_dir = None
    seg = None
    if not args.no_seg:
        seg = SegAnalyzer.from_benchmark(resolve_cli_path(args.seg_model) if args.seg_model else None, args.device)
        if seg is None:
            missing.append(f"Chưa có model phân đoạn ({default_seg_dir() / 'selection.json'}) — chạy: "
                           "python -m throat_ai.seg_benchmark")
    if diagnoser is None and seg is None:
        raise FileNotFoundError("\n".join(missing))
    for message in missing:
        print("! " + message.splitlines()[0])

    prepared = resolve_cli_path(args.prepared_dir)
    pool = None
    if args.image:
        items = [{"image_path": str(resolve_cli_path(args.image)), "label": None}]
    else:
        pool = load_pool(seg, diagnoser, prepared)
        items = pick_images(pool, args.split, args.n, args.seed, args.label)
        if args.split == "train":
            print("! Ảnh lấy từ tập train: model đã thấy ảnh này khi học, kết quả sẽ lạc quan hơn thực tế.")

    folder = (out_dir if out_dir is not None else (resolve_cli_path(args.seg_model) if args.seg_model else default_seg_dir())) / "demo"
    folder.mkdir(parents=True, exist_ok=True)
    correct = judged = 0
    for item in items:
        result = analyse_item(item, diagnoser, seg, pool)
        stem = f"{timestamp()}_{Path(item['image_path']).stem}".replace(" ", "_")
        png = render(result, folder / f"{stem}.png")
        public = _public(result)
        public.pop("lines", None)
        public.update(created_at=iso_now(), figure=str(png) if png else None, explanation=result["lines"])
        write_json(folder / f"{stem}.json", public)
        if result.get("correct") is not None:
            judged += 1
            correct += bool(result["correct"])
        print(f"\nẢnh: {item['image_path']}")
        print(f"  {result['header']}")
        for line in result["lines"]:
            print(f"  {line}")
        print(f"  Hình: {png or '(cài matplotlib để xuất hình)'}")
    if judged > 1:
        print(f"\nĐúng {correct}/{judged} ảnh.")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
