"""
Валидация и оценка модели на val/test split.

Метрики:
- mAP@0.5 и mAP@0.5:0.95 по классам
- Precision, Recall, F1 по классам
- Confusion matrix
- Анализ по размерам bbox (small/medium/large)

Использование:
    python src/val.py --weights runs/detect/fire_v1/weights/best.pt
    python src/val.py --weights runs/detect/fire_v1/weights/best.pt --split test
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.utils import CLASS_NAMES, get_device


def validate(
    weights: str,
    data_cfg: str = None,
    split: str = "val",
    device: str = None,
    imgsz: int = 1024,
    conf: float = 0.001,
    iou: float = 0.6,
    save_dir: str = None,
):
    """
    Валидация модели с расширенными метриками.

    Args:
        weights: Путь к весам модели
        data_cfg: Путь к YAML датасета
        split: 'val' или 'test'
        device: Устройство
        imgsz: Размер входа
        conf: Порог confidence
        iou: Порог IoU для NMS
        save_dir: Директория для сохранения результатов
    """
    if device is None:
        device = get_device()

    data_cfg = data_cfg or str(PROJECT_ROOT / "configs" / "dataset.yaml")

    print(f"Модель: {weights}")
    print(f"Split: {split}")
    print(f"Устройство: {device}")

    model = YOLO(weights)

    # Запуск валидации
    results = model.val(
        data=data_cfg,
        split=split,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
    )

    # Извлечение метрик
    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТЫ ({split.upper()})")
    print("=" * 60)

    # Общие метрики
    print(f"\nmAP@0.5:      {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")

    # По классам
    print(f"\n{'Класс':<15} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>8}")
    print("-" * 55)

    class_names = list(CLASS_NAMES.values())
    for i, name in enumerate(class_names):
        if i < len(results.box.ap50):
            p = results.box.p[i] if i < len(results.box.p) else 0
            r = results.box.r[i] if i < len(results.box.r) else 0
            ap50 = results.box.ap50[i]
            ap = results.box.ap[i] if i < len(results.box.ap) else 0
            f1 = 2 * p * r / (p + r + 1e-16)
            print(f"{name:<15} {p:>8.4f} {r:>8.4f} {ap50:>8.4f} {ap:>8.4f}")
            print(f"{'  F1':>15} {f1:>8.4f}")

    # Сохранение результатов
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        metrics = {
            "split": split,
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "per_class": {},
        }

        for i, name in enumerate(class_names):
            if i < len(results.box.ap50):
                metrics["per_class"][name] = {
                    "precision": float(results.box.p[i]) if i < len(results.box.p) else 0,
                    "recall": float(results.box.r[i]) if i < len(results.box.r) else 0,
                    "ap50": float(results.box.ap50[i]),
                    "ap50_95": float(results.box.ap[i]) if i < len(results.box.ap) else 0,
                }

        with open(save_path / f"metrics_{split}.json", "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nМетрики сохранены: {save_path}/metrics_{split}.json")

    # Скорость
    speed = results.speed
    print(f"\nСкорость:")
    print(f"  Preprocess:  {speed.get('preprocess', 0):.1f} ms")
    print(f"  Inference:   {speed.get('inference', 0):.1f} ms")
    print(f"  Postprocess: {speed.get('postprocess', 0):.1f} ms")
    total = sum(speed.values())
    if total > 0:
        print(f"  FPS:         {1000 / total:.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Валидация модели детекции огня")
    parser.add_argument("--weights", required=True, help="Путь к весам модели")
    parser.add_argument("--data-cfg", type=str, help="Путь к YAML датасета")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", type=str, help="Устройство")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold для NMS")
    parser.add_argument("--save-dir", type=str, help="Директория для сохранения метрик")
    args = parser.parse_args()

    validate(
        weights=args.weights,
        data_cfg=args.data_cfg,
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save_dir=args.save_dir,
    )
