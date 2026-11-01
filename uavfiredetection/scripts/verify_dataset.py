"""
Визуальная проверка датасета: рисует bounding boxes на изображениях.

Выбирает случайные изображения из каждого split и отрисовывает аннотации.
Сохраняет результаты в data/verification/ для ручного осмотра.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VERIFY_DIR = PROJECT_ROOT / "data" / "verification"

CLASS_NAMES = {0: "fire", 1: "smoke", 2: "false_source"}
CLASS_COLORS = {
    0: (0, 0, 255),     # fire — красный (BGR)
    1: (0, 255, 255),   # smoke — жёлтый
    2: (255, 128, 0),   # false_source — синий
}


def draw_yolo_bboxes(img: np.ndarray, label_path: str) -> np.ndarray:
    """Отрисовка YOLO bboxes на изображении."""
    h, w = img.shape[:2]
    result = img.copy()

    label_file = Path(label_path)
    if not label_file.exists():
        return result

    content = label_file.read_text().strip()
    if not content:
        cv2.putText(result, "NEGATIVE (no objects)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        return result

    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Денормализация
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result


def verify_split(split: str, n_samples: int = 20, seed: int = 42):
    """Проверка одного split."""
    img_dir = PROCESSED_DIR / split / "images"
    lbl_dir = PROCESSED_DIR / split / "labels"

    if not img_dir.exists():
        print(f"Split {split} не найден: {img_dir}")
        return

    images = sorted(img_dir.glob("*.png"))
    if not images:
        images = sorted(img_dir.glob("*.jpg"))
    if not images:
        print(f"Нет изображений в {img_dir}")
        return

    random.seed(seed)
    selected = random.sample(images, min(n_samples, len(images)))

    out_dir = VERIFY_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{split.upper()}: проверка {len(selected)} из {len(images)} изображений")

    for img_file in selected:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        lbl_file = lbl_dir / (img_file.stem + ".txt")
        annotated = draw_yolo_bboxes(img, str(lbl_file))
        cv2.imwrite(str(out_dir / img_file.name), annotated)

    print(f"  Сохранено в {out_dir}")


def print_dataset_summary():
    """Краткая сводка по датасету."""
    print("\nСводка по датасету:")
    print("-" * 40)

    for split in ["train", "val", "test"]:
        img_dir = PROCESSED_DIR / split / "images"
        lbl_dir = PROCESSED_DIR / split / "labels"

        if not img_dir.exists():
            print(f"  {split}: не найден")
            continue

        n_images = len(list(img_dir.glob("*.*")))
        n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        # Подсчёт bbox
        total_bboxes = 0
        class_counts = {}
        for lbl_file in lbl_dir.glob("*.txt"):
            for line in lbl_file.read_text().strip().split("\n"):
                if line.strip():
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    total_bboxes += 1

        print(f"  {split}: {n_images} images, {n_labels} labels, {total_bboxes} bboxes")
        for cls_id in sorted(class_counts):
            name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
            print(f"    {name}: {class_counts[cls_id]} bboxes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуальная проверка датасета")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--n-samples", type=int, default=20, help="Число изображений для проверки")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print_dataset_summary()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        verify_split(split, args.n_samples, args.seed)

    print(f"\nРезультаты проверки в {VERIFY_DIR}")
