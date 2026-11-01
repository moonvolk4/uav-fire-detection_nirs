"""
Конвертация FLAME 3 CV Dataset в YOLO формат.

Реальная структура датасета:
    FLAME 3 CV Dataset (Sycan Marsh)/
    ├── Fire/
    │   ├── RGB/...
    │   └── Thermal/
    │       ├── Celsius TIFF/   ← термальные TIFF
    │       └── Raw JPG/        ← термальные JPG
    └── No Fire/
        ├── RGB/...
        └── Thermal/
            ├── Celsius TIFF/
            └── Raw JPG/

Подход:
- Fire/Thermal → автодетекция горячих областей через порог → bbox class 0 (fire)
- No Fire/Thermal → горячие области → bbox class 2 (false_source), либо пустой label
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Добавляем scripts/ в PATH
sys.path.insert(0, str(Path(__file__).parent))
from ir_preprocessing import load_thermal_image, normalize_thermal, apply_clahe, to_3channel


def find_hot_regions(img_gray, threshold_pct=90.0, min_area=30, max_area=100000):
    """Находит горячие области в термальном изображении."""
    thresh_val = np.percentile(img_gray, threshold_pct)
    _, binary = cv2.threshold(img_gray, int(thresh_val), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    h, w = img_gray.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 6.0:
            continue

        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        bboxes.append((x_center, y_center, norm_w, norm_h))

    return bboxes


def process_thermal_to_3ch(path):
    """Загрузка и конвертация термального изображения в 3-канальный PNG."""
    if path.suffix.lower() in ('.tif', '.tiff'):
        img = load_thermal_image(str(path))
        img_norm = normalize_thermal(img)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        img_norm = img

    img_clahe = apply_clahe(img_norm)
    img_3ch = to_3channel(img_clahe)
    return img_3ch, img_norm


def convert_flame3(raw_dir, output_images_dir, output_labels_dir):
    """Конвертация FLAME 3 в YOLO формат."""
    raw_path = Path(raw_dir)
    out_img = Path(output_images_dir)
    out_lbl = Path(output_labels_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # Поиск корневой папки датасета
    dataset_root = None
    for d in raw_path.rglob("Fire"):
        if d.is_dir() and (d / "Thermal").exists():
            dataset_root = d.parent
            break

    if dataset_root is None:
        # Может быть сам raw_dir — корень
        if (raw_path / "Fire").exists():
            dataset_root = raw_path
        else:
            print(f"Не найдена структура FLAME 3 в {raw_dir}")
            print("Ожидается: .../Fire/Thermal/ и .../No Fire/Thermal/")
            return

    print(f"Корень датасета: {dataset_root}")

    total = 0

    # === FIRE images (class 0) ===
    fire_thermal = dataset_root / "Fire" / "Thermal"
    fire_images = []
    for subdir in ["Raw JPG", "Celsius TIFF"]:
        d = fire_thermal / subdir
        if d.exists():
            fire_images.extend(sorted(d.glob("*.*")))
            break  # Берём первый найденный вариант

    if not fire_images:
        # Пробуем найти любые изображения
        fire_images = sorted(fire_thermal.rglob("*.*"))
        fire_images = [f for f in fire_images if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')]

    print(f"Fire thermal: {len(fire_images)} изображений")

    for img_path in tqdm(fire_images, desc="Fire"):
        try:
            img_3ch, img_gray = process_thermal_to_3ch(img_path)
            if img_3ch is None:
                continue

            name = f"fire_{img_path.stem}"
            cv2.imwrite(str(out_img / f"{name}.png"), img_3ch)

            # Автодетекция горячих областей
            bboxes = find_hot_regions(img_gray, threshold_pct=85.0, min_area=20)

            with open(str(out_lbl / f"{name}.txt"), "w") as f:
                if bboxes:
                    for xc, yc, w, h in bboxes:
                        f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                # Пустой файл если нет bbox — negative в fire папке

            total += 1
        except Exception as e:
            print(f"Ошибка {img_path.name}: {e}")

    # === NO FIRE images (class 2 для ярких областей, или пустой label) ===
    nofire_thermal = dataset_root / "No Fire" / "Thermal"
    nofire_images = []
    for subdir in ["Raw JPG", "Celsius TIFF"]:
        d = nofire_thermal / subdir
        if d.exists():
            nofire_images.extend(sorted(d.glob("*.*")))
            break

    if not nofire_images:
        nofire_images = sorted(nofire_thermal.rglob("*.*"))
        nofire_images = [f for f in nofire_images if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')]

    print(f"No Fire thermal: {len(nofire_images)} изображений")

    for img_path in tqdm(nofire_images, desc="No Fire"):
        try:
            img_3ch, img_gray = process_thermal_to_3ch(img_path)
            if img_3ch is None:
                continue

            name = f"nofire_{img_path.stem}"
            cv2.imwrite(str(out_img / f"{name}.png"), img_3ch)

            # Яркие области в no-fire → false_source (class 2)
            bboxes = find_hot_regions(img_gray, threshold_pct=95.0, min_area=50)

            with open(str(out_lbl / f"{name}.txt"), "w") as f:
                for xc, yc, w, h in bboxes:
                    f.write(f"2 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                # Пустой файл если нет ярких областей — чистый negative

            total += 1
        except Exception as e:
            print(f"Ошибка {img_path.name}: {e}")

    print(f"\nИтого сконвертировано: {total} изображений")
    print(f"Изображения: {out_img}")
    print(f"Labels: {out_lbl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Конвертация FLAME 3 в YOLO формат")
    parser.add_argument("--dataset", choices=["flame2", "flame3"], default="flame3")
    parser.add_argument("--raw-dir", required=True, help="Путь к распакованному датасету")
    parser.add_argument("--output-images", required=True, help="Директория для изображений")
    parser.add_argument("--output-labels", required=True, help="Директория для аннотаций")
    args = parser.parse_args()

    convert_flame3(args.raw_dir, args.output_images, args.output_labels)
