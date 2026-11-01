"""
Подготовка false_source аннотаций из не-огневых кадров.

Извлекает ИК-яркие области из кадров без огня и размечает их как false_source (class 2).
Это помогает модели отличать огонь от животных, тёплых камней и других ИК-ярких объектов.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ir_preprocessing import load_thermal_image, normalize_thermal, to_3channel, apply_clahe
from convert_flame import save_yolo_labels


FALSE_SOURCE_CLASS_ID = 2
MIN_BRIGHT_AREA = 50
MAX_BRIGHT_AREA = 50000


def extract_bright_regions(
    img_gray: np.ndarray,
    threshold_pct: float = 95.0,
    min_area: int = MIN_BRIGHT_AREA,
    max_area: int = MAX_BRIGHT_AREA,
) -> list[tuple[int, float, float, float, float]]:
    """
    Находит ИК-яркие области в изображении через пороговую фильтрацию.

    Args:
        img_gray: Нормализованное grayscale изображение (uint8)
        threshold_pct: Перцентиль яркости для порога
        min_area: Минимальная площадь области
        max_area: Максимальная площадь (слишком большие — скорее всего фон)

    Returns:
        YOLO bboxes для false_source
    """
    threshold_val = np.percentile(img_gray, threshold_pct)
    _, binary = cv2.threshold(img_gray, int(threshold_val), 255, cv2.THRESH_BINARY)

    # Морфология для очистки шума
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

        # Фильтр по aspect ratio (слишком вытянутые — скорее артефакты)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 5.0:
            continue

        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        bboxes.append((FALSE_SOURCE_CLASS_ID, x_center, y_center, norm_w, norm_h))

    return bboxes


def process_negative_frames(
    images_dir: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    max_samples: int = 500,
):
    """
    Обработка не-огневых кадров для извлечения false_source.

    Берёт изображения с пустыми label-файлами (negative samples)
    и создаёт false_source аннотации для ИК-ярких регионов.
    """
    img_path = Path(images_dir)
    lbl_path = Path(labels_dir)
    out_img = Path(output_images_dir)
    out_lbl = Path(output_labels_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # Находим negative samples (пустые label-файлы)
    negative_images = []
    for lbl_file in lbl_path.glob("*.txt"):
        content = lbl_file.read_text().strip()
        if not content:
            img_stem = lbl_file.stem
            for ext in (".png", ".jpg", ".jpeg"):
                img_file = img_path / (img_stem + ext)
                if img_file.exists():
                    negative_images.append(img_file)
                    break

    if not negative_images:
        print("Не найдены negative samples (изображения с пустыми labels)")
        return

    print(f"Найдено {len(negative_images)} negative samples")
    count = 0

    for img_file in tqdm(negative_images[:max_samples], desc="False sources"):
        try:
            img_gray = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue

            bboxes = extract_bright_regions(img_gray)
            if not bboxes:
                continue

            # Копируем изображение и сохраняем label
            out_name = f"false_{img_file.stem}"
            cv2.imwrite(str(out_img / f"{out_name}.png"), cv2.imread(str(img_file)))
            save_yolo_labels(bboxes, str(out_lbl / f"{out_name}.txt"))
            count += 1

        except Exception as e:
            print(f"Ошибка {img_file.name}: {e}")

    print(f"Создано {count} изображений с false_source аннотациями")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка false_source из negative samples")
    parser.add_argument("--images", required=True, help="Директория с изображениями")
    parser.add_argument("--labels", required=True, help="Директория с labels")
    parser.add_argument("--output-images", required=True, help="Выходная директория изображений")
    parser.add_argument("--output-labels", required=True, help="Выходная директория labels")
    parser.add_argument("--max-samples", type=int, default=500, help="Максимум false_source samples")
    args = parser.parse_args()

    process_negative_frames(
        args.images,
        args.labels,
        args.output_images,
        args.output_labels,
        args.max_samples,
    )
