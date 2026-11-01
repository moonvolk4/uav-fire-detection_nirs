"""
ИК-препроцессинг: конвертация 16-bit thermal TIFF в 3-channel 8-bit PNG.

Поддерживает:
- 16-bit и 32-bit float TIFF (radiometric thermal)
- Одноканальные grayscale изображения
- Percentile clipping для устранения выбросов
- CLAHE для усиления контраста огонь/фон
- Дублирование в 3 канала для совместимости с pretrained YOLOv8
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_thermal_image(path: str) -> np.ndarray:
    """Загрузка термального изображения (TIFF/PNG/JPG) в float32."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def normalize_thermal(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Percentile clipping + нормализация в [0, 255]."""
    low = np.percentile(img, low_pct)
    high = np.percentile(img, high_pct)
    if high - low < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img_clipped = np.clip(img, low, high)
    img_norm = ((img_clipped - low) / (high - low) * 255).astype(np.uint8)
    return img_norm


def apply_clahe(img: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    """CLAHE для усиления контраста (вход: uint8 grayscale)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def to_3channel(img: np.ndarray) -> np.ndarray:
    """Дублирование одноканального в 3-канальное изображение."""
    return cv2.merge([img, img, img])


def process_thermal_image(
    path: str,
    use_clahe: bool = True,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """Полный pipeline: загрузка → нормализация → CLAHE → 3 канала."""
    img = load_thermal_image(path)
    img_norm = normalize_thermal(img, low_pct, high_pct)
    if use_clahe:
        img_norm = apply_clahe(img_norm)
    return to_3channel(img_norm)


def process_directory(
    input_dir: str,
    output_dir: str,
    extensions: tuple = (".tiff", ".tif", ".png", ".jpg", ".jpeg"),
    use_clahe: bool = True,
):
    """Обработка всех термальных изображений в директории."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    if not files:
        print(f"Не найдено изображений в {input_dir}")
        return

    print(f"Обработка {len(files)} изображений из {input_dir}")
    for f in tqdm(files):
        try:
            img = process_thermal_image(str(f), use_clahe=use_clahe)
            out_name = f.stem + ".png"
            cv2.imwrite(str(output_path / out_name), img)
        except Exception as e:
            print(f"Ошибка при обработке {f.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ИК-препроцессинг термальных изображений")
    parser.add_argument("--input", required=True, help="Директория с исходными изображениями")
    parser.add_argument("--output", required=True, help="Директория для сохранения результатов")
    parser.add_argument("--no-clahe", action="store_true", help="Отключить CLAHE")
    parser.add_argument("--low-pct", type=float, default=1.0, help="Нижний перцентиль (default: 1.0)")
    parser.add_argument("--high-pct", type=float, default=99.0, help="Верхний перцентиль (default: 99.0)")
    args = parser.parse_args()

    process_directory(
        args.input,
        args.output,
        use_clahe=not args.no_clahe,
    )
