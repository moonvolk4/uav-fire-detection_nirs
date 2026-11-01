"""
Инференс модели на произвольных ИК-изображениях.

Поддерживает:
- Одиночные изображения (.tiff, .tif, .png, .jpg)
- Директории с изображениями
- Автоматический препроцессинг thermal TIFF

Использование:
    python src/predict.py --weights runs/detect/fire_v1/weights/best.pt --source path/to/image.tiff
    python src/predict.py --weights best.pt --source path/to/dir/ --conf 0.3
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.utils import CLASS_NAMES, get_device

# Цвета для визуализации (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),     # fire — красный
    1: (0, 255, 255),   # smoke — жёлтый
    2: (255, 128, 0),   # false_source — синий
}


def preprocess_if_thermal(image_path: str) -> str:
    """
    Если изображение — thermal TIFF, конвертировать в 3-channel PNG.
    Возвращает путь к готовому изображению.
    """
    path = Path(image_path)
    if path.suffix.lower() in (".tiff", ".tif"):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from ir_preprocessing import process_thermal_image

        img = process_thermal_image(str(path))
        temp_path = path.with_suffix(".png")
        cv2.imwrite(str(temp_path), img)
        return str(temp_path)
    return image_path


def draw_detections(image: np.ndarray, results) -> np.ndarray:
    """Отрисовка детекций на изображении."""
    annotated = image.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
            label = f"{name} {conf:.2f}"

            # Bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Фон для текста
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return annotated


def predict(
    weights: str,
    source: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 1024,
    device: str = None,
    save_dir: str = None,
    show: bool = False,
):
    """
    Инференс на изображениях.

    Args:
        weights: Путь к весам модели
        source: Путь к изображению или директории
        conf: Порог confidence
        iou: Порог IoU для NMS
        imgsz: Размер входа
        device: Устройство
        save_dir: Директория для сохранения (None = runs/predict/)
        show: Показать результат в окне
    """
    if device is None:
        device = get_device()

    save_dir = save_dir or str(PROJECT_ROOT / "runs" / "predict")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)

    source_path = Path(source)
    if source_path.is_dir():
        extensions = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
        image_files = [f for f in sorted(source_path.iterdir()) if f.suffix.lower() in extensions]
    else:
        image_files = [source_path]

    print(f"Модель: {weights}")
    print(f"Изображений: {len(image_files)}")
    print(f"Confidence: {conf}")

    total_detections = {name: 0 for name in CLASS_NAMES.values()}

    for img_file in image_files:
        # Препроцессинг thermal если нужно
        processed_path = preprocess_if_thermal(str(img_file))

        # Инференс
        results = model.predict(
            source=processed_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        # Загрузка и аннотирование
        image = cv2.imread(processed_path)
        annotated = draw_detections(image, results)

        # Подсчёт детекций
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                    if name in total_detections:
                        total_detections[name] += 1

        # Сохранение
        out_name = f"pred_{img_file.stem}.png"
        cv2.imwrite(str(save_path / out_name), annotated)

        if show:
            cv2.imshow("Detection", annotated)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break

    if show:
        cv2.destroyAllWindows()

    print(f"\nРезультаты сохранены в {save_path}")
    print(f"Детекции: {total_detections}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс детекции огня")
    parser.add_argument("--weights", required=True, help="Путь к весам")
    parser.add_argument("--source", required=True, help="Изображение или директория")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Размер входа")
    parser.add_argument("--device", type=str, help="Устройство")
    parser.add_argument("--save-dir", type=str, help="Директория сохранения")
    parser.add_argument("--show", action="store_true", help="Показать результат")
    args = parser.parse_args()

    predict(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_dir=args.save_dir,
        show=args.show,
    )
