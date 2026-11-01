"""
Обучение YOLOv8s-p2 для детекции огня в ИК-изображениях.

Использование:
    # Полное обучение
    python src/train.py

    # Sanity check (5 эпох)
    python src/train.py --epochs 5 --name sanity_check

    # Продолжение обучения
    python src/train.py --resume runs/detect/fire_v1/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.utils import get_device, load_config


def train(
    model_cfg: str = None,
    data_cfg: str = None,
    hyp_cfg: str = None,
    pretrained: str = "yolov8s.pt",
    resume: str = None,
    epochs: int = None,
    batch: int = None,
    imgsz: int = None,
    device: str = None,
    name: str = "fire_v1",
    freeze: int = 0,
):
    """
    Запуск обучения YOLOv8.

    Args:
        model_cfg: Путь к YAML архитектуры модели
        data_cfg: Путь к YAML датасета
        hyp_cfg: Путь к YAML гиперпараметров
        pretrained: Предобученные веса (yolov8s.pt)
        resume: Путь к чекпоинту для продолжения
        epochs: Число эпох (переопределяет hyp_cfg)
        batch: Размер батча (переопределяет hyp_cfg)
        imgsz: Размер входа (переопределяет hyp_cfg)
        device: Устройство ('0', 'mps', 'cpu')
        name: Название эксперимента
        freeze: Число слоёв backbone для заморозки
    """
    configs_dir = PROJECT_ROOT / "configs"

    # Загрузка гиперпараметров
    hyp_path = hyp_cfg or str(configs_dir / "hyp-fire.yaml")
    hyp = load_config(Path(hyp_path).name) if Path(hyp_path).exists() else {}

    # Определение устройства
    if device is None:
        device = get_device()
    print(f"Устройство: {device}")

    # Создание/загрузка модели
    if resume:
        print(f"Продолжение обучения: {resume}")
        model = YOLO(resume)
    else:
        model_yaml = model_cfg or str(configs_dir / "yolov8s-p2-fire.yaml")
        print(f"Архитектура: {model_yaml}")
        print(f"Предобученные веса: {pretrained}")
        model = YOLO(model_yaml).load(pretrained)

    # Параметры обучения (приоритет: CLI args > hyp.yaml)
    train_args = {
        "data": data_cfg or str(configs_dir / "dataset.yaml"),
        "epochs": epochs or hyp.get("epochs", 150),
        "batch": batch or hyp.get("batch", 16),
        "imgsz": imgsz or hyp.get("imgsz", 1024),
        "device": device,
        "project": str(PROJECT_ROOT / "runs" / "detect"),
        "name": name,
        "pretrained": True,
        "optimizer": hyp.get("optimizer", "AdamW"),
        "lr0": hyp.get("lr0", 0.001),
        "lrf": hyp.get("lrf", 0.01),
        "cos_lr": hyp.get("cos_lr", True),
        "momentum": hyp.get("momentum", 0.937),
        "weight_decay": hyp.get("weight_decay", 0.0005),
        "warmup_epochs": hyp.get("warmup_epochs", 5),
        "warmup_momentum": hyp.get("warmup_momentum", 0.8),
        "warmup_bias_lr": hyp.get("warmup_bias_lr", 0.1),
        "patience": hyp.get("patience", 30),
        "save_period": 10,
        "val": True,
        "plots": True,
        "exist_ok": True,
        # ИК-специфичные аугментации
        "hsv_h": hyp.get("hsv_h", 0.0),
        "hsv_s": hyp.get("hsv_s", 0.0),
        "hsv_v": hyp.get("hsv_v", 0.3),
        "degrees": hyp.get("degrees", 15.0),
        "translate": hyp.get("translate", 0.2),
        "scale": hyp.get("scale", 0.5),
        "shear": hyp.get("shear", 5.0),
        "flipud": hyp.get("flipud", 0.5),
        "fliplr": hyp.get("fliplr", 0.5),
        "mosaic": hyp.get("mosaic", 1.0),
        "mixup": hyp.get("mixup", 0.1),
        "copy_paste": hyp.get("copy_paste", 0.1),
        "close_mosaic": hyp.get("close_mosaic", 15),
        # Веса потерь
        "box": hyp.get("box", 7.5),
        "cls": hyp.get("cls", 0.5),
        "dfl": hyp.get("dfl", 1.5),
    }

    train_args["workers"] = 4  # меньше workers чтобы не упираться в shared memory

    if freeze > 0:
        train_args["freeze"] = freeze
        print(f"Заморозка первых {freeze} слоёв backbone")

    if resume:
        train_args["resume"] = True

    print(f"\nПараметры обучения:")
    print(f"  Эпохи: {train_args['epochs']}")
    print(f"  Батч: {train_args['batch']}")
    print(f"  Размер входа: {train_args['imgsz']}")
    print(f"  Оптимизатор: {train_args['optimizer']}")
    print(f"  LR: {train_args['lr0']}")
    print()

    # Запуск обучения
    results = model.train(**train_args)

    print("\nОбучение завершено!")
    print(f"Лучшие веса: {PROJECT_ROOT}/runs/detect/{name}/weights/best.pt")
    print(f"Последние веса: {PROJECT_ROOT}/runs/detect/{name}/weights/last.pt")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 для детекции огня в ИК")
    parser.add_argument("--model-cfg", type=str, help="Путь к YAML модели")
    parser.add_argument("--data-cfg", type=str, help="Путь к YAML датасета")
    parser.add_argument("--hyp-cfg", type=str, help="Путь к YAML гиперпараметров")
    parser.add_argument("--pretrained", type=str, default="yolov8s.pt", help="Предобученные веса")
    parser.add_argument("--resume", type=str, help="Чекпоинт для продолжения")
    parser.add_argument("--epochs", type=int, help="Число эпох")
    parser.add_argument("--batch", type=int, help="Размер батча")
    parser.add_argument("--imgsz", type=int, help="Размер входного изображения")
    parser.add_argument("--device", type=str, help="Устройство (0, mps, cpu)")
    parser.add_argument("--name", type=str, default="fire_v1", help="Название эксперимента")
    parser.add_argument("--freeze", type=int, default=0, help="Число слоёв для заморозки")
    args = parser.parse_args()

    train(
        model_cfg=args.model_cfg,
        data_cfg=args.data_cfg,
        hyp_cfg=args.hyp_cfg,
        pretrained=args.pretrained,
        resume=args.resume,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        freeze=args.freeze,
    )
