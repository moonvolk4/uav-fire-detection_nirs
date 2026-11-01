"""
Обучение YOLOv8s-p2-CBAM.

Регистрирует CBAM модуль в ultralytics и запускает обучение.
"""

import sys
sys.path.insert(0, "/app/src")

from cbam import CBAM

# Регистрируем CBAM везде где ultralytics ищет модули
import ultralytics.nn.modules as modules
import ultralytics.nn.modules.block as block
import ultralytics.nn.modules.conv as conv
import ultralytics.nn.tasks as tasks
import builtins

# Добавляем во все места где parse_model делает globals()[m]
for target in [modules, block, conv, tasks]:
    setattr(target, "CBAM", CBAM)

# globals() fallback
builtins.CBAM = CBAM

from ultralytics import YOLO

model = YOLO("configs/yolov8s-p2-cbam.yaml").load("yolov8s.pt")

model.train(
    data="configs/dataset_dfire.yaml",
    epochs=300,
    imgsz=640,
    batch=32,
    workers=2,
    device=0,
    project="runs/detect",
    name="dfire_v5_cbam",
    exist_ok=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    patience=50,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    close_mosaic=20,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    warmup_epochs=5,
    save_period=25,
    plots=True,
)
