"""Общие утилиты для проекта."""

import os
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CLASS_NAMES = {0: "fire", 1: "smoke", 2: "false_source"}


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_config(config_name: str) -> dict:
    """Загрузка YAML конфига из configs/."""
    config_path = PROJECT_ROOT / "configs" / config_name
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_class_weights(labels_dir: str) -> dict[int, float]:
    """
    Вычисление весов классов для борьбы с дисбалансом.
    Возвращает inverse frequency weights.
    """
    class_counts = Counter()
    total_bboxes = 0

    for lbl_file in Path(labels_dir).glob("*.txt"):
        for line in lbl_file.read_text().strip().split("\n"):
            if line.strip():
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1
                total_bboxes += 1

    if total_bboxes == 0:
        return {}

    n_classes = max(class_counts.keys()) + 1
    weights = {}
    for cls_id in range(n_classes):
        count = class_counts.get(cls_id, 1)
        weights[cls_id] = total_bboxes / (n_classes * count)

    return weights


def plot_bbox_size_distribution(labels_dir: str, img_size: tuple = (640, 512), save_path: str = None):
    """Гистограмма размеров bounding boxes."""
    widths, heights, areas = [], [], []

    for lbl_file in Path(labels_dir).glob("*.txt"):
        for line in lbl_file.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                w = float(parts[3]) * img_size[0]
                h = float(parts[4]) * img_size[1]
                widths.append(w)
                heights.append(h)
                areas.append(w * h)

    if not areas:
        print("Нет bounding boxes")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(widths, bins=50, color="steelblue", edgecolor="black")
    axes[0].set_title("Ширина bbox (px)")
    axes[0].set_xlabel("Ширина")

    axes[1].hist(heights, bins=50, color="coral", edgecolor="black")
    axes[1].set_title("Высота bbox (px)")
    axes[1].set_xlabel("Высота")

    axes[2].hist(areas, bins=50, color="seagreen", edgecolor="black")
    axes[2].set_title("Площадь bbox (px^2)")
    axes[2].set_xlabel("Площадь")

    # Категории COCO: small < 32^2, medium < 96^2, large >= 96^2
    for ax in axes:
        ax.set_ylabel("Количество")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {save_path}")
    else:
        plt.show()
    plt.close()


def get_device() -> str:
    """Определение доступного устройства."""
    import torch

    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
