"""
ИК-специфичные аугментации для thermal imagery.

Стандартные цветовые аугментации (hue, saturation) не имеют смысла для
одноканальных ИК-изображений. Вместо этого используются:
- Вариации яркости/контраста (разные условия съёмки)
- Шум (тепловой шум сенсора)
- Размытие (расфокусировка оптики)
- Геометрические преобразования
"""

import albumentations as A
import cv2
import numpy as np


def get_ir_augmentation_pipeline(p: float = 0.5) -> A.Compose:
    """
    Аугментации для ИК-изображений с bbox-safe трансформациями.

    Args:
        p: Общая вероятность применения каждой аугментации

    Returns:
        albumentations.Compose с bbox_params для YOLO формата
    """
    return A.Compose(
        [
            # Яркость/контраст — имитация разных температурных условий и высот
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=p,
            ),
            # Гамма — нелинейный отклик сенсора
            A.RandomGamma(gamma_limit=(70, 130), p=p * 0.5),
            # CLAHE — адаптивное усиление контраста (случайное)
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=p * 0.3),
            # Гауссов шум — тепловой шум сенсора
            A.GaussNoise(var_limit=(10.0, 50.0), p=p),
            # Гауссово размытие — расфокусировка оптики
            A.GaussianBlur(blur_limit=(3, 7), p=p * 0.3),
            # Имитация более низкого разрешения сенсора
            A.Downscale(scale_min=0.5, scale_max=0.9, p=p * 0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def apply_ir_augmentation(
    image: np.ndarray,
    bboxes: list[list[float]],
    class_labels: list[int],
    pipeline: A.Compose = None,
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """
    Применение ИК-аугментаций к изображению и bbox.

    Args:
        image: BGR uint8 изображение
        bboxes: [[x_center, y_center, width, height], ...] в YOLO формате
        class_labels: [class_id, ...] для каждого bbox
        pipeline: Albumentations pipeline (создаётся по умолчанию)

    Returns:
        (augmented_image, augmented_bboxes, augmented_labels)
    """
    if pipeline is None:
        pipeline = get_ir_augmentation_pipeline()

    result = pipeline(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels,
    )

    return result["image"], result["bboxes"], result["class_labels"]
