"""
Скачивание и подготовка D-Fire датасета.

D-Fire: 21,000+ изображений с YOLO-аннотациями fire/smoke.
Уже размечен в YOLO формате — не нужна конвертация.

Источник: https://github.com/gaiasd/DFireDataset
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def download_dfire():
    """Скачивание D-Fire через git clone."""
    dfire_dir = DATA_DIR / "raw" / "dfire"
    dfire_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("СКАЧИВАНИЕ D-FIRE DATASET")
    print("=" * 60)

    # Проверяем есть ли уже
    if list(dfire_dir.rglob("*.jpg")) or list(dfire_dir.rglob("*.png")):
        print(f"Датасет уже скачан в {dfire_dir}")
        return dfire_dir

    print("\nСкачиваем D-Fire датасет...")
    os.system(f"pip install gdown 2>/dev/null")

    # D-Fire доступен через GitHub releases
    # Пробуем git clone
    ret = os.system(f"git clone https://github.com/gaiasd/DFireDataset.git {dfire_dir}/repo")
    if ret == 0:
        print("Скачано через git!")
        return dfire_dir

    print("\nАвтоматическое скачивание не удалось.")
    print("Скачай вручную:")
    print("1. Перейди: https://github.com/gaiasd/DFireDataset")
    print("2. Скачай train.zip и test.zip из Releases")
    print(f"3. Распакуй в: {dfire_dir}/")
    print("   Структура должна быть:")
    print(f"   {dfire_dir}/train/images/")
    print(f"   {dfire_dir}/train/labels/")
    print(f"   {dfire_dir}/test/images/")
    print(f"   {dfire_dir}/test/labels/")

    return dfire_dir


def prepare_dfire_dataset():
    """Подготовка D-Fire: копирование в processed с правильным split."""
    dfire_dir = DATA_DIR / "raw" / "dfire"
    processed = DATA_DIR / "processed"

    # Поиск images и labels
    train_imgs = None
    test_imgs = None

    # Ищем структуру
    for candidate in [dfire_dir, dfire_dir / "repo"]:
        if (candidate / "train" / "images").exists():
            train_imgs = candidate / "train" / "images"
            train_lbls = candidate / "train" / "labels"
            test_imgs = candidate / "test" / "images"
            test_lbls = candidate / "test" / "labels"
            break

    # Альтернативная структура
    if train_imgs is None:
        for candidate in dfire_dir.rglob("images"):
            if "train" in str(candidate).lower():
                train_imgs = candidate
                train_lbls = candidate.parent / "labels"
            elif "test" in str(candidate).lower():
                test_imgs = candidate
                test_lbls = candidate.parent / "labels"

    if train_imgs is None or not train_imgs.exists():
        print("Не найдена структура D-Fire!")
        print(f"Содержимое {dfire_dir}:")
        for item in sorted(dfire_dir.rglob("*"))[:30]:
            print(f"  {item.relative_to(dfire_dir)}")
        return False

    # Считаем файлы
    train_images = sorted(list(train_imgs.glob("*.jpg")) + list(train_imgs.glob("*.png")))
    test_images = sorted(list(test_imgs.glob("*.jpg")) + list(test_imgs.glob("*.png")))
    print(f"\nD-Fire: {len(train_images)} train, {len(test_images)} test")

    # Split: train → train + val (85/15)
    import random
    random.seed(42)
    random.shuffle(train_images)
    val_count = int(len(train_images) * 0.15)
    val_images = train_images[:val_count]
    train_images_final = train_images[val_count:]

    # Копирование
    splits = {
        "train": (train_images_final, train_lbls),
        "val": (val_images, train_lbls),
        "test": (test_images, test_lbls),
    }

    for split_name, (images, lbl_dir) in splits.items():
        out_img = processed / split_name / "images"
        out_lbl = processed / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        # Очистка старых файлов
        for f in out_img.glob("*"):
            f.unlink()
        for f in out_lbl.glob("*"):
            f.unlink()

        print(f"\n{split_name}: копирование {len(images)} изображений...")
        for img_file in images:
            shutil.copy2(img_file, out_img / img_file.name)
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                shutil.copy2(lbl_file, out_lbl / (img_file.stem + ".txt"))
            else:
                (out_lbl / (img_file.stem + ".txt")).write_text("")

    print(f"\nГотово! Данные в {processed}")
    return True


if __name__ == "__main__":
    download_dfire()
    prepare_dfire_dataset()
