"""
Объединение конвертированных датасетов и разбиение на train/val/test.

Собирает все пары image+label из разных источников,
выполняет стратифицированный split 70/15/15 и копирует в data/processed/.
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def collect_samples(source_dirs: list[tuple[str, str]]) -> list[dict]:
    """
    Собирает пары (image, label) из нескольких директорий.

    Args:
        source_dirs: список пар (images_dir, labels_dir)

    Returns:
        Список словарей {image_path, label_path, classes}
    """
    samples = []
    for img_dir, lbl_dir in source_dirs:
        img_path = Path(img_dir)
        lbl_path = Path(lbl_dir)

        if not img_path.exists():
            print(f"Пропуск: {img_dir} не существует")
            continue

        for img_file in sorted(img_path.glob("*.png")):
            lbl_file = lbl_path / (img_file.stem + ".txt")
            classes = set()

            if lbl_file.exists():
                for line in lbl_file.read_text().strip().split("\n"):
                    if line.strip():
                        cls_id = int(line.split()[0])
                        classes.add(cls_id)

            samples.append({
                "image": img_file,
                "label": lbl_file if lbl_file.exists() else None,
                "classes": classes,
                "has_fire": 0 in classes,
                "has_smoke": 1 in classes,
                "has_false": 2 in classes,
            })

    return samples


def stratified_split(samples: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """
    Стратифицированный split по наличию классов.

    Группирует по типу (fire, smoke, false_source, negative) и
    разбивает каждую группу пропорционально.
    """
    random.seed(seed)

    # Группировка
    groups = {"fire_only": [], "smoke_only": [], "fire_smoke": [], "false_source": [], "negative": []}
    for s in samples:
        if s["has_false"]:
            groups["false_source"].append(s)
        elif s["has_fire"] and s["has_smoke"]:
            groups["fire_smoke"].append(s)
        elif s["has_fire"]:
            groups["fire_only"].append(s)
        elif s["has_smoke"]:
            groups["smoke_only"].append(s)
        else:
            groups["negative"].append(s)

    splits = {"train": [], "val": [], "test": []}

    for group_name, group_samples in groups.items():
        random.shuffle(group_samples)
        n = len(group_samples)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits["train"].extend(group_samples[:n_train])
        splits["val"].extend(group_samples[n_train : n_train + n_val])
        splits["test"].extend(group_samples[n_train + n_val :])

    # Перемешиваем каждый split
    for split_name in splits:
        random.shuffle(splits[split_name])

    return splits


def copy_to_processed(splits: dict[str, list[dict]]):
    """Копирование файлов в data/processed/{train,val,test}/{images,labels}/."""
    for split_name, samples in splits.items():
        img_dir = PROCESSED_DIR / split_name / "images"
        lbl_dir = PROCESSED_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for s in tqdm(samples, desc=f"Копирование {split_name}"):
            shutil.copy2(s["image"], img_dir / s["image"].name)
            if s["label"] and s["label"].exists():
                shutil.copy2(s["label"], lbl_dir / (s["image"].stem + ".txt"))
            else:
                # Пустой label для negative samples
                (lbl_dir / (s["image"].stem + ".txt")).write_text("")


def print_statistics(splits: dict[str, list[dict]]):
    """Вывод статистики по датасету."""
    class_names = {0: "fire", 1: "smoke", 2: "false_source"}

    print("\n" + "=" * 60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)

    total = 0
    for split_name, samples in splits.items():
        print(f"\n{split_name.upper()}: {len(samples)} изображений")
        total += len(samples)

        class_counts = Counter()
        bbox_counts = []
        for s in samples:
            for cls in s["classes"]:
                class_counts[cls] += 1
            if s["label"] and s["label"].exists():
                lines = [l for l in s["label"].read_text().strip().split("\n") if l.strip()]
                bbox_counts.append(len(lines))
            else:
                bbox_counts.append(0)

        for cls_id, name in class_names.items():
            print(f"  {name}: {class_counts.get(cls_id, 0)} изображений")
        negatives = sum(1 for s in samples if not s["classes"])
        print(f"  negative (без объектов): {negatives}")
        if bbox_counts:
            print(f"  среднее bbox/изображение: {sum(bbox_counts)/len(bbox_counts):.1f}")

    print(f"\nВСЕГО: {total} изображений")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Объединение датасетов и split")
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Пары img_dir:lbl_dir через пробел (например: dir1/images:dir1/labels dir2/images:dir2/labels)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.sources:
        source_dirs = []
        for s in args.sources:
            parts = s.split(":")
            if len(parts) != 2:
                print(f"Неверный формат: {s}. Ожидается images_dir:labels_dir")
                return
            source_dirs.append((parts[0], parts[1]))
    else:
        # По умолчанию ищем в стандартных директориях
        base = PROJECT_ROOT / "data"
        source_dirs = []
        for subdir in ["flame2_converted", "flame3_converted", "false_sources"]:
            img_d = base / subdir / "images"
            lbl_d = base / subdir / "labels"
            if img_d.exists():
                source_dirs.append((str(img_d), str(lbl_d)))

        if not source_dirs:
            print("Не найдены конвертированные датасеты.")
            print("Сначала запустите convert_flame.py для конвертации.")
            print("Или укажите директории вручную: --sources img1:lbl1 img2:lbl2")
            return

    print(f"Источники данных: {len(source_dirs)}")
    samples = collect_samples(source_dirs)
    print(f"Найдено {len(samples)} пар image+label")

    if not samples:
        print("Нет данных для обработки!")
        return

    splits = stratified_split(samples, seed=args.seed)
    print_statistics(splits)
    copy_to_processed(splits)
    print(f"\nДанные сохранены в {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
