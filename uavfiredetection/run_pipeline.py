"""
Единый скрипт для запуска всего pipeline: конвертация → merge → проверка → обучение.
Запуск: python run_pipeline.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def run(cmd: str):
    """Запуск команды с выводом в консоль."""
    print(f"\n{'='*60}")
    print(f">>> {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"ОШИБКА! Код: {result.returncode}")
        sys.exit(1)

def main():
    # Создаём все нужные директории заранее
    dirs = [
        "data/raw/flame3",
        "data/flame3_converted/images",
        "data/flame3_converted/labels",
        "data/flame2_converted/images",
        "data/flame2_converted/labels",
        "data/false_sources/images",
        "data/false_sources/labels",
        "data/processed/train/images",
        "data/processed/train/labels",
        "data/processed/val/images",
        "data/processed/val/labels",
        "data/processed/test/images",
        "data/processed/test/labels",
        "data/verification",
        "runs",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"  OK: {d}")

    # Проверяем что данные скачаны
    raw_flame3 = PROJECT_ROOT / "data" / "raw" / "flame3"
    files = list(raw_flame3.rglob("*.*"))
    if not files:
        print(f"\nОШИБКА: Папка data/raw/flame3/ пуста!")
        print("Скачай датасет с Kaggle и распакуй в data/raw/flame3/")
        print("https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh")
        sys.exit(1)
    print(f"\nНайдено {len(files)} файлов в data/raw/flame3/")

    # Шаг 1: Конвертация
    run(
        "python scripts/convert_flame.py"
        " --dataset flame3"
        " --raw-dir data/raw/flame3"
        " --output-images data/flame3_converted/images"
        " --output-labels data/flame3_converted/labels"
    )

    # Шаг 2: Merge + split
    run("python scripts/merge_datasets.py")

    # Шаг 3: Проверка
    run("python scripts/verify_dataset.py")

    print(f"\n{'='*60}")
    print("ГОТОВО! Датасет подготовлен.")
    print(f"{'='*60}")
    print("\nДля обучения запусти:")
    print("  python src/train.py --epochs 5 --name sanity_check")
    print("\nДля полного обучения:")
    print("  python src/train.py")


if __name__ == "__main__":
    main()
