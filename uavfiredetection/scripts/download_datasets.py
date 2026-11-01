"""
Скачивание и распаковка датасетов FLAME 2/3.

FLAME 2 и FLAME 3 доступны на IEEE DataPort (требуется бесплатная регистрация).
Скрипт предоставляет инструкции для ручного скачивания и автоматически
распаковывает скачанные архивы.

Если датасет доступен на Kaggle — скачивает автоматически через kaggle API.
"""

import argparse
import os
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

FLAME2_INFO = """
=== FLAME 2 Dataset ===
1. Перейдите на: https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset
2. Зарегистрируйтесь / войдите (бесплатно)
3. Скачайте архив и поместите его в:
   {dest}
4. Запустите этот скрипт снова с --extract
"""

FLAME3_INFO = """
=== FLAME 3 Dataset ===
Вариант 1 (IEEE DataPort):
1. Перейдите на: https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management
2. Скачайте и поместите архив в:
   {dest}

Вариант 2 (Kaggle — CV subset, проще):
1. Установите kaggle: pip install kaggle
2. Настройте API token: https://www.kaggle.com/docs/api
3. Запустите:
   kaggle datasets download -d brycehopkins/flame-3-computer-vision-subset-sycan-marsh -p {dest}
4. Запустите этот скрипт снова с --extract
"""


def extract_archives(directory: Path):
    """Распаковка всех zip-архивов в указанной директории."""
    zips = list(directory.glob("*.zip"))
    if not zips:
        print(f"Нет zip-архивов в {directory}")
        return

    for z in zips:
        print(f"Распаковка {z.name}...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(directory)
        print(f"  Готово: {z.name}")


def try_kaggle_download(dataset_slug: str, dest: Path):
    """Попытка скачать датасет через Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"Скачивание {dataset_slug} через Kaggle API...")
        dest.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_slug, path=str(dest), unzip=True)
        print("Скачивание завершено!")
        return True
    except ImportError:
        print("Kaggle API не установлен. Установите: pip install kaggle")
        return False
    except Exception as e:
        print(f"Ошибка Kaggle API: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Скачивание датасетов FLAME")
    parser.add_argument(
        "--dataset",
        choices=["flame2", "flame3", "all"],
        default="all",
        help="Какой датасет скачать",
    )
    parser.add_argument("--extract", action="store_true", help="Распаковать скачанные архивы")
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Попробовать скачать FLAME 3 через Kaggle API",
    )
    args = parser.parse_args()

    flame2_dir = DATA_DIR / "flame2"
    flame3_dir = DATA_DIR / "flame3"
    flame2_dir.mkdir(parents=True, exist_ok=True)
    flame3_dir.mkdir(parents=True, exist_ok=True)

    if args.extract:
        if args.dataset in ("flame2", "all"):
            extract_archives(flame2_dir)
        if args.dataset in ("flame3", "all"):
            extract_archives(flame3_dir)
        return

    if args.dataset in ("flame2", "all"):
        print(FLAME2_INFO.format(dest=flame2_dir))

    if args.dataset in ("flame3", "all"):
        if args.kaggle:
            success = try_kaggle_download(
                "brycehopkins/flame-3-computer-vision-subset-sycan-marsh",
                flame3_dir,
            )
            if success:
                return
        print(FLAME3_INFO.format(dest=flame3_dir))


if __name__ == "__main__":
    main()
