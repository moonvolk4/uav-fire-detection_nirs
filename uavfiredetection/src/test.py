"""
Финальная оценка модели на тестовой выборке.

Генерирует таблицу метрик для академического отчёта,
сохраняет все предсказания и детальный анализ.

Использование:
    python src/test.py --weights runs/detect/fire_v1/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.val import validate
from src.utils import CLASS_NAMES


def generate_report_table(metrics_path: str):
    """Генерация LaTeX/Markdown таблицы для отчёта."""
    import json

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Markdown таблица
    print("\n### Результаты на тестовой выборке\n")
    print("| Класс | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |")
    print("|-------|-----------|--------|----|---------|--------------|")

    for name, m in metrics.get("per_class", {}).items():
        p = m["precision"]
        r = m["recall"]
        f1 = 2 * p * r / (p + r + 1e-16)
        print(f"| {name} | {p:.3f} | {r:.3f} | {f1:.3f} | {m['ap50']:.3f} | {m['ap50_95']:.3f} |")

    print(f"| **Среднее** | — | — | — | **{metrics['map50']:.3f}** | **{metrics['map50_95']:.3f}** |")

    # LaTeX таблица
    print("\n% LaTeX таблица для отчёта")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Результаты детекции на тестовой выборке}")
    print("\\begin{tabular}{|l|c|c|c|c|c|}")
    print("\\hline")
    print("Класс & Precision & Recall & F1 & mAP@0.5 & mAP@0.5:0.95 \\\\")
    print("\\hline")

    for name, m in metrics.get("per_class", {}).items():
        p = m["precision"]
        r = m["recall"]
        f1 = 2 * p * r / (p + r + 1e-16)
        print(f"{name} & {p:.3f} & {r:.3f} & {f1:.3f} & {m['ap50']:.3f} & {m['ap50_95']:.3f} \\\\")

    print("\\hline")
    print(f"Среднее & — & — & — & {metrics['map50']:.3f} & {metrics['map50_95']:.3f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Тестирование модели на test split")
    parser.add_argument("--weights", required=True, help="Путь к весам")
    parser.add_argument("--data-cfg", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold для теста")
    args = parser.parse_args()

    save_dir = str(PROJECT_ROOT / "runs" / "test")

    print("=" * 60)
    print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ")
    print("=" * 60)

    # Валидация на тестовом split
    validate(
        weights=args.weights,
        data_cfg=args.data_cfg,
        split="test",
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        save_dir=save_dir,
    )

    # Генерация таблицы для отчёта
    metrics_file = Path(save_dir) / "metrics_test.json"
    if metrics_file.exists():
        generate_report_table(str(metrics_file))


if __name__ == "__main__":
    main()
