
from pathlib import Path
from collections import Counter
import json
import math
import statistics

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

CLASSES = [
    "Кровать", "Диван", "Стул", "Стол", "Лампа", "Телевизор",
    "Ноутбук", "Шкаф", "Окно", "Дверь", "Растение в горшке",
    "Фоторамка",
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def analyze_split(split):
    images_dir = DATA / "images" / split
    labels_dir = DATA / "labels" / split

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Не найдены каталоги для сплита {split}")

    images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    labels = sorted(labels_dir.rglob("*.txt"))

    image_ids = {p.relative_to(images_dir).with_suffix("") for p in images}
    label_ids = {p.relative_to(labels_dir).with_suffix("") for p in labels}

    counts = Counter()
    errors = Counter()
    areas = []
    empty_labels = 0

    for label in labels:
        lines = [
            line.split()
            for line in label.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1
            continue

        for row in lines:
            if len(row) != 5:
                errors["wrong_field_count"] += 1
                continue

            try:
                class_id = int(row[0])
                x, y, w, h = map(float, row[1:])
            except ValueError:
                errors["invalid_values"] += 1
                continue

            if not all(math.isfinite(v) for v in (x, y, w, h)):
                errors["non_finite_coordinates"] += 1
                continue

            valid = True

            if not 0 <= class_id < len(CLASSES):
                errors["class_id_out_of_range"] += 1
                valid = False

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                errors["coordinates_out_of_range"] += 1
                valid = False

            if w <= 0 or h <= 0:
                errors["non_positive_size"] += 1
                valid = False

            if not valid:
                continue

            counts[class_id] += 1
            areas.append(w * h)

    return {
        "images": len(images),
        "label_files": len(labels),
        "empty_label_files": empty_labels,
        "images_without_labels": len(image_ids - label_ids),
        "labels_without_images": len(label_ids - image_ids),
        "total_boxes": sum(counts.values()),
        "boxes_by_class": {
            CLASSES[i]: counts[i] for i in range(len(CLASSES))
        },
        "box_area": {
            "min": min(areas) if areas else None,
            "median": statistics.median(areas) if areas else None,
            "max": max(areas) if areas else None,
            "small_boxes_count": sum(a < 0.01 for a in areas),
            "small_boxes_fraction": (
                sum(a < 0.01 for a in areas) / len(areas)
                if areas else None
            ),
        },
        "invalid_rows": dict(errors),
    }


def main():
    stats = {
        split: analyze_split(split)
        for split in ("train", "val")
    }

    total_counts = Counter()
    for result in stats.values():
        for class_id, name in enumerate(CLASSES):
            total_counts[name] += result["boxes_by_class"][name]

    stats["rarest_classes"] = [
        {"class": name, "boxes": count}
        for name, count in sorted(
            total_counts.items(), key=lambda item: (item[1], item[0])
        )[:3]
    ]

    (REPORTS / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    x = list(range(len(CLASSES)))
    width = 0.4

    plt.figure(figsize=(12, 6))
    for offset, split in ((-width / 2, "train"), (width / 2, "val")):
        plt.bar(
            [i + offset for i in x],
            [stats[split]["boxes_by_class"][name] for name in CLASSES],
            width,
            label=split.capitalize(),
        )

    plt.xticks(x, CLASSES, rotation=45, ha="right")
    plt.ylabel("Количество боксов")
    plt.title("Распределение классов датасета")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "class_distribution.png", dpi=150)
    plt.close()

    print(f"\nОтчёты сохранены в: {REPORTS}")


if __name__ == "__main__":
    main()
