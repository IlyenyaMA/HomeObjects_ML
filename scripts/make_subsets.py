
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data"
DEST = ROOT / "data_furniture"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    7: 4,
}

CLASS_NAMES = {
    0: "bed",
    1: "sofa",
    2: "chair",
    3: "table",
    4: "wardrobe",
}

if DEST.exists():
    shutil.rmtree(DEST)

for split in ("train", "val"):
    images_source = SOURCE / "images" / split
    labels_source = SOURCE / "labels" / split
    images_dest = DEST / "images" / split
    labels_dest = DEST / "labels" / split

    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in images_source.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    for image in images:
        relative_path = image.relative_to(images_source)
        source_label = labels_source / relative_path.with_suffix(".txt")
        dest_image = images_dest / relative_path
        dest_label = labels_dest / relative_path.with_suffix(".txt")

        if not source_label.is_file():
            raise FileNotFoundError(
                f"Не найдена разметка для изображения: {image}"
            )

        dest_image.parent.mkdir(parents=True, exist_ok=True)
        dest_label.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image, dest_image)

        new_lines = []

        for line in source_label.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Некорректная строка разметки: {source_label}: {line}"
                )

            old_id = int(parts[0])

            if old_id in CLASS_MAP:
                parts[0] = str(CLASS_MAP[old_id])
                new_lines.append(" ".join(parts))

        dest_label.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8",
        )

    print(
        f"{split}: обработано изображений — {len(images)}"
    )

for split in ("train", "val"):
    images_dir = DEST / "images" / split
    labels_dir = DEST / "labels" / split

    images = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    for image in images:
        label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
        assert label.is_file(), f"Нет label-файла для {image}"

        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            class_id = int(line.split()[0])
            assert 0 <= class_id <= 4, (
                f"Недопустимый class id {class_id} в {label}"
            )

print("Проверка целостности пройдена.")
print("Классы нового датасета:", CLASS_NAMES)
print("Подмножество сохранено в:", DEST)
