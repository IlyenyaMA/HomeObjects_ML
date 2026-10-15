
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "images" / "train"
SPLITS = ROOT / "splits"
SPLITS.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted(
    p for p in IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

if len(images) < 2:
    raise ValueError("Для разделения нужно не менее двух изображений.")

# Фиксированный seed обеспечивает воспроизводимое разбиение
random.Random(14).shuffle(images)

test_count = round(len(images) * 0.1)
test_images = images[:test_count]
train_images = images[test_count:]


def save_paths(filename, paths):
    lines = [
        p.relative_to(ROOT).as_posix()
        for p in paths
    ]

    (SPLITS / filename).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


save_paths("train_90.txt", train_images)
save_paths("test_10.txt", test_images)

print(f"Всего изображений: {len(images)}")
print(f"Train 90%: {len(train_images)}")
print(f"Test 10%: {len(test_images)}")
print("Списки сохранены в:", SPLITS)
print("Пример пути:", train_images[0].relative_to(ROOT / "data").as_posix())
