
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "splits"

train_file = SPLITS / "train_90.txt"
test_file = SPLITS / "test_10.txt"


def read_paths(file_path):
    return {
        line.strip().replace("\\", "/")
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


train_paths = read_paths(train_file)
test_paths = read_paths(test_file)

overlap = train_paths & test_paths

print(f"Уникальных путей в train_90: {len(train_paths)}")
print(f"Уникальных путей в test_10: {len(test_paths)}")

if overlap:
    print(f"\nОШИБКА: найдено пересечений: {len(overlap)}")
    for path in sorted(overlap):
        print(path)
else:
    print("\nУСПЕХ: пересечений между train_90 и test_10 нет.")

if len(train_paths) != len([
    line for line in train_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]):
    print("ВНИМАНИЕ: в train_90.txt есть повторяющиеся пути.")

if len(test_paths) != len([
    line for line in test_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]):
    print("ВНИМАНИЕ: в test_10.txt есть повторяющиеся пути.")
