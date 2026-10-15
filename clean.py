
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

deleted = 0

if not REPORTS_DIR.is_dir():
    print(f"Папка не найдена: {REPORTS_DIR}")
else:
    for file_path in REPORTS_DIR.glob("*_pred.jpg"):
        if file_path.is_file():
            file_path.unlink()
            print(f"Удалён: {file_path.name}")
            deleted += 1

    print(f"\nУдалено файлов: {deleted}")