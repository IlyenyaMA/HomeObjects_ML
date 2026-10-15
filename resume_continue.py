
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

last_path = (
    ROOT / "runs" / "resume_test" / "resume_demo"
    / "weights" / "last.pt"
)

if not last_path.exists():
    raise FileNotFoundError(f"Не найден checkpoint: {last_path}")

model = YOLO(str(last_path))

# Эксперимент: пытаемся изменить lr0 при возобновлении.
model.train(resume=True, lr0=0.001, imgsz=640)
