from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# Укажи ту же модель, которая использовалась в E1.
MODEL = "yolo26n.pt"

model = YOLO(MODEL)

model.train(
    data=str(ROOT / "configs" / "data_split.yaml"),
    epochs=10,
    imgsz=448,
    batch=4,
    device="cpu",
    workers=0,
    project=str(ROOT / "runs" / "resume_test"),
    name="resume_demo",
    exist_ok=True,
    save=True,
    fraction=0.3
)
