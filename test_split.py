from pathlib import Path
from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parent
settings.update({"datasets_dir": str(ROOT)})

model = YOLO(str(ROOT / "yolo26n.pt"))

metrics = model.val(
    data=str(ROOT / "configs" / "data_split.yaml"),
    split="test",
)

print(metrics.results_dict)
