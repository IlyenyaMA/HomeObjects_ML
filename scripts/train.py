
from copy import deepcopy
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "train_baseline.yaml"
MODEL_PATH = ROOT / "yolo26n.pt"

# Все эксперименты используют общие настройки из train_baseline.yaml.
# В каждом эксперименте меняются только указанные ниже параметры.
EXPERIMENTS = {
    "E1_baseline": {},

    "E2_freeze10": {
        "freeze": 10,
    },

    "E3_augmentation": {
        "fliplr": 0.0,
        "mosaic": 0.0,
        "scale": 0.3,
    },

    "E4_optim": {
        "cos_lr": True,
        "lr0": 0.001,
        "optimizer": "auto",
    },
}


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    for name, overrides in EXPERIMENTS.items():
        print(f"\n{'=' * 60}")
        print(f"Запуск эксперимента: {name}")
        print(f"{'=' * 60}")

        # Не изменяем исходный словарь настроек.
        cfg = deepcopy(base_cfg)
        cfg.update(overrides)

        # Преобразуем относительные пути в абсолютные.
        data_path = Path(cfg["data"])
        if not data_path.is_absolute():
            cfg["data"] = str(ROOT / data_path)

        cfg["project"] = str(ROOT / "runs" / "homeobjects")
        cfg["name"] = name
        cfg["exist_ok"] = True

        # Для каждого эксперимента загружаем исходные pretrained-веса.
        model = YOLO(str(MODEL_PATH))

        print("Изменения:", overrides)
        print("Датасет:", cfg["data"])
        print("Результаты:", Path(cfg["project"]) / name)

        model.train(**cfg)

        print(f"Эксперимент {name} завершён.")


if __name__ == "__main__":
    main()
