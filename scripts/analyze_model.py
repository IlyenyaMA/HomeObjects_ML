
from pathlib import Path

import csv
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO
import torch
from ultralytics.utils.metrics import box_iou


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "E4_optim"

CHECKPOINT_PATH = (
    ROOT / "runs" / "detect" / "runs" / "homeobjects"
    / EXPERIMENT / "weights" / "best.pt"
)
BASELINE_CONFIG_PATH = ROOT / "configs" / "train_baseline.yaml"
DATA_CONFIG_PATH = ROOT / "configs" / "data_split.yaml"
TEST_LIST_PATH = ROOT / "splits" / "test_10.txt"
REPORTS_DIR = ROOT / "reports"


def analyze_test_metrics(model):
    """Пункт 3.2: метрики тестовой выборки и таблица AP в PNG."""
    metrics = model.val(
        data=str(DATA_CONFIG_PATH),
        split="test",
    )
    box = metrics.box

    print("\n=== Метрики тестовой выборки ===")
    print(f"mAP50-95:  {box.map:.4f}")
    print(f"mAP50:     {box.map50:.4f}")
    print(f"mAP75:     {box.map75:.4f}")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall:    {box.mr:.4f}")
    print("AP по классам:", box.maps)

    class_maps = np.asarray(box.maps)
    class_indices = np.asarray(box.ap_class_index, dtype=int)

    class_rows = sorted(
        [
            (model.names[int(class_id)], float(class_maps[class_id]))
            for class_id in class_indices
        ],
        key=lambda item: item[1],
    )

    print("\n=== AP50-95 по классам: от худшего к лучшему ===")
    for name, ap in class_rows:
        print(f"{name:<20} {ap:.4f}")

    print("\n=== Три худших класса ===")
    for name, ap in class_rows[:3]:
        print(f"{name:<20} {ap:.4f}")

    print("\n=== Скорость (мс на изображение) ===")
    print(metrics.speed)

    print("\n=== Матрица ошибок ===")
    print(metrics.confusion_matrix.matrix)

    if not class_rows:
        print("Нет данных для построения таблицы AP.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(8, max(2.5, 0.45 * len(class_rows) + 1.2))
    )
    ax.axis("off")
    ax.set_title("AP50-95 по классам", fontsize=13, pad=18)

    table = ax.table(
        cellText=[[name, f"{ap:.4f}"] for name, ap in class_rows],
        colLabels=["Класс", "AP50-95"],
        cellLoc="center",
        loc="center",
        colWidths=[0.55, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for col in range(2):
        table[(0, col)].set_facecolor("#D9EAF7")
        table[(0, col)].set_text_props(weight="bold")

    for row in range(1, min(4, len(class_rows) + 1)):
        for col in range(2):
            table[(row, col)].set_facecolor("#F8D7DA")

    output_path = REPORTS_DIR / "class_ap50_95.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nТаблица AP сохранена: {output_path}")


def get_test_images():
    """Читает пути изображений из списка test_10.txt."""
    if not TEST_LIST_PATH.is_file():
        raise FileNotFoundError(f"Не найден список изображений: {TEST_LIST_PATH}")

    image_paths = []

    for line in TEST_LIST_PATH.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        line = line.strip()
        if not line:
            continue

        path = Path(line)
        if not path.is_absolute():
            candidates = [
                ROOT / path,
                ROOT / "data" / path,
                ROOT / "data" / "images" / path,
            ]
            path = next((p for p in candidates if p.is_file()), candidates[0])

        if not path.is_file():
            print(f"Предупреждение: изображение не найдено: {line}")
            continue

        image_paths.append(path.resolve())

    if not image_paths:
        raise FileNotFoundError(
            f"Не найдено доступных изображений из {TEST_LIST_PATH}"
        )

    return image_paths


def analyze_predictions(model):
    """Пункт 3.3: детекции, DataFrame в памяти и изображения в reports/."""
    image_paths = get_test_images()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    details = []

    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=0.25,
        iou=0.5,
        stream=True,
        verbose=False,
    )

    saved_count = 0
    MAX_IMAGES = 5

    for result in results:
        image_path = Path(result.path)
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Не удалось прочитать изображение: {image_path}")
            continue

        names = result.names
        orig_shape = result.orig_shape
        speed = result.speed
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            xywhn = boxes.xywhn.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for coords, normalized, confidence, class_id in zip(
                xyxy, xywhn, confidences, class_ids
            ):
                x1, y1, x2, y2 = map(int, coords)
                class_name = names[int(class_id)]

                # Устойчивый цвет для каждого класса (BGR для OpenCV).
                color = (
                    int((37 * (int(class_id) + 1)) % 256),
                    int((17 * (int(class_id) + 3)) % 256),
                    int((97 * (int(class_id) + 5)) % 256),
                )

                rows.append({
                    "изображение": image_path.name,
                    "имя класса": class_name,
                    "confidence": float(confidence),
                    "x1": float(coords[0]),
                    "y1": float(coords[1]),
                    "x2": float(coords[2]),
                    "y2": float(coords[3]),
                })

                details.append({
                    "изображение": image_path.name,
                    "класс_id": int(class_id),
                    "xywhn": normalized.tolist(),
                    "names": names,
                    "orig_shape": orig_shape,
                    "speed": speed,
                })

                # Собственная отрисовка, без result.plot().
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} {confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                label_y = max(y1, text_h + baseline)

                cv2.rectangle(
                    image,
                    (x1, label_y - text_h - baseline),
                    (x1 + text_w, label_y),
                    color,
                    -1,
                )
                cv2.putText(
                    image,
                    label,
                    (x1, label_y - baseline),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        
        if saved_count < MAX_IMAGES:
            output_image = REPORTS_DIR / f"{image_path.stem}_pred.png"
            cv2.imwrite(str(output_image), image)
            saved_count += 1

    columns = [
        "изображение", "имя класса", "confidence",
        "x1", "y1", "x2", "y2",
    ]
    detections_df = pd.DataFrame(rows, columns=columns)

    print(f"\nОбработано изображений: {len(image_paths)}")
    print(f"Всего детекций: {len(detections_df)}")
    print(f"Изображения с рамками сохранены в: {REPORTS_DIR}")

    if detections_df.empty:
        print("При conf=0.25 детекций не найдено.")
    else:
        print("\n=== Первые строки DataFrame детекций ===")
        print(detections_df.head(10).to_string(index=False))

    return detections_df, details

def manual_metrics_50(model, limit=50, conf_threshold=0.25, iou_threshold=0.5):
    """Пункт 3.4: ручной подсчёт TP, FP и FN на тестовых изображениях."""

    image_paths = get_test_images()[:limit]

    if not image_paths:
        raise FileNotFoundError("Нет тестовых изображений.")

    totals = {"TP": 0, "FP": 0, "FN": 0}
    processed = 0

    for image_path in image_paths:
        # YOLO labels: images/.../name.jpg -> labels/.../name.txt
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label_path = Path(*parts).with_suffix(".txt")
        else:
            label_path = None

        if label_path is None or not label_path.is_file():
            print(f"Нет разметки, пропуск: {image_path.name}")
            continue

        # Читаем ground-truth в формате YOLO:
        # class x_center y_center width height
        label_rows = []
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            values = line.split()
            if len(values) != 5:
                continue

            cls, xc, yc, w, h = map(float, values)
            label_rows.append([int(cls), xc, yc, w, h])

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Не удалось прочитать: {image_path.name}")
            continue

        height, width = image.shape[:2]

        # Переводим ground-truth из нормализованного xywh в xyxy.
        gt_boxes = []
        gt_classes = []

        for cls, xc, yc, w, h in label_rows:
            gt_boxes.append([
                (xc - w / 2) * width,
                (yc - h / 2) * height,
                (xc + w / 2) * width,
                (yc + h / 2) * height,
            ])
            gt_classes.append(cls)

        predictions = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=0.5,
            verbose=False,
        )[0]

        pred_boxes = predictions.boxes.xyxy.cpu()
        pred_classes = predictions.boxes.cls.cpu().long()

        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4)
        gt_classes = torch.tensor(gt_classes, dtype=torch.long)

        # Матрица IoU между предсказаниями и реальными объектами.
        if len(pred_boxes) and len(gt_boxes):
            ious = box_iou(pred_boxes, gt_boxes)
        else:
            ious = torch.zeros((len(pred_boxes), len(gt_boxes)))

        matched_gt = set()
        tp = 0
        fp = 0

        # Каждое предсказание может соответствовать только одному объекту.
        # Сначала рассматриваем предсказания с большей уверенностью.
        order = predictions.boxes.conf.cpu().argsort(descending=True)

        for pred_idx in order.tolist():
            best_gt = -1
            best_iou = iou_threshold

            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue

                if pred_classes[pred_idx] != gt_classes[gt_idx]:
                    continue

                overlap = float(ious[pred_idx, gt_idx])

                if overlap >= best_iou:
                    best_iou = overlap
                    best_gt = gt_idx

            if best_gt >= 0:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        totals["TP"] += tp
        totals["FP"] += fp
        totals["FN"] += fn
        processed += 1

        print(
            f"{image_path.name}: "
            f"TP={tp}, FP={fp}, FN={fn}"
        )

    if processed == 0:
        raise RuntimeError(
            "Не обработано ни одного изображения. "
            "Проверь расположение файлов labels и список test_10.txt."
        )

    precision = totals["TP"] / max(totals["TP"] + totals["FP"], 1)
    recall = totals["TP"] / max(totals["TP"] + totals["FN"], 1)

    print("\n=== Ручные метрики ===")
    print(f"Обработано изображений: {processed}")
    print(f"TP: {totals['TP']}")
    print(f"FP: {totals['FP']}")
    print(f"FN: {totals['FN']}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "images_processed": processed,
    }


def log_epoch(trainer):
    """Записывает метрики и learning rate в CSV."""
    csv_path = REPORTS_DIR / "training_callback.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "epoch": trainer.epoch + 1,
        **(trainer.metrics or {}),
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
    }

    write_header = (
        not csv_path.exists() or csv_path.stat().st_size == 0
    )

    with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    print(f"Эпоха {row['epoch']}: LR={row['learning_rate']:.6g}")

def train_with_callback(model):
    """Отдельный запуск обучения на 3 эпохи с callback."""
    csv_path = REPORTS_DIR / "training_callback.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("", encoding="utf-8")

    model.add_callback("on_fit_epoch_end", log_epoch)

    model.train(
        data=str(DATA_CONFIG_PATH),
        epochs=3,
        project=str(ROOT / "runs" / "detect" / "runs" / "homeobjects"),
        name="Callbacks",
        fraction=0.3,
        exist_ok=True,
    )

    print(f"CSV сохранён: {csv_path}")

def compare_onnx_export(model, limit=20):
    """Пункт 3.6: экспорт ONNX и сравнение с PyTorch."""
    import pandas as pd
    from ultralytics import YOLO
    from ultralytics.utils.metrics import box_iou

    image_paths = get_test_images()[:limit]
    if not image_paths:
        raise FileNotFoundError("Нет тестовых изображений для сравнения.")

    # Экспорт модели
    onnx_path = model.export(format="onnx", imgsz=640)
    onnx_model = YOLO(str(onnx_path))

    rows = []
    total_torch = 0
    total_onnx = 0
    coordinate_diffs = []

    for image_path in image_paths:
        torch_result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            iou=0.5,
            verbose=False,
        )[0]

        onnx_result = onnx_model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            iou=0.5,
            verbose=False,
        )[0]

        torch_boxes = torch_result.boxes
        onnx_boxes = onnx_result.boxes

        boxes_a = torch_boxes.xyxy.cpu()
        classes_a = torch_boxes.cls.cpu().long()
        conf_a = torch_boxes.conf.cpu()

        boxes_b = onnx_boxes.xyxy.cpu()
        classes_b = onnx_boxes.cls.cpu().long()
        conf_b = onnx_boxes.conf.cpu()

        count_a = len(boxes_a)
        count_b = len(boxes_b)
        total_torch += count_a
        total_onnx += count_b

        # Сопоставляем боксы одного класса по максимальному IoU.
        matched_a = set()
        matched_b = set()
        pairs = []

        if count_a and count_b:
            ious = box_iou(boxes_a, boxes_b)

            candidates = []
            for i in range(count_a):
                for j in range(count_b):
                    if classes_a[i] == classes_b[j]:
                        candidates.append((float(ious[i, j]), i, j))

            candidates.sort(reverse=True)

            for overlap, i, j in candidates:
                if i in matched_a or j in matched_b:
                    continue
                matched_a.add(i)
                matched_b.add(j)
                pairs.append((i, j))

                diff = (boxes_a[i] - boxes_b[j]).abs().max().item()
                coordinate_diffs.append(diff)

        rows.append({
            "изображение": image_path.name,
            "боксы PyTorch": count_a,
            "боксы ONNX": count_b,
            "совпавшие пары": len(pairs),
            "макс. расхождение координат, px": (
                max(
                    (
                        (boxes_a[i] - boxes_b[j]).abs().max().item()
                        for i, j in pairs
                    ),
                    default=float("nan"),
                )
            ),
        })

    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Сохраняем отчёт в PNG, без CSV.
    fig, ax = plt.subplots(
        figsize=(12, max(3, 0.4 * len(df) + 1.5))
    )
    ax.axis("off")
    ax.set_title("Сравнение PyTorch и ONNX", fontsize=13, pad=16)

    display_df = df.copy()
    display_df["макс. расхождение координат, px"] = (
        display_df["макс. расхождение координат, px"]
        .map(lambda x: f"{x:.4f}" if np.isfinite(x) else "нет пар")
    )

    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    output_path = REPORTS_DIR / "onnx_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Сравнение PyTorch и ONNX ===")
    print(f"Изображений проверено: {len(image_paths)}")
    print(f"Всего боксов PyTorch: {total_torch}")
    print(f"Всего боксов ONNX:    {total_onnx}")
    print(f"Совпавших пар:        {len(coordinate_diffs)}")

    if coordinate_diffs:
        print(
            "Максимальное расхождение координат (px): "
            f"{max(coordinate_diffs):.6f}"
        )
    else:
        print("Не удалось сопоставить боксы для сравнения координат.")

    print(f"Таблица сохранена: {output_path}")
    return df

def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Не найден чекпойнт: {CHECKPOINT_PATH}")

    if not BASELINE_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Не найден baseline-конфиг: {BASELINE_CONFIG_PATH}"
        )

    model = YOLO(str(CHECKPOINT_PATH))

    # Информация о модели
    print("Чекпойнт:", CHECKPOINT_PATH)
    print("\n=== Информация о модели ===")
    print("Классы:", model.names)
    print("Задача:", model.task)
    model.info()

    # Сравнение аргументов обучения с baseline
    train_args = model.ckpt.get("train_args") or {}

    with BASELINE_CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        baseline_args = yaml.safe_load(file) or {}

    print("\n=== Сравнение с train_baseline.yaml ===")
    for key, expected in baseline_args.items():
        actual = train_args.get(key)
        status = "OK" if actual == expected else "ОТЛИЧАЕТСЯ"
        print(
            f"{key}: конфиг={expected!r}; "
            f"checkpoint={actual!r} [{status}]"
        )

    analyze_test_metrics(model)
    detections_df, details = analyze_predictions(model)

    manual_metrics_50(model)

    RUN_CALLBACK_TRAINING = False
    if RUN_CALLBACK_TRAINING:
        train_with_callback(model)

    compare_onnx_export(model)

if __name__ == "__main__":
    main()
