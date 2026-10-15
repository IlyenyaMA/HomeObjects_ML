from pathlib import Path
import csv
import math

import pandas as pd
import yaml


# ============================================================
# Настройки проекта
# Файл collect_experiments.py находится в корне проекта.
# ============================================================

ROOT = Path(__file__).resolve().parent

RUNS_DIR = ROOT / "runs" / "detect" / "runs" / "homeobjects"
REPORTS_DIR = ROOT / "reports"

CSV_PATH = REPORTS_DIR / "experiments.csv"
MD_PATH = ROOT / "REPORT.md"

EXPERIMENTS = [
    "E1_baseline",
    "E2_freeze10",
    "E3_augmentation",
    "E4_optim",
]

# Столбцы метрик в results.csv у Ultralytics YOLO.
METRICS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}

# Параметры, которые не описывают различия обучения
# между экспериментами.
IGNORE_FOR_DIFF = {
    "name",
    "project",
    "exist_ok",
    "save_dir",
}


# ============================================================
# Вспомогательные функции
# ============================================================

def read_yaml(path: Path) -> dict:
    """Читает YAML-файл и проверяет, что его содержимое — словарь."""
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Ошибка чтения YAML-файла {path}:\n{exc}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Ожидался YAML-словарь в файле {path}, "
            f"получено: {type(data).__name__}"
        )

    return data


def read_results(path: Path) -> pd.DataFrame:
    """Читает results.csv и проверяет наличие данных."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать CSV-файл {path}:\n{exc}") from exc

    # Убираем пробелы из названий столбцов.
    df.columns = [str(column).strip() for column in df.columns]

    if df.empty:
        raise ValueError(f"Файл результатов пуст: {path}")

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"В файле {path} обнаружены повторяющиеся столбцы: {duplicated}"
        )

    return df


def is_missing(value) -> bool:
    """Проверяет, является ли значение отсутствующим."""
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_value(value):
    """Приводит значения параметров к удобному для сравнения виду."""
    if is_missing(value):
        return None

    # Например, 1 и 1.0 считаются одинаковыми значениями.
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        number = float(value)

        if not math.isfinite(number):
            return str(value)

        if number.is_integer():
            return int(number)

        return number

    return value


def format_value(value) -> str:
    """Форматирует значение параметра для отчёта."""
    value = normalize_value(value)

    if value is None:
        return "null"

    return repr(value)


def format_metric(value) -> str:
    """Форматирует метрику до четырёх знаков после запятой."""
    if is_missing(value):
        return ""

    number = float(value)

    if not math.isfinite(number):
        return ""

    return f"{number:.4f}"


def argument_diff(args: dict, baseline_args: dict) -> str:
    """Возвращает отличия аргументов эксперимента от baseline."""
    differences = []

    all_keys = sorted(set(args) | set(baseline_args))

    for key in all_keys:
        if key in IGNORE_FOR_DIFF:
            continue

        baseline_value = normalize_value(baseline_args.get(key))
        experiment_value = normalize_value(args.get(key))

        if baseline_value != experiment_value:
            differences.append(
                f"{key}: {format_value(baseline_value)}"
                f" -> {format_value(experiment_value)}"
            )

    return "; ".join(differences) if differences else "Нет"


def numeric_column(df: pd.DataFrame, column: str, file_path: Path):
    """Преобразует столбец в числовой и сообщает об ошибке, если его нет."""
    if column not in df.columns:
        raise ValueError(
            f"В файле {file_path} отсутствует столбец '{column}'.\n"
            f"Доступные столбцы: {list(df.columns)}"
        )

    values = pd.to_numeric(df[column], errors="coerce")
    values = values.where(values.map(
        lambda x: pd.isna(x) or math.isfinite(float(x))
    ))

    return values


# ============================================================
# Сбор результатов одного эксперимента
# ============================================================

def collect_experiment(name: str, baseline_args: dict) -> dict:
    run_dir = RUNS_DIR / name

    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"

    for path in (args_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Не найден необходимый файл:\n{path}\n\n"
                f"Проверь имя эксперимента и расположение папок."
            )

    args = read_yaml(args_path)
    df = read_results(results_path)

    # Для выбора лучшей эпохи необходима метрика mAP50-95.
    map_column = METRICS["mAP50-95"]
    map_values = numeric_column(df, map_column, results_path)

    valid_indices = map_values.dropna().index

    if len(valid_indices) == 0:
        raise ValueError(
            f"В {results_path} нет корректных значений "
            f"'{map_column}'."
        )

    # Выбираем строку с максимальным mAP50-95.
    best_index = map_values.loc[valid_indices].idxmax()
    best_row = df.loc[best_index]

    # В results.csv Ultralytics обычно записывает epoch начиная с 1.
    # Поэтому используем значение из файла без прибавления единицы.
    if "epoch" in df.columns and not is_missing(best_row["epoch"]):
        epoch_value = float(best_row["epoch"])

        if not math.isfinite(epoch_value):
            best_epoch = ""
        elif epoch_value.is_integer():
            best_epoch = int(epoch_value)
        else:
            best_epoch = epoch_value
    else:
        # Не подменяем номер эпохи индексом строки.
        best_epoch = ""

    row = {
        "Название": name,
        "Отличия от baseline": (
            "Baseline"
            if name == "E1_baseline"
            else argument_diff(args, baseline_args)
        ),
    }

    # Метрики берём именно из строки лучшей эпохи.
    for output_name, source_column in METRICS.items():
        values = numeric_column(df, source_column, results_path)
        row[output_name] = format_metric(values.loc[best_index])

    row["Лучшая эпоха"] = best_epoch

    # В актуальных версиях Ultralytics столбец time обычно содержит
    # накопленное время обучения в секундах.
    # Берём последнее корректное значение, а не время лучшей эпохи.
    row["Время обучения, с"] = ""

    if "time" in df.columns:
        time_values = numeric_column(df, "time", results_path).dropna()

        if not time_values.empty:
            row["Время обучения, с"] = round(float(time_values.iloc[-1]), 2)

    return row


# ============================================================
# Формирование отчётов
# ============================================================

def main():
    baseline_path = RUNS_DIR / "E1_baseline" / "args.yaml"

    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Не найден файл параметров baseline:\n{baseline_path}"
        )

    baseline_args = read_yaml(baseline_path)

    rows = []

    for experiment_name in EXPERIMENTS:
        print(f"Обрабатывается эксперимент: {experiment_name}")

        row = collect_experiment(experiment_name, baseline_args)
        rows.append(row)

        print(
            f"  mAP50-95: {row['mAP50-95']}; "
            f"лучшая эпоха: {row['Лучшая эпоха']}"
        )

    columns = [
        "Название",
        "Отличия от baseline",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "Лучшая эпоха",
        "Время обучения, с",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV удобно открывать в Excel; utf-8-sig помогает корректно
    # отображать русские названия столбцов в Windows.
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Копия таблицы для Markdown-отчёта.
    lines = [
        "# Результаты экспериментов",
        "",
        "Метрики взяты из results.csv. Параметры экспериментов "
        "прочитаны из args.yaml.",
        "",
        "Лучшая эпоха — строка с максимальным mAP50-95. "
        "Время обучения — последнее корректное значение столбца time "
        "из results.csv, если этот столбец присутствует.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in rows:
        values = []

        for column in columns:
            value = str(row[column])
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)

        lines.append("| " + " | ".join(values) + " |")

    MD_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nГотово!")
    print(f"CSV-отчёт: {CSV_PATH}")
    print(f"Markdown-отчёт: {MD_PATH}")
    print(f"Экспериментов обработано: {len(rows)}")


if __name__ == "__main__":
    main()
