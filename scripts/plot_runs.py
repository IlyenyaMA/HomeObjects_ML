
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Корневая папка проекта — на уровень выше scripts.
ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR = ROOT / "runs" / "detect" / "runs" / "homeobjects"
REPORTS_DIR = ROOT / "reports"

EXPERIMENTS = [
    "E1_baseline",
    "E2_freeze10",
    "E3_augmentation",
    "E4_optim",
]

METRIC = "metrics/mAP50-95(B)"
OUTPUT_PATH = REPORTS_DIR / "map50_95_comparison.png"


def load_results(experiment_name: str) -> pd.DataFrame:
    """Загружает и проверяет результаты одного эксперимента."""
    results_path = RUNS_DIR / experiment_name / "results.csv"

    if not results_path.is_file():
        raise FileNotFoundError(
            f"Не найден файл результатов:\n{results_path}"
        )

    df = pd.read_csv(results_path, encoding="utf-8-sig")
    df.columns = [column.strip() for column in df.columns]

    required_columns = {"epoch", METRIC}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"В файле {results_path} отсутствуют столбцы: "
            f"{sorted(missing_columns)}\n"
            f"Доступные столбцы: {list(df.columns)}"
        )

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")

    # Удаляем только строки, в которых нет номера эпохи или метрики.
    df = df.dropna(subset=["epoch", METRIC])
    df = df.sort_values("epoch")

    if df.empty:
        raise ValueError(
            f"В файле {results_path} нет корректных данных для графика."
        )

    return df


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted_experiments = 0

    for experiment_name in EXPERIMENTS:
        df = load_results(experiment_name)

        ax.plot(
            df["epoch"],
            df[METRIC],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=experiment_name,
        )

        plotted_experiments += 1
        print(
            f"{experiment_name}: "
            f"{len(df)} эпох, "
            f"максимальный mAP50-95 = {df[METRIC].max():.4f}"
        )

    ax.set_title("Сравнение экспериментов по mAP50-95")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("mAP50-95")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"\nПостроено кривых: {plotted_experiments}")
    print(f"График сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
