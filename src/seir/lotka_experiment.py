"""Сравнение методов идентификации параметров модели Лотки—Вольтерры.

Запуск из корня проекта:
    python -m seir.lotka_experiment --runs 3 --trials 100

Параметры восстанавливаются по двум наблюдаемым траекториям: численности
жертв и хищников. Данные синтетические, поэтому истинные параметры известны.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})
import optuna
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .lotka_volterra import simulate

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Порядок параметров совпадает с lotka_volterra.simulate:
# alpha — рост жертв, beta — интенсивность хищничества,
# delta — рост хищников за счёт жертв, gamma — смертность хищников.
TRUE_PARAMETERS = np.array([1.0, 0.1, 0.075, 1.5], dtype=float)
INITIAL_STATE = np.array([40.0, 9.0], dtype=float)
TIME_GRID = np.linspace(0.0, 30.0, 151)
NOISE_RELATIVE_STD = 0.03
BOUNDS = [(0.2, 1.8), (0.02, 0.25), (0.02, 0.16), (0.5, 2.5)]
TOLERANCE = 1e-5
METHODS = ("NM", "DE", "CMA", "BO")
LABELS = {"NM": "Nelder–Mead", "DE": "Differential evolution",
          "CMA": "CMA-ES", "BO": "Optuna TPE"}


def run_simulation(parameters: np.ndarray | list[float]) -> np.ndarray:
    """Рассчитать обе наблюдаемые траектории для набора параметров."""
    return simulate(TIME_GRID.tolist(), INITIAL_STATE.tolist(), np.asarray(parameters, dtype=float).tolist())


def loss_function(parameters: np.ndarray | list[float], data: np.ndarray) -> float:
    """Нормированная MSE по жертвам и хищникам."""
    prediction = run_simulation(parameters)
    scale = np.maximum(np.mean(np.abs(data), axis=0), 1e-12)
    residual = (prediction - data) / scale
    value = float(np.mean(residual**2))
    return value if np.isfinite(value) else 1e12


def safe_loss(parameters: np.ndarray | list[float], data: np.ndarray) -> float:
    try:
        return loss_function(parameters, data)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError):
        return 1e12


def make_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Сгенерировать синтетические наблюдения с относительным гауссовым шумом."""
    clean = run_simulation(TRUE_PARAMETERS)
    rng = np.random.default_rng(seed)
    noise_scale = NOISE_RELATIVE_STD * np.maximum(clean, 1.0)
    noisy = np.maximum(clean + rng.normal(0.0, noise_scale), 0.0)
    return clean, noisy


def fit(method: str, data: np.ndarray, seed: int, trials: int) -> tuple[np.ndarray, float, int, bool]:
    """Оценить параметры одним из четырёх методов."""
    start = time.perf_counter()
    x0 = np.array([0.7, 0.18, 0.12, 2.1], dtype=float)
    bounds = BOUNDS

    if method == "NM":
        # Nelder–Mead не поддерживает bounds во всех версиях SciPy, поэтому
        # используем явное ограничение функции потерь.
        def bounded_loss(p: np.ndarray) -> float:
            if np.any(p < np.array([b[0] for b in bounds])) or np.any(p > np.array([b[1] for b in bounds])):
                return 1e12
            return safe_loss(p, data)

        result = minimize(bounded_loss, x0, method="Nelder-Mead", options={"xatol": TOLERANCE, "fatol": TOLERANCE})
        parameters, iterations, success = result.x, result.nit, bool(result.success)
    elif method == "DE":
        result = differential_evolution(
            safe_loss, bounds, args=(data,), tol=TOLERANCE,
            rng=np.random.default_rng(seed), polish=True,
        )
        parameters, iterations, success = result.x, result.nit, bool(result.success)
    elif method == "CMA":
        options = {"bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
                   "tolfun": TOLERANCE, "seed": seed + 1, "verbose": -9}
        strategy = cma.CMAEvolutionStrategy(x0, 0.25, options)
        while not strategy.stop():
            candidates = strategy.ask()
            strategy.tell(candidates, [safe_loss(p, data) for p in candidates])
        parameters, iterations, success = strategy.result.xbest, strategy.countiter, True
    elif method == "BO":
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        names = ("alpha", "beta", "delta", "gamma")

        def objective(trial: optuna.Trial) -> float:
            parameters = [trial.suggest_float(name, low, high)
                          for name, (low, high) in zip(names, bounds)]
            return safe_loss(parameters, data)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=trials)
        parameters = np.array([study.best_params[name] for name in names], dtype=float)
        iterations, success = trials, True
    else:
        raise ValueError(f"Неизвестный метод идентификации: {method}")

    return np.asarray(parameters, dtype=float), time.perf_counter() - start, int(iterations), success


def metrics(clean: np.ndarray, data: np.ndarray, parameters: np.ndarray) -> dict[str, float]:
    prediction = run_simulation(parameters)
    return {
        "parameter_error_%": float(np.mean(100.0 * np.abs(parameters - TRUE_PARAMETERS) / TRUE_PARAMETERS)),
        "RMSE": float(np.sqrt(mean_squared_error(clean, prediction))),
        "MAE": float(mean_absolute_error(clean, prediction)),
        "data_loss": loss_function(parameters, data),
    }


def run_experiment(runs: int, trials: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in range(runs):
        clean, data = make_data(run)
        for method in METHODS:
            parameters, elapsed, iterations, success = fit(method, data, run, trials)
            row: dict[str, object] = {
                "run": run, "method": LABELS[method], "time_s": elapsed,
                "iterations": iterations, "success": success,
                "alpha": parameters[0], "beta": parameters[1],
                "delta": parameters[2], "gamma": parameters[3],
            }
            row.update(metrics(clean, data, parameters))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_trajectories(results: pd.DataFrame) -> None:
    """Показать сравнение истинных, шумных и восстановленных траекторий."""
    clean, data = make_data(0)
    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1, 1), wspace=0.12, hspace=0.28)
    top_axis = figure.add_subplot(grid[0, 0])
    bottom_axis = figure.add_subplot(grid[1, 0], sharex=top_axis)
    axes = [top_axis, bottom_axis]
    table_axis = figure.add_subplot(grid[0, 1])
    time_axis = figure.add_subplot(grid[1, 1])
    names = ("Популяция жертвы", "Популяция хищника")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for index, (axis, name) in enumerate(zip(axes, names)):
        axis.scatter(TIME_GRID, data[:, index], s=14, color="0.65", alpha=0.55,
                     label="Шумные наблюдения")
        axis.plot(TIME_GRID, clean[:, index], "k--", linewidth=2.5,
                  label="Истинная траектория")
        for color, method in zip(colors, METHODS):
            row = results[(results["run"] == 0) & (results["method"] == LABELS[method])].iloc[0]
            parameters = [row["alpha"], row["beta"], row["delta"], row["gamma"]]
            prediction = run_simulation(parameters)
            axis.plot(TIME_GRID, prediction[:, index], color=color, linewidth=1.8,
                      label=LABELS[method])
        axis.set_ylabel(name)
        axis.set_title(name, loc="left")
        axis.grid(True, linestyle="--", alpha=0.45)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()

    error_columns = ("alpha", "beta", "delta", "gamma")
    error_matrix = []
    for method in METHODS:
        row = results[(results["run"] == 0) & (results["method"] == LABELS[method])].iloc[0]
        error_matrix.append([
            100.0 * abs(float(row[column]) - true) / true
            for column, true in zip(error_columns, TRUE_PARAMETERS)
        ])
    error_image = table_axis.imshow(error_matrix, cmap="YlOrRd", aspect="auto")
    table_axis.set_title("Ошибки восстановления, %", loc="left", pad=18)
    table_axis.set_xticks(range(len(error_columns)))
    table_axis.set_xticklabels(["α", "β", "δ", "γ"])
    table_axis.set_yticks(range(len(METHODS)))
    table_axis.set_yticklabels(["Н-М", "ДЭ", "CMA-ES", "Optuna"])
    table_axis.tick_params(axis="y", pad=8)
    table_axis.set_xlabel("Параметр")
    table_axis.set_ylabel("Метод")
    for row_index, row in enumerate(error_matrix):
        for column_index, value in enumerate(row):
            table_axis.text(column_index, row_index, f"{value:.2f}",
                            ha="center", va="center", color="black")
    figure.colorbar(error_image, ax=table_axis, label="Ошибка, %", pad=0.04)

    method_names = [LABELS[method] for method in METHODS]
    median_times = [
        float(results.loc[results["method"] == name, "time_s"].median())
        for name in method_names
    ]
    time_axis.bar(method_names, median_times, color=colors)
    time_axis.set_yscale("log")
    time_axis.set_title("Время работы алгоритмов", loc="left")
    time_axis.set_xlabel("Метод")
    time_axis.set_ylabel("Время, с")
    time_axis.grid(True, axis="y", linestyle="--", alpha=0.45)
    time_axis.tick_params(axis="x", rotation=25)
    for index, value in enumerate(median_times):
        time_axis.text(index, value, f"{value:.2g}", ha="center", va="bottom", fontsize=11)

    figure.legend(legend_handles, legend_labels, loc="center left",
                  ncol=3, fontsize=11, frameon=False)

    time_ticks = np.arange(0.0, TIME_GRID[-1] + 1.0, 5.0)
    for axis in axes:
        axis.set_xticks(time_ticks)
        axis.tick_params(axis="x", labelbottom=True)
    axes[0].set_xlabel("Время")
    axes[-1].set_xlabel("Время")
    figure.suptitle("Идентификация параметров модели Лотки—Вольтерры", fontsize=16)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="число независимых шумовых реализаций")
    parser.add_argument("--trials", type=int, default=100, help="число испытаний Optuna")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "lotka_results.csv")
    args = parser.parse_args()
    if args.runs < 1 or args.trials < 1:
        parser.error("--runs и --trials должны быть положительными")

    results = run_experiment(args.runs, args.trials)
    results.to_csv(args.output, index=False)
    plot_trajectories(results)
    summary = results.groupby("method", sort=False).agg(
        parameter_error_median=("parameter_error_%", "median"),
        RMSE_median=("RMSE", "median"),
        time_median_s=("time_s", "median"),
        success_rate=("success", "mean"),
    )
    print(summary.to_string(float_format="%.4f"))
    print(f"\nРезультаты сохранены в: {args.output}")


if __name__ == "__main__":
    main()
