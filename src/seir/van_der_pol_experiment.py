"""Сравнение методов идентификации параметра осциллятора Ван-дер-Поля.

Запуск из корня проекта:
    python -m seir.van_der_pol_experiment --runs 3 --trials 100

Параметр mu восстанавливается по двум наблюдаемым траекториям:
положению и скорости осциллятора. Данные синтетические, поэтому истинный
параметр известен.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .van_der_pol import simulate

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRUE_PARAMETER = 1.0
INITIAL_STATE = np.array([2.0, 0.0], dtype=float)
TIME_GRID = np.linspace(0.0, 20.0, 201)
NOISE_RELATIVE_STD = 0.03
BOUNDS = [(0.1, 3.0)]
TOLERANCE = 1e-5
METHODS = ("NM", "DE", "CMA", "BO")
LABELS = {
    "NM": "Nelder–Mead",
    "DE": "Differential evolution",
    "CMA": "CMA-ES",
    "BO": "Optuna TPE",
}
DISPLAY_LABELS = {
    "NM": "Нелдер-Мид",
    "DE": "Дифференциальная эволюция",
    "CMA": "CMA-ES",
    "BO": "Байесовская оптимизация",
}


def run_simulation(parameters: np.ndarray | list[float]) -> np.ndarray:
    """Рассчитать положение и скорость для значения параметра mu."""
    rates = np.asarray(parameters, dtype=float).reshape(-1)
    if rates.size != 1:
        raise ValueError("Параметры должны содержать ровно одно значение mu")
    return simulate(TIME_GRID.tolist(), INITIAL_STATE.tolist(), float(rates[0]))


def loss_function(parameters: np.ndarray | list[float], data: np.ndarray) -> float:
    """Нормированная MSE по положению и скорости."""
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
    clean = run_simulation([TRUE_PARAMETER])
    rng = np.random.default_rng(seed)
    noise_scale = NOISE_RELATIVE_STD * np.maximum(np.abs(clean), 1.0)
    noisy = clean + rng.normal(0.0, noise_scale)
    return clean, noisy


def fit(method: str, data: np.ndarray, seed: int, trials: int) -> tuple[np.ndarray, float, int, bool]:
    """Оценить параметр mu одним из четырёх методов."""
    start = time.perf_counter()
    # Намеренно неидеальное начальное приближение для демонстрации методов.
    # x0 = np.array([1.5], dtype=float) # Наилучшее приближение
    x0 = np.array([2.5], dtype=float) # Наихудшее приближение
    bounds = BOUNDS

    if method == "NM":
        def bounded_loss(parameters: np.ndarray) -> float:
            if np.any(parameters < np.array([b[0] for b in bounds])) or np.any(
                parameters > np.array([b[1] for b in bounds])
            ):
                return 1e12
            return safe_loss(parameters, data)

        result = minimize(
            bounded_loss,
            x0,
            method="Nelder-Mead",
            options={"xatol": TOLERANCE, "fatol": TOLERANCE},
        )
        parameters, iterations, success = result.x, result.nit, bool(result.success)
    elif method == "DE":
        result = differential_evolution(
            safe_loss,
            bounds,
            args=(data,),
            tol=TOLERANCE,
            rng=np.random.default_rng(seed),
            polish=True,
        )
        parameters, iterations, success = result.x, result.nit, bool(result.success)
    elif method == "CMA":
        def bounded_loss(parameters: np.ndarray) -> float:
            if np.any(parameters < np.array([b[0] for b in bounds])) or np.any(
                parameters > np.array([b[1] for b in bounds])
            ):
                return 1e12
            return safe_loss(parameters, data)

        # Для одномерного CMA-ES ограничение bounds в некоторых версиях
        # библиотеки cma вызывает ошибку инициализации sigma_vec.
        # Ограничение задаём штрафом функции потерь.
        options = {
            "tolfun": TOLERANCE,
            "seed": seed + 1,
            "verbose": -9,
        }
        strategy = cma.CMAEvolutionStrategy(x0, 0.5, options)
        while not strategy.stop():
            candidates = strategy.ask()
            strategy.tell(candidates, [bounded_loss(p) for p in candidates])
        parameters, iterations, success = strategy.result.xbest, strategy.countiter, True
    elif method == "BO":
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            parameter = [trial.suggest_float("mu", BOUNDS[0][0], BOUNDS[0][1])]
            return safe_loss(parameter, data)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=trials)
        parameters = np.array([study.best_params["mu"]], dtype=float)
        iterations, success = trials, True
    else:
        raise ValueError(f"Неизвестный метод идентификации: {method}")

    return np.asarray(parameters, dtype=float), time.perf_counter() - start, int(iterations), success


def metrics(clean: np.ndarray, data: np.ndarray, parameters: np.ndarray) -> dict[str, float]:
    prediction = run_simulation(parameters)
    return {
        "parameter_error_%": float(100.0 * abs(parameters[0] - TRUE_PARAMETER) / TRUE_PARAMETER),
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
                "run": run,
                "method": LABELS[method],
                "time_s": elapsed,
                "iterations": iterations,
                "success": success,
                "mu": parameters[0],
            }
            row.update(metrics(clean, data, parameters))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_trajectories(results: pd.DataFrame) -> None:
    """Показать временные зависимости, фазовый портрет и сводные показатели."""
    clean, data = make_data(0)
    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1, 1), wspace=0.12, hspace=0.28)
    time_axis = figure.add_subplot(grid[0, 0])
    phase_axis = figure.add_subplot(grid[1, 0])
    table_axis = figure.add_subplot(grid[0, 1])
    time_bar_axis = figure.add_subplot(grid[1, 1])
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")

    time_axis.scatter(
        TIME_GRID, data[:, 0], s=14, color="0.65", alpha=0.55,
        label="Шумные наблюдения",
    )
    time_axis.plot(
        TIME_GRID, clean[:, 0], "k--", linewidth=2.5,
        label="Истинная траектория",
    )
    phase_axis.scatter(
        data[:, 0], data[:, 1], s=14, color="0.65", alpha=0.35,
        label="Шумные наблюдения",
    )
    phase_axis.plot(
        clean[:, 0], clean[:, 1], "k--", linewidth=2.5,
        label="Истинная траектория",
    )

    for color, method in zip(colors, METHODS):
        row = results[
            (results["run"] == 0) & (results["method"] == LABELS[method])
        ].iloc[0]
        prediction = run_simulation([row["mu"]])
        time_axis.plot(
            TIME_GRID, prediction[:, 0], color=color, linewidth=1.8,
            label=DISPLAY_LABELS[method],
        )
        phase_axis.plot(
            prediction[:, 0], prediction[:, 1], color=color, linewidth=1.8,
            label=DISPLAY_LABELS[method],
        )

    time_axis.set_title("Временные зависимости", loc="left", fontweight="bold")
    time_axis.set_xlabel("Время")
    time_axis.set_ylabel("Амплитуда x(t)")
    time_axis.set_xticks(np.arange(0, int(TIME_GRID[-1]) + 1, 5))
    time_axis.set_xlim(TIME_GRID[0], TIME_GRID[-1])
    time_axis.grid(True, linestyle="--", alpha=0.45)

    phase_axis.set_title("Фазовый портрет", loc="left", fontweight="bold")
    phase_axis.set_xlabel("x(t)")
    phase_axis.set_ylabel("ẋ(t)")
    phase_axis.grid(True, linestyle="--", alpha=0.45)

    legend_handles, legend_labels = time_axis.get_legend_handles_labels()

    error_matrix = []
    for method in METHODS:
        row = results[
            (results["run"] == 0) & (results["method"] == LABELS[method])
        ].iloc[0]
        error_matrix.append([100.0 * abs(float(row["mu"]) - TRUE_PARAMETER) / TRUE_PARAMETER])

    error_image = table_axis.imshow(error_matrix, cmap="YlOrRd", aspect="auto")
    table_axis.set_title(
        "Ошибка восстановления параметра", loc="left", pad=0, fontweight="bold"
    )
    table_axis.set_xticks([0])
    table_axis.set_xticklabels(["μ"])
    table_axis.set_yticks(range(len(METHODS)))
    table_axis.set_yticklabels([
        "Нелдер-\nМид", "Дифф.\nэволюция", "CMA-ES", "Байесов.\nоптимизация",
    ])
    table_axis.tick_params(axis="y", pad=8)
    table_axis.set_xlabel("Параметр")
    table_axis.set_ylabel("Метод")
    for row_index, row in enumerate(error_matrix):
        table_axis.text(
            0, row_index, f"{row[0]:.2f}", ha="center", va="center", color="black"
        )
    figure.colorbar(error_image, ax=table_axis, label="Относительная ошибка, %", pad=0.04)

    method_names = [LABELS[method] for method in METHODS]
    median_times = [
        float(results.loc[results["method"] == name, "time_s"].median())
        for name in method_names
    ]
    axis_labels = [
        "Нелдер-\nМид", "Дифференциальная\nэволюция", "CMA-ES",
        "Байесовская\nоптимизация",
    ]
    time_bar_axis.bar(range(len(method_names)), median_times, color=colors)
    time_bar_axis.set_xticks(range(len(method_names)))
    time_bar_axis.set_xticklabels(axis_labels)
    time_bar_axis.set_yscale("log")
    time_bar_axis.set_ylim(bottom=0.1, top=100)
    time_bar_axis.yaxis.set_major_locator(LogLocator(base=10))
    time_bar_axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    time_bar_axis.set_title("Время работы алгоритмов", loc="left", fontweight="bold")
    time_bar_axis.set_xlabel("Метод")
    time_bar_axis.set_ylabel("Время, с")
    time_bar_axis.grid(True, axis="y", linestyle="--", alpha=0.45)
    time_bar_axis.tick_params(axis="x", rotation=0)
    for index, value in enumerate(median_times):
        time_bar_axis.text(
            index, value, f"{value:.2g}", ha="center", va="bottom", fontsize=11
        )

    figure.legend(legend_handles, legend_labels, loc="center left", ncol=2,
                  fontsize=16, frameon=True, fancybox=True)
    figure.suptitle("Модель осциллятор Ван-дер-Поля", fontsize=20, fontweight="bold")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="число независимых шумовых реализаций")
    parser.add_argument("--trials", type=int, default=100, help="число испытаний Optuna")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "van_der_pol_results.csv")
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
