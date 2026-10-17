"""Сравнение методов идентификации параметров модели SEIR.

GRNN не используется. Параметры beta, alpha и gamma восстанавливаются по
синтетической траектории числа инфицированных с добавленным шумом.

Запуск из корня проекта:
    python -m seir.seir_experiment --runs 3 --trials 100
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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

POPULATION_N = 10_000.0
INITIAL_STATE = np.array([9_998.0, 1.0, 1.0, 0.0], dtype=float)
TRUE_PARAMETERS = np.array([0.65, 0.30, 0.10], dtype=float)
TIME_GRID = np.linspace(0.0, 100.0, 100)
NOISE_STD = 5.0
BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
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


def seir_derivatives(
    state: tuple[float, float, float, float] | list[float] | np.ndarray,
    time_point: float,
    population: float,
    beta: float,
    alpha: float,
    gamma: float,
) -> tuple[float, float, float, float]:
    """Уравнения SEIR из grnn.py без использования GRNN."""
    del time_point
    susceptible, exposed, infected, _recovered = state
    new_infections = beta * susceptible * infected / population
    return (
        -new_infections,
        new_infections - alpha * exposed,
        alpha * exposed - gamma * infected,
        gamma * infected,
    )


def run_simulation(parameters: np.ndarray | list[float]) -> np.ndarray:
    """Рассчитать траекторию числа инфицированных."""
    rates = np.asarray(parameters, dtype=float).reshape(-1)
    if rates.size != 3:
        raise ValueError("Параметры должны содержать beta, alpha и gamma")
    solution = odeint(
        seir_derivatives,
        INITIAL_STATE,
        TIME_GRID,
        args=(POPULATION_N, rates[0], rates[1], rates[2]),
    )
    infected = solution[:, 2]
    return infected if np.all(np.isfinite(infected)) else np.full_like(TIME_GRID, np.inf)


def loss_function(parameters: np.ndarray | list[float], data: np.ndarray) -> float:
    """Среднеквадратичная ошибка между наблюдаемой и расчётной I(t)."""
    prediction = run_simulation(parameters)
    value = float(mean_squared_error(data, prediction))
    return value if np.isfinite(value) else 1e12


def safe_loss(parameters: np.ndarray | list[float], data: np.ndarray) -> float:
    try:
        return loss_function(parameters, data)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError):
        return 1e12


def make_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Создать истинную и зашумлённую траектории SEIR."""
    clean = run_simulation(TRUE_PARAMETERS)
    rng = np.random.default_rng(seed)
    noisy = np.maximum(clean + rng.normal(0.0, NOISE_STD, size=clean.size), 0.0)
    return clean, noisy


def fit(
    method: str,
    data: np.ndarray,
    seed: int,
    trials: int,
) -> tuple[np.ndarray, float, int, bool]:
    """Оценить beta, alpha и gamma одним из четырёх методов."""
    start = time.perf_counter()
    initial_guess = np.array([0.9, 0.9, 0.9], dtype=float)
    bounds = BOUNDS

    if method == "NM":
        def bounded_loss(parameters: np.ndarray) -> float:
            if np.any(parameters < np.array([bound[0] for bound in bounds])) or np.any(
                parameters > np.array([bound[1] for bound in bounds])
            ):
                return 1e12
            return safe_loss(parameters, data)

        result = minimize(
            bounded_loss,
            initial_guess,
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
        options = {
            "bounds": [[bound[0] for bound in bounds], [bound[1] for bound in bounds]],
            "tolfun": TOLERANCE,
            "seed": seed + 1,
            "verbose": -9,
        }
        strategy = cma.CMAEvolutionStrategy(initial_guess, 0.2, options)
        while not strategy.stop():
            candidates = strategy.ask()
            strategy.tell(candidates, [safe_loss(candidate, data) for candidate in candidates])
        parameters, iterations, success = strategy.result.xbest, strategy.countiter, True
    elif method == "BO":
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            parameters = [
                trial.suggest_float(name, lower, upper)
                for name, (lower, upper) in zip(("beta", "alpha", "gamma"), bounds)
            ]
            return safe_loss(parameters, data)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=trials)
        parameters = np.array(
            [study.best_params[name] for name in ("beta", "alpha", "gamma")],
            dtype=float,
        )
        iterations, success = trials, True
    else:
        raise ValueError(f"Неизвестный метод идентификации: {method}")

    return np.asarray(parameters, dtype=float), time.perf_counter() - start, int(iterations), success


def metrics(clean: np.ndarray, data: np.ndarray, parameters: np.ndarray) -> dict[str, float]:
    prediction = run_simulation(parameters)
    return {
        "parameter_error_%": float(
            np.mean(100.0 * np.abs(parameters - TRUE_PARAMETERS) / TRUE_PARAMETERS)
        ),
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
                "beta": parameters[0],
                "alpha": parameters[1],
                "gamma": parameters[2],
            }
            row.update(metrics(clean, data, parameters))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(results: pd.DataFrame) -> None:
    """Построить асимметричную сетку из трёх панелей."""
    clean, data = make_data(0)
    figure = plt.figure(figsize=(20, 20), constrained_layout=True)
    grid = figure.add_gridspec(
        2, 2,
        height_ratios=(1.5, 1.0),
        width_ratios=(1.0, 1.0),
        wspace=0.18,
        hspace=0.32,
    )
    trajectory_axis = figure.add_subplot(grid[0, :])
    trajectory_axis.set_box_aspect(0.3)
    time_axis = figure.add_subplot(grid[1, 0])
    error_axis = figure.add_subplot(grid[1, 1])
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    predictions = []

    trajectory_axis.scatter(
        TIME_GRID, data, s=16, color="0.65", alpha=0.55, label="Шумные наблюдения"
    )
    trajectory_axis.plot(
        TIME_GRID, clean, "k--", linewidth=2.5, label="Истинная траектория"
    )
    for color, method in zip(colors, METHODS):
        row = results[
            (results["run"] == 0) & (results["method"] == LABELS[method])
        ].iloc[0]
        prediction = run_simulation([row["beta"], row["alpha"], row["gamma"]])
        predictions.append(prediction)
        trajectory_axis.plot(
            TIME_GRID,
            prediction,
            color=color,
            linewidth=1.8,
            label=DISPLAY_LABELS[method],
        )

    # trajectory_axis.set_title("Временная зависимость SEIR", loc="left", fontweight="bold")
    trajectory_axis.set_xlabel("Время")
    trajectory_axis.set_ylabel("Число инфицированных I(t)")
    trajectory_axis.set_xlim(TIME_GRID[0], TIME_GRID[-1])
    trajectory_axis.set_xticks(np.arange(0, int(TIME_GRID[-1]) + 1, 10))
    trajectory_axis.grid(True, linestyle="--", alpha=0.45)

    # Увеличенная вставка для интервала с 37-го по 53-й день.
    zoom_mask = (TIME_GRID >= 34.0) & (TIME_GRID <= 53.0)

    x1, x2 = 34.0, 53.0
    y1, y2 = 3200.0, 4070.0

    inset_axis = zoomed_inset_axes(trajectory_axis, zoom=2, loc="upper right", borderpad=3)
    inset_axis.scatter(TIME_GRID[zoom_mask], data[zoom_mask], s=16, color="0.65", alpha=0.55)
    inset_axis.plot(TIME_GRID[zoom_mask], clean[zoom_mask], "k--", linewidth=2.5)
    for color, prediction in zip(colors, predictions):
        inset_axis.plot(
            TIME_GRID[zoom_mask], prediction[zoom_mask],
            color=color, linewidth=1.8,
        )
    inset_axis.set_xlim(x1, x2)
    inset_axis.set_ylim(y1, y2)
    inset_axis.set_xticks([])
    inset_axis.set_yticks([])
    for spine in inset_axis.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)
    mark_inset(
        trajectory_axis, inset_axis,
        loc1=1, loc2=2, fc="none", ec="0.5", linewidth=1,
    )

    legend_handles, legend_labels = trajectory_axis.get_legend_handles_labels()

    method_names = [LABELS[method] for method in METHODS]
    median_times = [
        float(results.loc[results["method"] == name, "time_s"].median())
        for name in method_names
    ]
    axis_labels = [
        "Нелдер-\nМид",
        "Дифференциальная\nэволюция",
        "CMA-ES",
        "Байесовская\nоптимизация",
    ]
    time_axis.bar(range(len(method_names)), median_times, color=colors)
    time_axis.set_xticks(range(len(method_names)))
    time_axis.set_xticklabels(axis_labels)
    time_axis.set_yscale("log")
    time_axis.set_ylim(bottom=0.1, top=100)
    time_axis.yaxis.set_major_locator(LogLocator(base=10))
    time_axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    time_axis.set_title("Время работы алгоритмов", loc="left", fontweight="bold")
    time_axis.set_xlabel("Метод")
    time_axis.set_ylabel("Время, с")
    time_axis.grid(True, axis="y", linestyle="--", alpha=0.45)
    time_axis.tick_params(axis="x", rotation=0)
    for index, value in enumerate(median_times):
        time_axis.text(index, value, f"{value:.2g}", ha="center", va="bottom", fontsize=11)

    error_columns = ("beta", "alpha", "gamma")
    error_matrix = []
    for method in METHODS:
        row = results[
            (results["run"] == 0) & (results["method"] == LABELS[method])
        ].iloc[0]
        error_matrix.append([
            100.0 * abs(float(row[column]) - true) / true
            for column, true in zip(error_columns, TRUE_PARAMETERS)
        ])

    error_image = error_axis.imshow(error_matrix, cmap="YlOrRd", aspect="auto")
    error_axis.set_title("Матрица ошибок восстановления параметров", loc="left", pad=0, fontweight="bold")
    error_axis.set_xticks(range(len(error_columns)))
    error_axis.set_xticklabels(["β", "α", "γ"])
    error_axis.set_yticks(range(len(METHODS)))
    error_axis.set_yticklabels([
        "Нелдер-\nМид",
        "Дифф.\nэволюция",
        "CMA-ES",
        "Байесов.\nоптимизация",
    ])
    error_axis.set_xlabel("Параметр")
    error_axis.set_ylabel("Метод")
    for row_index, row in enumerate(error_matrix):
        for column_index, value in enumerate(row):
            error_axis.text(
                column_index, row_index, f"{value:.2f}",
                ha="center", va="center", color="black"
            )
    figure.colorbar(error_image, ax=error_axis, label="Относительная ошибка, %", pad=0.04)

    figure.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=6,
        fontsize=15,
        frameon=True,
        fancybox=True
    )
    figure.suptitle("Модель SEIR", fontsize=20, fontweight="bold")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="число независимых шумовых реализаций")
    parser.add_argument("--trials", type=int, default=100, help="число испытаний Optuna")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "seir_results.csv")
    args = parser.parse_args()
    if args.runs < 1 or args.trials < 1:
        parser.error("--runs и --trials должны быть положительными")

    results = run_experiment(args.runs, args.trials)
    results.to_csv(args.output, index=False)
    plot_comparison(results)
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
