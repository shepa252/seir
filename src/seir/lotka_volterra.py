"""Модель Лотки—Вольтерры для динамики популяций хищников и жертв."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


def lotka_volterra_derivatives(
    time: float,
    state: Sequence[float],
    prey_growth: float,
    predation_rate: float,
    predator_growth: float,
    predator_death: float,
) -> tuple[float, float]:
    """Вычислить производные численности жертв и хищников."""
    del time  # Автономная система явно не зависит от времени.
    prey, predators = state
    return (
        prey_growth * prey - predation_rate * prey * predators,
        predator_growth * prey * predators - predator_death * predators,
    )


def simulate(
    times: Sequence[float],
    initial_state: Sequence[float] = (40.0, 9.0),
    parameters: Sequence[float] = (1.0, 0.1, 0.075, 1.5),
) -> FloatArray:
    """Рассчитать численность популяций в моменты времени ``times``.

    Параметры задаются в порядке (alpha, beta, delta, gamma).
    Уравнения модели:
    d(жертвы)/dt = alpha*жертвы - beta*жертвы*хищники;
    d(хищники)/dt = delta*жертвы*хищники - gamma*хищники.
    """
    time_points = np.asarray(times, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    rates = np.asarray(parameters, dtype=float)

    if time_points.ndim != 1 or time_points.size < 2:
        raise ValueError("Times должен содержать не менее двух значений в одномерной последовательности")
    if not np.all(np.isfinite(time_points)) or np.any(np.diff(time_points) <= 0):
        raise ValueError("Times должен содержать конечные значения в строго возрастающем порядке")
    if initial.shape != (2,) or not np.all(np.isfinite(initial)) or np.any(initial < 0):
        raise ValueError("Initial_state должен содержать две конечные неотрицательные численности")
    if rates.shape != (4,) or not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Parameters должен содержать четыре конечных неотрицательных коэффициента")

    alpha, beta, delta, gamma = rates
    solution = solve_ivp(
        lambda time, state: lotka_volterra_derivatives(
            time, state, alpha, beta, delta, gamma
        ),
        (time_points[0], time_points[-1]),
        initial,
        t_eval=time_points,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Не удалось выполнить интегрирование модели Лотки—Вольтерры: {solution.message}")
    return solution.y.T


def main() -> None:
    parser = argparse.ArgumentParser(description="Симуляция модели Лотки—Вольтерры")
    parser.add_argument("--duration", type=float, default=30.0, help="длительность симуляции")
    parser.add_argument("--steps", type=int, default=1000, help="число точек вывода")
    args = parser.parse_args()
    if args.duration <= 0 or args.steps < 2:
        parser.error("Длительность должна быть положительной, а число шагов — не меньше 2")

    times = np.linspace(0.0, args.duration, args.steps)
    populations = simulate(times)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(times, populations[:, 1], color="tab:red")
    axes[0].set(ylabel="Численность", title="Хищники")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(times, populations[:, 0], color="tab:green")
    axes[1].set(xlabel="Время", ylabel="Численность", title="Жертвы")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    figure.suptitle("Модель Лотки—Вольтерры")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
