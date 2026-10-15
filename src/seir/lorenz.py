"""Динамическая система Лоренца, известная примером хаотического поведения."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


def lorenz_derivatives(
    time: float,
    state: Sequence[float],
    sigma: float,
    rho: float,
    beta: float,
) -> tuple[float, float, float]:
    """Вычислить производные координат системы Лоренца."""
    del time  # Автономная система явно не зависит от времени.
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def simulate(
    times: Sequence[float],
    initial_state: Sequence[float] = (1.0, 1.0, 1.0),
    parameters: Sequence[float] = (10.0, 28.0, 8.0 / 3.0),
) -> FloatArray:
    """Рассчитать координаты (x, y, z) в заданные моменты времени.

    Параметры задаются в порядке (sigma, rho, beta).
    """
    time_points = np.asarray(times, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    rates = np.asarray(parameters, dtype=float)

    if time_points.ndim != 1 or time_points.size < 2:
        raise ValueError("Times должен содержать не менее двух значений в одномерной последовательности")
    if not np.all(np.isfinite(time_points)) or np.any(np.diff(time_points) <= 0):
        raise ValueError("Times должен содержать конечные значения в строго возрастающем порядке")
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial_state должен содержать три конечные координаты")
    if rates.shape != (3,) or not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Parameters должен содержать три конечных неотрицательных коэффициента")

    sigma, rho, beta = rates
    solution = solve_ivp(
        lambda time, state: lorenz_derivatives(time, state, sigma, rho, beta),
        (time_points[0], time_points[-1]),
        initial,
        t_eval=time_points,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Не удалось выполнить интегрирование системы Лоренца: {solution.message}")
    return solution.y.T


def main() -> None:
    parser = argparse.ArgumentParser(description="Симуляция динамической системы Лоренца")
    parser.add_argument("--duration", type=float, default=40.0, help="длительность симуляции")
    parser.add_argument("--steps", type=int, default=4000, help="число точек вывода")
    parser.add_argument("--sigma", type=float, default=10.0, help="параметр sigma")
    parser.add_argument("--rho", type=float, default=28.0, help="параметр rho")
    parser.add_argument("--beta", type=float, default=8.0 / 3.0, help="параметр beta")
    args = parser.parse_args()
    if args.duration <= 0 or args.steps < 2 or min(args.sigma, args.rho, args.beta) < 0:
        parser.error("Длительность должна быть положительной, число шагов — не меньше 2, параметры — неотрицательными")

    times = np.linspace(0.0, args.duration, args.steps)
    trajectory = simulate(times, parameters=(args.sigma, args.rho, args.beta))

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=0.7)
    axis.set(xlabel="x", ylabel="y", zlabel="z", title="Аттрактор Лоренца")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
