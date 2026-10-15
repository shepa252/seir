"""Динамическая модель нелинейного осциллятора Ван-дер-Поля."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


def van_der_pol_derivatives(
    time: float,
    state: Sequence[float],
    mu: float,
) -> tuple[float, float]:
    """Вычислить производные координаты и скорости осциллятора."""
    del time  # Автономная система явно не зависит от времени.
    position, velocity = state
    return velocity, mu * (1.0 - position**2) * velocity - position


def simulate(
    times: Sequence[float],
    initial_state: Sequence[float] = (2.0, 0.0),
    mu: float = 1.0,
) -> FloatArray:
    """Рассчитать положение и скорость осциллятора в заданные моменты времени.

    Уравнение модели: x'' - mu * (1 - x**2) * x' + x = 0.
    Начальное состояние задаётся как (положение, скорость).
    """
    time_points = np.asarray(times, dtype=float)
    initial = np.asarray(initial_state, dtype=float)

    if time_points.ndim != 1 or time_points.size < 2:
        raise ValueError("Times должен содержать не менее двух значений в одномерной последовательности")
    if not np.all(np.isfinite(time_points)) or np.any(np.diff(time_points) <= 0):
        raise ValueError("Times должен содержать конечные значения в строго возрастающем порядке")
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial_state должен содержать два конечных значения: положение и скорость")
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("Mu должен быть конечным неотрицательным коэффициентом")

    solution = solve_ivp(
        lambda time, state: van_der_pol_derivatives(time, state, mu),
        (time_points[0], time_points[-1]),
        initial,
        t_eval=time_points,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Не удалось выполнить интегрирование осциллятора Ван-дер-Поля: {solution.message}")
    return solution.y.T


def main() -> None:
    parser = argparse.ArgumentParser(description="Симуляция осциллятора Ван-дер-Поля")
    parser.add_argument("--duration", type=float, default=20.0, help="длительность симуляции")
    parser.add_argument("--steps", type=int, default=2000, help="число точек вывода")
    parser.add_argument("--mu", type=float, default=1.0, help="параметр нелинейности")
    args = parser.parse_args()
    if args.duration <= 0 or args.steps < 2 or args.mu < 0:
        parser.error("Длительность должна быть положительной, число шагов — не меньше 2, а mu — неотрицательным")

    times = np.linspace(0.0, args.duration, args.steps)
    state = simulate(times, mu=args.mu)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(times, state[:, 0], label="Положение")
    axes[0].plot(times, state[:, 1], label="Скорость")
    axes[0].set(xlabel="Время", ylabel="Состояние", title="Осциллятор Ван-дер-Поля")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(state[:, 0], state[:, 1])
    axes[1].set(xlabel="Положение", ylabel="Скорость", title="Фазовый портрет")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
