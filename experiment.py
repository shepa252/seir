"""
Мультизапуск: устойчивость и точность восстановления параметров SEIR.

Использует модель и настройки из main.py (CMA-ES: sigma0=0.2, tolfun=1e-5;
Нелдер-Мид: tol=1e-5; ДЭ: tol=1e-5; Optuna(TPE): 300 испытаний).

Запуск:
    python experiment.py                      # 20 запусков на каждый старт
    python experiment.py --runs 5 --workers 1 # быстрый тест / честные замеры времени
    python experiment.py --plot-only          # перерисовать графики из results.csv
"""
import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import cma
from scipy.optimize import minimize, differential_evolution

from main import (run_simulation, loss_function, calculate_metrics, POPULATION_N, Y0,
                  TRUE_BETA, TRUE_ALPHA, TRUE_GAMMA, NOISE_STD, DAYS_DURATION,
                  BOUNDS, TOLERANCE, OPTUNA_TRIALS)

warnings.filterwarnings("ignore")

TRUE = np.array([TRUE_BETA, TRUE_ALPHA, TRUE_GAMMA])
TRUE_R0 = TRUE_BETA / TRUE_GAMMA
STARTS = {"good": [0.3, 0.2, 0.1], "bad": [0.1, 0.1, 0.1]}
METHODS = ["NM", "DE", "CMA", "BO"]
LABELS = {"NM": "Н-М", "DE": "ДЭ", "CMA": "CMA-ES", "BO": "БО\n(TPE)"}
T_GRID = np.linspace(0, DAYS_DURATION, DAYS_DURATION)


def safe_loss(p, *args):
    """MSE; если модель «взорвалась» (нефинитные значения), возвращаем большое число."""
    try:
        v = loss_function(p, *args)
    except ValueError:
        return 1e12
    return v if np.isfinite(v) else 1e12


def make_data(seed):
    I_true = run_simulation(TRUE, T_GRID, POPULATION_N, Y0)
    rng = np.random.default_rng(seed)
    return np.maximum(I_true + rng.normal(0, NOISE_STD, len(T_GRID)), 0.0)


def fit(method, x0, seed, args):
    nit, ok = np.nan, True
    t0 = time.perf_counter()
    if method == "NM":  # как в main.py: без ограничений
        r = minimize(safe_loss, x0, args=args, method="Nelder-Mead", tol=TOLERANCE)
        x, nit, ok = r.x, r.nit, bool(r.success)
    elif method == "DE":
        r = differential_evolution(safe_loss, BOUNDS, args=args, tol=TOLERANCE, seed=seed)
        x, nit = r.x, r.nit
    elif method == "CMA":
        opts = {"bounds": [[b[0] for b in BOUNDS], [b[1] for b in BOUNDS]],
                "tolfun": TOLERANCE, "seed": seed + 1, "verbose": -9}
        es = cma.CMAEvolutionStrategy(x0, 0.2, opts)
        while not es.stop():
            sols = es.ask()
            es.tell(sols, [safe_loss(s, *args) for s in sols])
        x, nit = es.result.xbest, es.countiter
    elif method == "BO":
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            p = [trial.suggest_float(n, 0.0, 1.0) for n in ("beta", "alpha", "gamma")]
            return safe_loss(p, *args)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        x = [study.best_params[n] for n in ("beta", "alpha", "gamma")]
        nit = OPTUNA_TRIALS
    dt = time.perf_counter() - t0
    return np.asarray(x, float), dt, nit, ok


def worker(task):
    start, seed = task
    I_noisy = make_data(seed)
    args = (T_GRID, POPULATION_N, Y0, I_noisy)
    rows = []
    for m in METHODS:
        x, dt, nit, ok = fit(m, STARTS[start], seed, args)
        try:
            met = calculate_metrics(I_noisy, run_simulation(x, T_GRID, POPULATION_N, Y0))
        except ValueError:
            met = {"RMSE": np.nan, "MAE": np.nan, "sMAPE": np.nan}
        rows.append(dict(start=start, seed=seed, method=m, beta=x[0], alpha=x[1], gamma=x[2],
                         time=dt, nit=nit, converged=ok,
                         rmse=met["RMSE"], mae=met["MAE"], smape=met["sMAPE"]))
    return rows


def add_errors(df):
    for k, n in enumerate(["beta", "alpha", "gamma"]):
        df[f"err_{n}"] = 100 * np.abs(df[n] - TRUE[k]) / TRUE[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["R0"] = df["beta"] / df["gamma"]
    df["R0"] = df["R0"].replace([np.inf, -np.inf], np.nan)
    df["err_R0"] = 100 * np.abs(df["R0"] - TRUE_R0) / TRUE_R0
    errs = df[["err_beta", "err_alpha", "err_gamma"]].fillna(1e6)
    df["fail"] = errs.max(axis=1) > 50
    return df


def summarize(df):
    rows = []
    for st in ["good", "bad"]:
        for m in METHODS:
            g = df[(df.start == st) & (df.method == m)]
            if g.empty:
                continue
            row = {"start": st, "method": m, "n": len(g)}
            for c in ["err_beta", "err_alpha", "err_gamma", "err_R0"]:
                row[c + "_med"] = g[c].median()
                row[c + "_iqr"] = g[c].quantile(.75) - g[c].quantile(.25)
            row["R0_med"] = g["R0"].median()
            row["fail_%"] = 100 * g["fail"].mean()
            row["time_med_s"] = g["time"].median()
            rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv("summary.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s.round(2).to_string(index=False))
    return s


def plot(df):
    cols = [("err_beta", "β"), ("err_alpha", "α"), ("err_gamma", "γ"), ("err_R0", "R₀ = β/γ")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharey=True)
    for i, (st, title) in enumerate([("good", "Хороший старт"), ("bad", "Плохой старт")]):
        for j, (c, name) in enumerate(cols):
            ax = axes[i, j]
            data = [np.clip(df[(df.start == st) & (df.method == m)][c].fillna(1e4).values, 1e-2, 1e4)
                    for m in METHODS]
            ax.boxplot(data, widths=0.6, flierprops=dict(markersize=3))
            ax.set_yscale("log")
            ax.axhline(50, color="r", ls=":", lw=1)
            ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=9)
            ax.grid(True, ls="--", alpha=0.4)
            if i == 0:
                ax.set_title(name, fontsize=14)
        axes[i, 0].set_ylabel(f"{title}\nОтносительная ошибка, %", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_param_errors.png", dpi=300)
    print("Сохранено: fig_param_errors.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()

    if a.plot_only:
        df = add_errors(pd.read_csv("results.csv"))
    else:
        tasks = [(st, s) for st in STARTS for s in range(a.runs)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = [r for chunk in ex.map(worker, tasks) for r in chunk]
        print(f"Готово за {time.time() - t0:.0f} с")
        df = pd.DataFrame(rows)
        df.to_csv("results.csv", index=False)  # сырые результаты КАЖДОГО запуска
        df = add_errors(df)
    summarize(df)
    plot(df)


if __name__ == "__main__":
    main()
