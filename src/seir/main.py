import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cma
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch

# ==========================================
# 1. КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ==========================================
POPULATION_N = 10000
# 🔧 ИСПРАВЛЕНИЕ 1: I0 должен быть > 0, чтобы заражение началось!
# Если I=0, член beta*S*I/N равен 0, и эпидемия не стартует.
E0 = 1
I0 = 1
R0 = 0
S0 = POPULATION_N - E0 - I0 - R0
Y0 = (S0, E0, I0, R0)

TRUE_BETA = 0.65
TRUE_ALPHA = 0.3
TRUE_GAMMA = 0.1
NOISE_STD = 5.0
DAYS_DURATION = 100

INITIAL_GUESS = [0.1, 0.1, 0.1]
BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
TOLERANCE = 1e-5
OPTUNA_TRIALS = 300


def format_legend_number(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace('.', ',')


plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'lines.linewidth': 3
})


# ==========================================
# 2. МОДЕЛЬ SEIR
# ==========================================
def seir_derivatives(y, t, N, beta, alpha, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def run_simulation(params, t_grid, N, y0):
    beta, alpha, gamma = params
    solution = odeint(seir_derivatives, y0, t_grid, args=(N, beta, alpha, gamma))
    return solution.T[2]  # Возвращаем только I(t)


def loss_function(params, t_grid, N, y0, data_true):
    I_pred = run_simulation(params, t_grid, N, y0)
    return mean_squared_error(data_true, I_pred)


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    smape_vals = np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    smape_vals[np.isinf(smape_vals)] = 0.0  # Fix divide by zero
    smape = 100 * np.mean(smape_vals)

    return {"RMSE": rmse, "MAE": mae, "sMAPE": smape, "MSE": mse}


# ==========================================
# 3. ДАННЫЕ
# ==========================================
def generate_synthetic_data():
    t_grid = np.linspace(0, DAYS_DURATION, DAYS_DURATION)
    I_true = run_simulation([TRUE_BETA, TRUE_ALPHA, TRUE_GAMMA], t_grid, POPULATION_N, Y0)

    np.random.seed(42)
    noise = np.random.normal(0, NOISE_STD, size=len(t_grid))
    I_noisy = I_true + noise
    I_noisy[I_noisy < 0] = 0
    return t_grid, I_true, I_noisy


# ==========================================
# 4. ОПТИМИЗАТОРЫ
# ==========================================
def run_nelder_mead(args, t_grid, I_noisy):
    start_time = time.time()
    res = minimize(loss_function, INITIAL_GUESS, args=args, method='Nelder-Mead', tol=TOLERANCE)
    I_pred = run_simulation(res.x, t_grid, POPULATION_N, Y0)
    return {
        "name": "Нелдер-Мид", "params": res.x,
        "time": time.time() - start_time, "iters": res.nit,
        "metrics": calculate_metrics(I_noisy, I_pred), "color": "red", "style": "-"
    }


def run_differential_evolution(args, t_grid, I_noisy):
    start_time = time.time()
    res = differential_evolution(loss_function, BOUNDS, args=args, tol=TOLERANCE)
    I_pred = run_simulation(res.x, t_grid, POPULATION_N, Y0)
    return {
        "name": "Дифференциальная эволюция", "params": res.x,
        "time": time.time() - start_time, "iters": res.nit,
        "metrics": calculate_metrics(I_noisy, I_pred), "color": "blue", "style": "-."
    }


def run_optuna(args, t_grid, I_noisy):
    def objective(trial):
        b = trial.suggest_float("beta", 0.0, 1.0)
        a = trial.suggest_float("alpha", 0.0, 1.0)
        g = trial.suggest_float("gamma", 0.0, 1.0)
        return loss_function([b, a, g], *args)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    best_params = [study.best_trial.params['beta'], study.best_trial.params['alpha'], study.best_trial.params['gamma']]
    I_pred = run_simulation(best_params, t_grid, POPULATION_N, Y0)
    return {
        "name": "Байесовская оптимизация", "params": best_params,
        "time": time.time() - start_time, "iters": study.best_trial.number,
        "metrics": calculate_metrics(I_noisy, I_pred), "color": "green", "style": "--"
    }


def run_cma_es(args, t_grid, I_noisy):
    start_time = time.time()
    bounds_cma = [[b[0] for b in BOUNDS], [b[1] for b in BOUNDS]]
    # sigma0 = 0.2 убрано предупреждение
    es = cma.CMAEvolutionStrategy(INITIAL_GUESS, 0.2, options={'bounds': bounds_cma, 'tolfun': TOLERANCE})

    while not es.stop():
        solutions = es.ask()
        fitness = [loss_function(sol, *args) for sol in solutions]
        es.tell(solutions, fitness)

    best_params = es.result.xbest
    I_pred = run_simulation(best_params, t_grid, POPULATION_N, Y0)
    return {
        "name": "CMA-ES", "params": best_params,
        "time": time.time() - start_time, "iters": es.countiter,
        "metrics": calculate_metrics(I_noisy, I_pred), "color": "purple", "style": ":"
    }


# ==========================================
# 5. ВЫВОД И ГРАФИК
# ==========================================
def print_results_table(results):
    print("\n" + "=" * 135)
    print(f"{'Метод':<25} | {'Beta':<6} | {'Alpha':<6} | {'Gamma':<6} | {'Время':<6} | {'RMSE':<8} | {'MAE':<8} | {'sMAPE':<8}")
    print("-" * 135)
    print(f"{'ИСТИНА':<25} | {TRUE_BETA:<6.3f} | {TRUE_ALPHA:<6.3f} | {TRUE_GAMMA:<6.3f} | {'-':<6} | {'-':<10}")
    for r in results:
        p = r['params']
        print(f"{r['name']:<25} | {p[0]:<6.3f} | {p[1]:<6.3f} | {p[2]:<6.3f} | {r['time']:<6.2f} | {r['metrics']['RMSE']:<8.2f} | {r['metrics']['MAE']:<8.2f} | {r['metrics']['sMAPE']:<8.2f}")
    print("=" * 135 + "\n")


def plot_comparison(t_grid, I_noisy, I_true, results):
    fig, ax = plt.subplots(figsize=(14, 10))

    # Данные с шумом
    ax.scatter(t_grid, I_noisy, color='gray', alpha=0.4, s=30, label='Наблюдения (шум)')

    # Истинная модель
    ax.plot(t_grid, I_true, 'k:', linewidth=4, label='Истинная SEIR')

    # Результаты оптимизации
    for r in results:
        I_pred = run_simulation(r['params'], t_grid, POPULATION_N, Y0)
        smape_str = format_legend_number(r['metrics']['sMAPE'], 1)
        label = f"{r['name']} (sMAPE={smape_str}%)"

        ax.plot(t_grid, I_pred, color=r['color'], linestyle=r['style'], linewidth=3, label=label)

    # ==========================================
    # ВРЕЗКА (ZOOM) С ФИКСИРОВАННЫМИ КООРДИНАТАМИ
    # ==========================================
    # Создаем врезку с увеличением в 3 раза, расположение в верхнем правом углу
    ax_ins = zoomed_inset_axes(ax, zoom=2, loc=1, borderpad=3)
    # pos = ax_ins.get_position()
    # ax_ins.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height])

    # Делаем рамку врезки черной и толстой
    for spine in ax_ins.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Отображаем те же данные на врезке
    ax_ins.scatter(t_grid, I_noisy, color='gray', alpha=0.4, s=30)
    ax_ins.plot(t_grid, I_true, 'k:', linewidth=4)

    for r in results:
        I_pred = run_simulation(r['params'], t_grid, POPULATION_N, Y0)
        ax_ins.plot(t_grid, I_pred, color=r['color'], linestyle=r['style'], linewidth=3)

    # 🔧 ФИКСИРОВАННЫЕ КООРДИНАТЫ ВРЕЗКИ (измените под свои нужды)
    x1, x2 = 35, 50  # Диапазон по оси X (дни)
    y1, y2 = 3200, 4600  # Диапазон по оси Y (число инфицированных)

    ax_ins.set_xlim(x1, x2)
    ax_ins.set_ylim(y1, y2)

    # Убираем деления на врезке для чистоты
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])

    # Рисуем рамку на основном графике
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         facecolor='none', edgecolor='0.5', linestyle='-', linewidth=1)
    ax.add_patch(rect)

    # Соединительные линии между рамкой и врезкой
    conn1 = ConnectionPatch(xyA=(x2, y2), coordsA=ax.transData,
                            xyB=(0, 1), coordsB=ax_ins.transAxes,
                            axesA=ax, axesB=ax_ins,
                            color="0.5", linestyle="-", linewidth=1)
    ax.add_artist(conn1)

    conn2 = ConnectionPatch(xyA=(x2, y1), coordsA=ax.transData,
                            xyB=(0, 0), coordsB=ax_ins.transAxes,
                            axesA=ax, axesB=ax_ins,
                            color="0.5", linestyle="-", linewidth=1)
    ax.add_artist(conn2)
    # ==========================================

    # Оформление графика
    ax.set_xlabel('Время (дни)', fontsize=22)
    ax.set_ylabel('Число инфицированных', fontsize=22)
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# def plot_comparison(t_grid, I_noisy, I_true, results):
#     fig, ax = plt.subplots(figsize=(14, 10))
#
#     # Данные с шумом
#     ax.scatter(t_grid, I_noisy, color='gray', alpha=0.4, s=30, label='Наблюдения (шум)')
#
#     # Истинная модель
#     ax.plot(t_grid, I_true, 'k:', linewidth=4, label='Истинная SEIR')
#
#     # Результаты оптимизации
#     for r in results:
#         I_pred = run_simulation(r['params'], t_grid, POPULATION_N, Y0)
#         smape_str = format_legend_number(r['metrics']['sMAPE'], 1)
#         label = f"{r['name']} (sMAPE={smape_str}%)"
#
#         ax.plot(t_grid, I_pred, color=r['color'], linestyle=r['style'], linewidth=3, label=label)
#
#     # Оформление графика
#     ax.set_xlabel('Время (дни)', fontsize=22)
#     ax.set_ylabel('Число инфицированных (I)', fontsize=22)
#     ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=16)
#     ax.grid(True, linestyle='--', alpha=0.6)
#
#     plt.tight_layout()
#     plt.show()


# ==========================================
# MAIN
# ==========================================
def main():
    t, I_true, I_noisy = generate_synthetic_data()
    opt_args = (t, POPULATION_N, Y0, I_noisy)

    results = []
    results.append(run_nelder_mead(opt_args, t, I_noisy))
    results.append(run_differential_evolution(opt_args, t, I_noisy))
    results.append(run_optuna(opt_args, t, I_noisy))
    results.append(run_cma_es(opt_args, t, I_noisy))

    print_results_table(results)
    plot_comparison(t, I_noisy, I_true, results)


if __name__ == "__main__":
    main()
