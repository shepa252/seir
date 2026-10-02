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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.patches import ConnectionPatch

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
FILE_NAME = 'volgograd05.csv'
POPULATION_N = 2521200.0  # Население Волгоградской области

INITIAL_GUESS = [0.3, 0.2, 0.05]  # [beta, alpha, gamma]
BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
TOLERANCE = 1e-5
OPTUNA_TRIALS = 300

def format_legend_number(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace('.', ',')

plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24,
    'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20, 'lines.linewidth': 3
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
    return solution.T[2]  # Возвращаем I(t) - активные случаи

def loss_function(params, t_grid, N, y0, data_true):
    I_pred = run_simulation(params, t_grid, N, y0)
    return mean_squared_error(data_true, I_pred)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    smape_vals = np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    smape_vals[np.isinf(smape_vals)] = 0.0
    smape = 100 * np.mean(smape_vals)
    return {"RMSE": rmse, "MAE": mae, "sMAPE": smape, "MSE": mse}

# ==========================================
# 3. ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ
# ==========================================
def load_real_data():
    print(f"--- Загрузка данных: {FILE_NAME} ---")
    df = pd.read_csv(FILE_NAME)
    region_data = df[df['Region'] == 'Volgograd_Oblast'].sort_values('Date').reset_index(drop=True)

    daily_conf = region_data['Daily_Confirmed'].values.astype(float)
    daily_rec = region_data['Daily_Recovered'].values.astype(float)
    daily_death = region_data['Daily_Deaths'].values.astype(float)

    cum_conf = np.cumsum(daily_conf)
    cum_rec = np.cumsum(daily_rec)
    cum_death = np.cumsum(daily_death)

    # Активные случаи = Подтверждено - Выздоровевшие - Умершие
    I_real = cum_conf - cum_rec - cum_death
    I_real[I_real < 0] = 0.0  # Защита от артефактов данных

    t_grid = np.linspace(0, len(I_real) - 1, len(I_real))
    N = POPULATION_N

    # Начальные условия для SEIR
    I0 = I_real[0]
    R0_val = cum_rec[0] + cum_death[0]  # В SEIR R включает выздоровевших и умерших
    E0 = max(I0 * 0.5, 10.0)            # Эвристика: латентный пул ~ половина активных
    S0 = N - E0 - I0 - R0_val
    Y0 = (S0, E0, I0, R0_val)

    print(f"Период: {len(t_grid)} дней. Активных в начале: {I0:.0f}")
    return t_grid, I_real, N, Y0, region_data['Date'].values

# ==========================================
# 4. ОПТИМИЗАТОРЫ
# ==========================================
def run_nelder_mead(args, t_grid, N, Y0, I_real):
    start_time = time.time()
    res = minimize(loss_function, INITIAL_GUESS, args=args, method='Nelder-Mead', tol=TOLERANCE)
    I_pred = run_simulation(res.x, t_grid, N, Y0)
    return {"name": "Нелдер-Мид", "params": res.x, "time": time.time() - start_time, "iters": res.nit,
            "metrics": calculate_metrics(I_real, I_pred), "color": "red", "style": "-"}

def run_differential_evolution(args, t_grid, N, Y0, I_real):
    start_time = time.time()
    res = differential_evolution(loss_function, BOUNDS, args=args, tol=TOLERANCE)
    I_pred = run_simulation(res.x, t_grid, N, Y0)
    return {"name": "Диф. эволюция", "params": res.x, "time": time.time() - start_time, "iters": res.nit,
            "metrics": calculate_metrics(I_real, I_pred), "color": "blue", "style": "-."}

def run_optuna(args, t_grid, N, Y0, I_real):
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
    I_pred = run_simulation(best_params, t_grid, N, Y0)
    return {"name": "Байесовская опт.", "params": best_params, "time": time.time() - start_time,
            "iters": study.best_trial.number, "metrics": calculate_metrics(I_real, I_pred), "color": "green", "style": "--"}

def run_cma_es(args, t_grid, N, Y0, I_real):
    start_time = time.time()
    bounds_cma = [[b[0] for b in BOUNDS], [b[1] for b in BOUNDS]]
    es = cma.CMAEvolutionStrategy(INITIAL_GUESS, 0.2, options={'bounds': bounds_cma, 'tolfun': TOLERANCE})
    while not es.stop():
        solutions = es.ask()
        fitness = [loss_function(sol, *args) for sol in solutions]
        es.tell(solutions, fitness)
    best_params = es.result.xbest
    I_pred = run_simulation(best_params, t_grid, N, Y0)
    return {"name": "CMA-ES", "params": best_params, "time": time.time() - start_time, "iters": es.countiter,
            "metrics": calculate_metrics(I_real, I_pred), "color": "purple", "style": ":"}

# ==========================================
# 5. ВЫВОД И ГРАФИК
# ==========================================
def print_results_table(results):
    print("\n" + "=" * 115)
    print(f"{'Метод':<25} | {'Beta':<6} | {'Alpha':<6} | {'Gamma':<6} | {'Время(с)':<8} | {'RMSE':<8} | {'MAE':<8} | {'sMAPE':<8}")
    print("-" * 115)
    for r in results:
        p = r['params']
        print(f"{r['name']:<25} | {p[0]:<6.3f} | {p[1]:<6.3f} | {p[2]:<6.3f} | {r['time']:<8.2f} | {r['metrics']['RMSE']:<8.2f} | {r['metrics']['MAE']:<8.2f} | {r['metrics']['sMAPE']:<8.2f}")
    print("=" * 115 + "\n")

def plot_comparison(t_grid, I_real, N, y0, results, dates):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(t_grid, I_real, color='black', alpha=0.7, s=60, label='COVID-19 (Факт)')

    for r in results:
        I_pred = run_simulation(r['params'], t_grid, N, y0)
        smape_str = format_legend_number(r['metrics']['sMAPE'], 1)
        label = f"{r['name']} (sMAPE={smape_str}%)"
        ax.plot(t_grid, I_pred, color=r['color'], linestyle=r['style'], linewidth=3, label=label)

    # === ВРЕЗКА (ZOOM) ===
    # ax_ins = zoomed_inset_axes(ax, zoom=2, loc=1, borderpad=3)
    # for spine in ax_ins.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linewidth(2)
    #
    # ax_ins.scatter(t_grid, I_real, color='black', alpha=0.7, s=60)
    # for r in results:
    #     I_pred = run_simulation(r['params'], t_grid, N, y0)
    #     ax_ins.plot(t_grid, I_pred, color=r['color'], linestyle=r['style'], linewidth=3)
    #
    # # Координаты врезки (подбираем под масштаб мая 2020)
    # x1, x2 = 15, 25
    # y1, y2 = 1500, 2300
    # ax_ins.set_xlim(x1, x2)
    # ax_ins.set_ylim(y1, y2)
    # ax_ins.set_xticks([])
    # ax_ins.set_yticks([])

    # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='0.5', linestyle='-', linewidth=1)
    # ax.add_patch(rect)
    # conn1 = ConnectionPatch(xyA=(x2, y2), coordsA=ax.transData, xyB=(0, 1), coordsB=ax_ins.transAxes, axesA=ax, axesB=ax_ins, color="0.5", linewidth=1)
    # ax.add_artist(conn1)
    # conn2 = ConnectionPatch(xyA=(x2, y1), coordsA=ax.transData, xyB=(0, 0), coordsB=ax_ins.transAxes, axesA=ax, axesB=ax_ins, color="0.5", linewidth=1)
    # ax.add_artist(conn2)

    # === ПОДПИСИ ДАТ НА ОСИ X ===
    tick_indices = [0, 4, 9, 14, 19, 24, 29]
    tick_labels = []
    for idx in tick_indices:
        if idx < len(dates):
            day = pd.to_datetime(dates[idx]).day
            tick_labels.append(f"{day} мая")
        else:
            tick_labels.append("")
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('Дата', fontsize=22)
    ax.set_ylabel('Число инфицированных', fontsize=22)
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=22)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN
# ==========================================
def main():
    t, I_real, N, Y0, dates = load_real_data()
    opt_args = (t, N, Y0, I_real)

    results = []
    print("Запуск Нелдер-Мид...")
    results.append(run_nelder_mead(opt_args, t, N, Y0, I_real))
    print("Запуск Дифференциальной эволюции...")
    results.append(run_differential_evolution(opt_args, t, N, Y0, I_real))
    print("Запуск Optuna...")
    results.append(run_optuna(opt_args, t, N, Y0, I_real))
    print("Запуск CMA-ES...")
    results.append(run_cma_es(opt_args, t, N, Y0, I_real))

    print_results_table(results)
    plot_comparison(t, I_real, N, Y0, results, dates)

if __name__ == "__main__":
    main()