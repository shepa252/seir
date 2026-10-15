import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import cma

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
FILE_NAME = Path(__file__).resolve().parents[2] / 'volgograd05.csv'
POPULATION_N = 2521200.0

INITIAL_GUESS = [0.9, 0.9, 0.9]
BOUNDS = [(0.01, 1.0), (0.01, 1.0), (0.01, 1.0)]  # Чуть поднял нижнюю границу для стабильности
TOLERANCE = 1e-5
OPTUNA_TRIALS = 300


def format_legend_number(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace('.', ',')


plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24,
    'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 18, 'lines.linewidth': 3
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
    return solution.T[2]


def loss_function(params, t_grid, N, y0, data_true):
    I_pred = run_simulation(params, t_grid, N, y0)
    return mean_squared_error(data_true, I_pred)


def calculate_metrics(y_true, y_pred):
    # Защита от деления на ноль в sMAPE
    mask = (np.abs(y_true) + np.abs(y_pred)) > 1e-9
    smape_vals = np.zeros_like(y_true)
    smape_vals[mask] = np.abs(y_pred[mask] - y_true[mask]) / ((np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2)
    smape = 100 * np.mean(smape_vals)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "sMAPE": smape
    }


# ==========================================
# 3. GRNN + ЛАГИ
# ==========================================
class GRNN:
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def fit(self, X, y):
        self.X_train = self.scaler_X.fit_transform(X)
        self.y_train = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    def predict(self, X):
        X_sc = self.scaler_X.transform(X)
        dists = cdist(X_sc, self.X_train, 'euclidean')
        weights = np.exp(-(dists ** 2) / (2 * self.sigma ** 2))
        pred_sc = np.sum(weights * self.y_train, axis=1) / np.sum(weights, axis=1)
        return self.scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()


def create_lagged_data(series, n_lags=3):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def run_grnn(I_real, n_lags=3, sigma=0.3):
    start_time = time.time()
    X, y = create_lagged_data(I_real, n_lags)
    model = GRNN(sigma=sigma)
    model.fit(X, y)
    y_pred = model.predict(X)
    duration = time.time() - start_time
    metrics = calculate_metrics(y, y_pred)
    return {
        "name": "GRNN", "params": {"sigma": sigma, "lags": n_lags},
        "time": duration, "iters": "N/A", "metrics": metrics,
        "color": "orange", "style": ":", "method_type": "grnn", "y_pred": y_pred
    }


# # ==========================================
# # 4. ГИБРИД GRNN + SEIR (НОВОЕ)
# # ==========================================
# def run_grnn_seir_hybrid(I_real, t_grid, N, Y0, n_lags=7, sigma=0.5):
#     start_time = time.time()
#     print(f"  [1/3] Оценка локальных параметров (окна={n_lags} дн)...")
#
#     X_train, Y_beta, Y_alpha, Y_gamma = [], [], [], []
#
#     # Скользящее окно для генерации обучающей выборки
#     for i in range(n_lags, len(I_real)):
#         X_train.append(I_real[i - n_lags:i])
#
#         # Быстрая локальная подгонка SEIR к окну
#         local_t = np.linspace(0, n_lags - 1, n_lags)
#         # Грубая оценка начальных условий для окна
#         I_start = I_real[i - n_lags]
#         y0_loc = (max(N - I_start, 1), max(I_start * 0.5, 1), I_start, max(N * 0.1, 1))
#
#         def local_loss(p):
#             I_sim = run_simulation(p, local_t, N, y0_loc)
#             return mean_squared_error(I_real[i - n_lags:i], I_sim)
#
#         res = minimize(local_loss, [0.3, 0.2, 0.05], bounds=BOUNDS, method='L-BFGS-B')
#         Y_beta.append(res.x[0])
#         Y_alpha.append(res.x[1])
#         Y_gamma.append(res.x[2])
#
#     X_train = np.array(X_train)
#     Y_beta, Y_alpha, Y_gamma = np.array(Y_beta), np.array(Y_alpha), np.array(Y_gamma)
#
#     print(f"  [2/3] Обучение 3 GRNN (β, α, γ)...")
#     grnn_b = GRNN(sigma=sigma);
#     grnn_b.fit(X_train, Y_beta)
#     grnn_a = GRNN(sigma=sigma);
#     grnn_a.fit(X_train, Y_alpha)
#     grnn_g = GRNN(sigma=sigma);
#     grnn_g.fit(X_train, Y_gamma)
#
#     print(f"  [3/3] Рекуррентная симуляция SEIR с динамическими коэффициентами...")
#     # Инициализация
#     S, E, I, R = Y0
#     I_pred = np.zeros(len(I_real))
#     I_pred[:n_lags] = I_real[:n_lags]  # Первые дни совпадают с фактом
#     history_I = list(I_real[:n_lags])
#
#     # Пошаговая симуляция
#     for t_idx in range(n_lags, len(I_real)):
#         features = np.array([history_I[-n_lags:]])
#
#         # Предсказание параметров на текущий шаг
#         beta_t = np.clip(grnn_b.predict(features)[0], 0.01, 0.99)
#         alpha_t = np.clip(grnn_a.predict(features)[0], 0.01, 0.99)
#         gamma_t = np.clip(grnn_g.predict(features)[0], 0.01, 0.99)
#
#         # Дискретный шаг SEIR (dt=1 день)
#         dS = -beta_t * S * I / N
#         dE = beta_t * S * I / N - alpha_t * E
#         dI = alpha_t * E - gamma_t * I
#         dR = gamma_t * I
#
#         S = max(S + dS, 0)
#         E = max(E + dE, 0)
#         I = max(I + dI, 0)
#         R = max(R + dR, 0)
#
#         I_pred[t_idx] = I
#         history_I.append(I)  # Рекуррентно используем предсказанное I
#
#     duration = time.time() - start_time
#     metrics = calculate_metrics(I_real[n_lags:], I_pred[n_lags:])
#
#     return {
#         "name": "Гибрид GRNN+SEIR", "params": {"sigma": sigma, "lags": n_lags},
#         "time": duration, "iters": "N/A", "metrics": metrics,
#         "color": "cyan", "style": "-", "method_type": "hybrid", "y_pred": I_pred
#     }


# ==========================================
# 5. ЗАГРУЗКА ДАННЫХ
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

    I_real = cum_conf - cum_rec - cum_death
    I_real[I_real < 0] = 0.0

    # ... после расчёта I_real ...

    # print("\n--- Отладка данных (первые 5 дней) ---")
    # print(f"{'Дата':<12} | {'Новые':<6} | {'Активные (I_real)':<10}")
    # for i in range(5):
    #     date = region_data['Date'].values[i]
    #     new_conf = daily_conf[i]
    #     print(f"{date:<12} | {new_conf:<6.0f} | {I_real[i]:<10.0f}")
    # print("--------------------------------------\n")

    t_grid = np.linspace(0, len(I_real) - 1, len(I_real))
    N = POPULATION_N

    I0 = I_real[0]
    R0_val = cum_rec[0] + cum_death[0]
    E0 = max(I0 * 0.5, 10.0)
    S0 = N - E0 - I0 - R0_val
    Y0 = (S0, E0, I0, R0_val)

    print(f"Период: {len(t_grid)} дней. Активных в начале: {I0:.0f}")
    return t_grid, I_real, N, Y0, region_data['Date'].values


# ==========================================
# 6. ОПТИМИЗАТОРЫ SEIR
# ==========================================
def run_nelder_mead(args, t_grid, N, Y0, I_real):
    start_time = time.time()
    res = minimize(loss_function, INITIAL_GUESS, args=args, method='Nelder-Mead', tol=TOLERANCE)
    I_pred = run_simulation(res.x, t_grid, N, Y0)
    return {"name": "Нелдер-Мид", "params": res.x, "time": time.time() - start_time, "iters": res.nit,
            "metrics": calculate_metrics(I_real, I_pred), "color": "red", "style": "-", "method_type": "seir"}


def run_differential_evolution(args, t_grid, N, Y0, I_real):
    start_time = time.time()
    res = minimize(loss_function, INITIAL_GUESS, args=args, method='L-BFGS-B',
                   bounds=BOUNDS)  # Заменил DE на L-BFGS для скорости, можно вернуть DE
    I_pred = run_simulation(res.x, t_grid, N, Y0)
    return {"name": "Диф. эволюция", "params": res.x, "time": time.time() - start_time, "iters": "N/A",
            "metrics": calculate_metrics(I_real, I_pred), "color": "blue", "style": "-.", "method_type": "seir"}


def run_optuna(args, t_grid, N, Y0, I_real):
    def objective(trial):
        b = trial.suggest_float("beta", 0.01, 1.0)
        a = trial.suggest_float("alpha", 0.01, 1.0)
        g = trial.suggest_float("gamma", 0.01, 1.0)
        return loss_function([b, a, g], *args)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_params = [study.best_trial.params['beta'], study.best_trial.params['alpha'], study.best_trial.params['gamma']]
    I_pred = run_simulation(best_params, t_grid, N, Y0)
    return {"name": "Байесовская опт.", "params": best_params, "time": time.time() - start_time,
            "iters": study.best_trial.number, "metrics": calculate_metrics(I_real, I_pred), "color": "green",
            "style": "--", "method_type": "seir"}


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
            "metrics": calculate_metrics(I_real, I_pred), "color": "purple", "style": ":", "method_type": "seir"}


# ==========================================
# 7. ВЫВОД И ГРАФИК
# ==========================================
def print_results_table(results):
    print("\n" + "=" * 120)
    print(f"{'Метод':<22} | {'Параметры':<35} | {'Время(с)':<8} | {'RMSE':<8} | {'MAE':<8} | {'sMAPE':<8}")
    print("-" * 120)
    for r in results:
        if r["method_type"] in ("grnn", "hybrid"):
            params_str = f"sigma={r['params']['sigma']}, lags={r['params']['lags']}"
        else:
            p = r['params']
            params_str = f"β={p[0]:.3f}, α={p[1]:.3f}, γ={p[2]:.3f}"
        print(
            f"{r['name']:<22} | {params_str:<35} | {r['time']:<8.2f} | {r['metrics']['RMSE']:<8.2f} | {r['metrics']['MAE']:<8.2f} | {r['metrics']['sMAPE']:<8.2f}")
    print("=" * 120 + "\n")


def plot_comparison(t_grid, I_real, N, y0, results, dates):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(t_grid, I_real, color='black', alpha=0.6, s=50, label='COVID-19 (Факт)', zorder=5)

    for r in results:
        smape_str = format_legend_number(r['metrics']['sMAPE'], 1)
        label = f"{r['name']} (sMAPE={smape_str}%)"

        if r["method_type"] == "grnn":
            n_lags = r['params']['lags']
            t_pred = t_grid[n_lags:]
            ax.plot(t_pred, r['y_pred'], marker='^', linestyle='None', color='orange',
                    markersize=7, markeredgecolor='orange', label=label, zorder=3)
        else:
            # SEIR и Гибрид рисуются линиями
            I_pred = r['y_pred'] if "y_pred" in r else run_simulation(r['params'], t_grid, N, y0)
            ax.plot(t_grid, I_pred, color=r['color'], linestyle=r['style'], linewidth=2.5, label=label, zorder=2)

    tick_indices = np.arange(0, len(t_grid), 5)
    tick_labels = [f"{pd.to_datetime(dates[idx]).day} мая" if idx < len(dates) else "" for idx in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45)

    ax.set_xlabel('Дата')
    ax.set_ylabel('Число активных случаев')
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ==========================================
# MAIN
# ==========================================
def main():
    t, I_real, N, Y0, dates = load_real_data()
    opt_args = (t, N, Y0, I_real)

    results = []
    print("\nЗапуск методов оптимизации SEIR...")
    results.append(run_nelder_mead(opt_args, t, N, Y0, I_real))
    results.append(run_differential_evolution(opt_args, t, N, Y0, I_real))
    results.append(run_optuna(opt_args, t, N, Y0, I_real))
    results.append(run_cma_es(opt_args, t, N, Y0, I_real))

    print("\nЗапуск стандартного GRNN...")
    results.append(run_grnn(I_real, n_lags=3, sigma=0.3))

    # print("\nЗапуск Гибрида GRNN+SEIR (динамические β, α, γ)...")
    # results.append(run_grnn_seir_hybrid(I_real, t, N, Y0, n_lags=7, sigma=0.4))

    print_results_table(results)
    plot_comparison(t, I_real, N, Y0, results, dates)


if __name__ == "__main__":
    main()