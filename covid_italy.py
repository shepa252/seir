import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. КОНФИГУРАЦИЯ И СТИЛЬ
# ==========================================
FILE_NAME = 'time_series_19-covid-Confirmed_archived_0325.csv'
COUNTRY_REGION = 'Italy'
POPULATION_N = 60000000.0
SIMULATION_DURATION = 80
MIN_CASES_THRESHOLD = 100

# Настройки оптимизации
INITIAL_GUESS = [5.0, 5.0]  # Плохое начальное приближение (для NM)
BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
OPTUNA_TRIALS = 500

# Настройки графиков
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'lines.linewidth': 3
})

# Функция для форматирования чисел в легенде с запятой
def format_legend_number(value: float, digits: int = 1) -> str:
    """Форматирует десятичное число с запятой вместо точки для легенды."""
    return f"{value:.{digits}f}".replace('.', ',')


# ==========================================
# 2. МАТЕМАТИКА (SIR + МЕТРИКИ)
# ==========================================
def sir_derivatives(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def run_simulation(params, t_grid, N, y0):
    beta, gamma = params
    beta = max(0, beta)
    gamma = max(0, gamma)
    ret = odeint(sir_derivatives, y0, t_grid, args=(N, beta, gamma))
    return ret.T[1]


def loss_function(params, t_grid, N, y0, data_real):
    I_pred = run_simulation(params, t_grid, N, y0)
    return mean_squared_error(data_real, I_pred)


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        smape_vals = numerator / denominator
        smape_vals[denominator == 0] = 0.0
    smape = 100 * np.mean(smape_vals)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "sMAPE": smape
    }


# ==========================================
# 3. ЗАГРУЗКА ДАННЫХ
# ==========================================
def load_real_data():
    print(f"--- Загрузка данных для: {COUNTRY_REGION} ---")
    data = pd.read_csv(FILE_NAME)
    country_data = data[data['Country/Region'] == COUNTRY_REGION]
    cases = country_data.iloc[:, 4:].values.flatten()

    try:
        start_index = np.where(cases > MIN_CASES_THRESHOLD)[0][0]
    except IndexError:
        start_index = 0

    I_real = cases[start_index: start_index + SIMULATION_DURATION].astype(float)
    days = len(I_real)
    t_grid = np.linspace(0, days - 1, days)

    I0 = I_real[0]
    y0 = (POPULATION_N - I0, I0, 0.0)

    print(f"Период: {days} дней. N={POPULATION_N / 1e6:.1f} млн.")
    return t_grid, I_real, POPULATION_N, y0


# ==========================================
# 4. МЕТОДЫ ОПТИМИЗАЦИИ
# ==========================================
def run_method(method_name, args, t_grid, N, y0, I_real):
    start_time = time.time()
    iters = 0

    if method_name == "Нелдер-Мид":
        res = minimize(loss_function, INITIAL_GUESS, args=args, method='Nelder-Mead', tol=1e-5)
        beta, gamma = res.x
        iters = res.nit

    elif method_name == "Дифференциальная эволюция":
        res = differential_evolution(loss_function, BOUNDS, args=args, strategy='best1bin', tol=1e-5)
        beta, gamma = res.x
        iters = res.nit

    elif method_name == "Байесовская оптимизация":
        def objective(trial):
            b = trial.suggest_float("beta", 0.0, 1.0)
            g = trial.suggest_float("gamma", 0.0, 1.0)
            return loss_function([b, g], *args)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        beta = study.best_trial.params['beta']
        gamma = study.best_trial.params['gamma']
        iters = study.best_trial.number

    duration = time.time() - start_time

    I_pred = run_simulation([beta, gamma], t_grid, N, y0)
    metrics = calculate_metrics(I_real, I_pred)

    return {
        "name": method_name,
        "beta": beta,
        "gamma": gamma,
        "time": duration,
        "iters": iters,
        "metrics": metrics
    }


# ==========================================
# 5. ВЫВОД И ВИЗУАЛИЗАЦИЯ
# ==========================================
def print_table(results):
    print("\n" + "=" * 125)
    # НОВЫЙ ПОРЯДОК: Beta | Gamma | Time | Iters | RMSE | MAE | sMAPE
    print(
        f"{'Метод':<20} | {'Beta':<6} | {'Gamma':<6} | {'Время(с)':<8} | {'Итераций':<8} | {'RMSE':<10} | {'MAE':<10} | {'sMAPE(%)':<10}")
    print("-" * 125)

    for r in results:
        m = r['metrics']
        # Вывод данных в новом порядке
        print(
            f"{r['name']:<20} | {r['beta']:<6.3f} | {r['gamma']:<6.3f} | {r['time']:<8.4f} | {r['iters']:<8} | {m['RMSE']:<10.0f} | {m['MAE']:<10.0f} | {m['sMAPE']:<10.2f}")
    print("=" * 125 + "\n")


def plot_final(t_grid, I_real, N, y0, results):
    plt.figure(figsize=(14, 8))

    plt.scatter(t_grid, I_real, color='black', alpha=0.7, s=60, label='COVID-19 (Факт)')

    colors = {"Нелдер-Мид": "red", "Дифференциальная эволюция": "blue", "Байесовская оптимизация": "green"}
    styles = {"Нелдер-Мид": "-", "Дифференциальная эволюция": "-.", "Байесовская оптимизация": "--"}

    for r in results:
        name = r['name']
        I_pred = run_simulation([r['beta'], r['gamma']], t_grid, N, y0)

        # Форматируем sMAPE с запятой для легенды
        smape_formatted = format_legend_number(r['metrics']['sMAPE'], 1)
        label_txt = f"{name}\nsMAPE={smape_formatted}%"

        plt.plot(t_grid, I_pred,
                 color=colors.get(name, 'gray'),
                 linestyle=styles.get(name, '-'),
                 label=label_txt,
                 alpha=0.8)

    # plt.title(f'Калибровка SIR на данных {COUNTRY_REGION} (Сравнение методов)', pad=20, fontweight='bold')
    plt.xlabel('Дни с начала моделирования')
    plt.ylabel('Число инфицированных')
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==========================================
# MAIN
# ==========================================
def main():
    t_grid, I_real, N, y0 = load_real_data()
    args = (t_grid, N, y0, I_real)

    results = []

    print("Запуск Nelder-Mead...")
    results.append(run_method("Нелдер-Мид", args, t_grid, N, y0, I_real))

    print("Запуск Differential Evolution...")
    results.append(run_method("Дифференциальная эволюция", args, t_grid, N, y0, I_real))

    print("Запуск Optuna...")
    results.append(run_method("Байесовская оптимизация", args, t_grid, N, y0, I_real))

    print_table(results)
    plot_final(t_grid, I_real, N, y0, results)


if __name__ == "__main__":
    main()