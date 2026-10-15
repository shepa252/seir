"""
Реальные данные (Волгоградская область, май 2020): идентифицируемость параметров
и честная проверка GRNN.

1) Мультистарт: N_STARTS случайных стартов -> локальная оптимизация (L-BFGS-B с границами).
   Рисунок: найденные (γ, β), цвет = MSE, размер/метка = R0. Если много разных (β, γ)
   дают почти одинаковый MSE, то параметры плохо идентифицируемы по одной кривой I(t).
2) GRNN на отложенных последних днях и наивный прогноз I(t)=I(t-1) для сравнения.

Запуск:  python real_identifiability.py
Файл volgograd05.csv и grnn.py должны лежать рядом.
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import grnn as g  # load_real_data, run_simulation, loss_function, GRNN, create_lagged_data, calculate_metrics

warnings.filterwarnings("ignore")
N_STARTS = 300
BOUNDS = [(0.01, 1.0)] * 3
HOLDOUT = 7


def multistart(args, n=N_STARTS, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x0 = rng.uniform(0.01, 1.0, 3)
        r = minimize(g.loss_function, x0, args=args, method="L-BFGS-B", bounds=BOUNDS)
        out.append((*r.x, r.fun))
    return np.array(out)  # beta, alpha, gamma, mse


def grnn_holdout(I_real, n_lags=3, sigma=0.3, h=HOLDOUT):
    X, y = g.create_lagged_data(I_real, n_lags)
    Xtr, ytr, Xte, yte = X[:-h], y[:-h], X[-h:], y[-h:]
    m = g.GRNN(sigma=sigma)
    m.fit(Xtr, ytr)
    pred_in = m.predict(Xtr)
    pred_out = m.predict(Xte)
    naive_out = Xte[:, -1]  # I(t) ~ I(t-1)
    return {
        "in-sample": g.calculate_metrics(ytr, pred_in),
        "holdout GRNN": g.calculate_metrics(yte, pred_out),
        "holdout naive I(t-1)": g.calculate_metrics(yte, naive_out),
    }


def main():
    t, I_real, N, Y0, dates = g.load_real_data()
    args = (t, N, Y0, I_real)

    res = multistart(args)
    best = res[:, 3].min()
    near = res[res[:, 3] <= best * 1.05]  # в пределах 5% от лучшего MSE
    r0 = res[:, 0] / res[:, 2]
    r0_near = near[:, 0] / near[:, 2]
    print(f"Лучший MSE: {best:.1f}; стартов в пределах 5% от лучшего: {len(near)} из {len(res)}")
    print(f"R0 среди них: min={r0_near.min():.1f}, медиана={np.median(r0_near):.1f}, max={r0_near.max():.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sc = ax[0].scatter(res[:, 2], res[:, 0], c=np.log10(res[:, 3]), s=22, cmap="viridis_r")
    ax[0].set_xlabel("γ"); ax[0].set_ylabel("β")
    ax[0].set_title("Найденные (γ, β) из случайных стартов")
    fig.colorbar(sc, ax=ax[0], label="lg(MSE)")
    ax[0].grid(True, ls="--", alpha=0.4)
    ax[1].scatter(r0, res[:, 3], s=22)
    ax[1].set_yscale("log"); ax[1].set_xscale("log")
    ax[1].set_xlabel("R₀ = β/γ"); ax[1].set_ylabel("MSE")
    ax[1].set_title("R₀ против качества подгонки")
    ax[1].grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig("fig_real_identifiability.png", dpi=300)
    print("Сохранено: fig_real_identifiability.png")

    print("\nGRNN (lags=3, sigma=0.3):")
    for k, v in grnn_holdout(I_real).items():
        print(f"  {k:<22} RMSE={v['RMSE']:.1f}  MAE={v['MAE']:.1f}  sMAPE={v['sMAPE']:.1f}%")


if __name__ == "__main__":
    main()
