"""
Параметрическая идентификация динамических моделей (версия для конференции).

Модели:  SEIR, Лотка-Вольтерра, Лоренц, Ван дер Поль
Методы:  Нелдер-Мид, Дифференциальная эволюция, Байесовская оптимизация (Optuna), CMA-ES
GUI:     выбор двух моделей -> сравнение методов (траектории, ошибки параметров, матрица ошибок,
         таблица показателей, время работы, влияние шума)

Единая постановка для всех моделей:
    θ* = argmin J(θ),  J(θ) = среднее по времени и компонентам от ((y_расч − y_набл) / s)²,  θ ∈ [θ_min, θ_max]

pip install numpy scipy matplotlib optuna cma
"""
import queue
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Tuple
import warnings

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
import optuna
import cma
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================================
# 1. НАСТРОЙКИ
# ==========================================================
NOISE_LEVELS = [1, 2, 5, 10, 20]   # % от СКО сигнала
SUCCESS_THR = 10.0                 # успех, если средняя отн. ошибка параметров < 10 %
METHODS = ["Нелдер-Мид", "Дифф. эволюция", "Байесовская опт.", "CMA-ES"]
M_COLORS = {"Нелдер-Мид": "tab:red", "Дифф. эволюция": "tab:blue",
            "Байесовская опт.": "tab:green", "CMA-ES": "tab:purple"}


# ==========================================================
# 2. МОДЕЛИ
# ==========================================================
@dataclass
class Model:
    name: str
    pnames: Tuple[str, ...]
    true: Tuple[float, ...]
    bounds: Tuple[Tuple[float, float], ...]
    y0: Tuple[float, ...]
    T: float
    npts: int
    rhs: Callable
    obs: Tuple[int, ...]       # наблюдаемые компоненты
    labels: Tuple[str, ...]
    positive: bool = False


N_POP = 10000.0


def seir_rhs(y, t, beta, alpha, gamma):
    S, E, I, R = y
    inf = beta * S * I / N_POP
    return [-inf, inf - alpha * E, alpha * E - gamma * I, gamma * I]


def lv_rhs(y, t, a, b, c, d):
    x, z = y
    return [a * x - b * x * z, -c * z + d * x * z]


def lorenz_rhs(y, t, sigma, rho, beta):
    x, y_, z = y
    return [sigma * (y_ - x), x * (rho - z) - y_, x * y_ - beta * z]


def vdp_rhs(y, t, mu, omega):
    x, v = y
    return [v, mu * (1 - x * x) * v - omega ** 2 * x]


MODELS = {
    "SEIR": Model("SEIR", ("β", "α", "γ"), (0.65, 0.3, 0.1), ((0.01, 1.0),) * 3,
                  (N_POP - 2, 1, 1, 0), 100, 100, seir_rhs, (2,), ("I(t)",), True),
    "Лотка-Вольтерра": Model("Лотка-Вольтерра", ("a", "b", "c", "d"), (1.0, 0.1, 1.5, 0.075),
                             ((0.1, 3), (0.01, 0.5), (0.1, 3), (0.01, 0.5)),
                             (10, 5), 15, 150, lv_rhs, (0, 1), ("жертвы x", "хищники y")),
    "Лоренц": Model("Лоренц", ("σ", "ρ", "β"), (10.0, 28.0, 8 / 3),
                    ((5, 15), (15, 40), (1, 5)), (1, 1, 1), 2, 200, lorenz_rhs,
                    (0, 1, 2), ("x", "y", "z")),
    "Ван дер Поль": Model("Ван дер Поль", ("μ", "ω"), (2.0, 1.0), ((0.1, 5), (0.3, 2)),
                          (2, 0), 20, 200, vdp_rhs, (0, 1), ("x", "v=dx/dt")),
}


def simulate(m: Model, p) -> np.ndarray:
    t = np.linspace(0, m.T, m.npts)
    try:
        return odeint(m.rhs, m.y0, t, args=tuple(p), rtol=1e-6, atol=1e-8)[:, list(m.obs)]
    except Exception:
        return np.full((m.npts, len(m.obs)), np.nan)


# ==========================================================
# 3. ОПТИМИЗАТОРЫ (работают в единичном кубе [0,1]^d)
# ==========================================================
def _seed(rng):
    return int(rng.integers(1, 2 ** 31 - 1))


def opt_nm(f, dim, rng):
    x0 = rng.uniform(0.1, 0.9, dim)
    return minimize(f, x0, method="Nelder-Mead", bounds=[(0, 1)] * dim,
                    options=dict(maxiter=400 * dim, xatol=1e-6, fatol=1e-10)).x


def opt_de(f, dim, rng):
    return differential_evolution(f, [(0, 1)] * dim, maxiter=80, popsize=10, tol=1e-8,
                                  polish=False, seed=_seed(rng)).x


def opt_optuna(f, dim, rng):
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=_seed(rng)))
    study.optimize(lambda tr: f(np.array([tr.suggest_float(f"u{i}", 0, 1) for i in range(dim)])),
                   n_trials=150)
    return np.array([study.best_params[f"u{i}"] for i in range(dim)])


def opt_cma(f, dim, rng):
    x0 = rng.uniform(0.1, 0.9, dim)
    es = cma.CMAEvolutionStrategy(x0, 0.25, {"bounds": [0, 1], "verbose": -9, "tolfun": 1e-12,
                                             "maxiter": 150, "seed": _seed(rng)})
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [f(x) for x in xs])
    return np.array(es.result.xbest)


OPTIMIZERS = dict(zip(METHODS, [opt_nm, opt_de, opt_optuna, opt_cma]))


# ==========================================================
# 4. ЭКСПЕРИМЕНТ
# ==========================================================
def identify(m, data, scale, method, rng):
    lo = np.array([b[0] for b in m.bounds]); hi = np.array([b[1] for b in m.bounds])

    def f(u):
        sim = simulate(m, lo + np.clip(u, 0, 1) * (hi - lo))
        v = np.mean(((sim - data) / scale) ** 2)
        return v if np.isfinite(v) else 1e6

    t0 = time.perf_counter()
    u = OPTIMIZERS[method](f, len(lo), rng)
    return lo + np.clip(u, 0, 1) * (hi - lo), time.perf_counter() - t0


def run_experiment(model_names, n_runs, base_noise, progress=lambda d, t: None):
    records, demo = [], {}
    total = len(model_names) * len(NOISE_LEVELS) * n_runs * len(METHODS)
    done = 0
    for mn in model_names:
        m = MODELS[mn]
        clean = simulate(m, m.true)
        scale = clean.std(axis=0) + 1e-12
        t = np.linspace(0, m.T, m.npts)
        for lvl in NOISE_LEVELS:
            for run in range(n_runs):
                rd = np.random.default_rng(zlib.crc32(f"{mn}-{lvl}-{run}".encode()))
                noisy = clean + lvl / 100 * scale * rd.normal(size=clean.shape)
                if m.positive:
                    noisy = np.maximum(noisy, 0)
                for method in METHODS:
                    rng = np.random.default_rng(zlib.crc32(f"{mn}-{lvl}-{run}-{method}".encode()))
                    p, dt = identify(m, noisy, scale, method, rng)
                    rel = np.abs(p - m.true) / np.abs(m.true) * 100
                    sim = simulate(m, p)
                    rmse = float(np.sqrt(np.nanmean(((sim - clean) / scale) ** 2)))
                    records.append(dict(model=mn, method=method, noise=lvl, run=run, rel=rel,
                                        mean_rel=float(rel.mean()),
                                        rmse=rmse if np.isfinite(rmse) else 1e3,
                                        time=dt, ok=bool(rel.mean() < SUCCESS_THR)))
                    if lvl == base_noise and run == 0:
                        demo.setdefault(mn, dict(t=t, clean=clean, noisy=noisy, fits={}))
                        demo[mn]["fits"][method] = sim
                    done += 1
                    progress(done, total)
    return records, demo


# ==========================================================
# 5. GUI
# ==========================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Параметрическая идентификация динамических моделей")
        self.geometry("1350x850")
        self.q = queue.Queue()
        self.records, self.demo, self.models_used, self.base = [], {}, [], 5

        top = ttk.Frame(self, padding=6); top.pack(fill="x")
        names = list(MODELS)
        self.cb_a = ttk.Combobox(top, values=names, state="readonly", width=18); self.cb_a.set("SEIR")
        self.cb_b = ttk.Combobox(top, values=names, state="readonly", width=18); self.cb_b.set("Лотка-Вольтерра")
        self.cb_noise = ttk.Combobox(top, values=NOISE_LEVELS, state="readonly", width=5); self.cb_noise.set(5)
        self.sp_runs = ttk.Spinbox(top, from_=1, to=30, width=5); self.sp_runs.set(3)
        for i, (txt, w) in enumerate([("Модель A:", self.cb_a), ("Модель B:", self.cb_b),
                                      ("Базовый шум, %:", self.cb_noise), ("Повторов:", self.sp_runs)]):
            ttk.Label(top, text=txt).grid(row=0, column=2 * i, padx=(8, 2)); w.grid(row=0, column=2 * i + 1)
        self.btn = ttk.Button(top, text="▶ Запустить", command=self.start); self.btn.grid(row=0, column=8, padx=10)
        self.pb = ttk.Progressbar(top, length=200); self.pb.grid(row=0, column=9, padx=10)
        self.status = ttk.Label(top, text="Готово к запуску"); self.status.grid(row=0, column=10)

        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True)
        self.figs = {}
        for key, title in [("traj", "Сравнение моделей"), ("err", "Ошибка параметров"),
                           ("mat", "Матрица ошибок"), ("tab", "Таблица показателей"),
                           ("time", "Время работы"), ("noise", "Влияние шума")]:
            fr = ttk.Frame(self.nb); self.nb.add(fr, text=title)
            if key == "tab":
                cols = ("Модель", "Метод", "Ср. отн. ошибка парам., %", "RMSE траектории (норм.)",
                        "Деградация, %/% шума", "Успешность, %", "Время, с")
                self.tree = ttk.Treeview(fr, columns=cols, show="headings")
                for c in cols:
                    self.tree.heading(c, text=c); self.tree.column(c, width=170, anchor="center")
                self.tree.tag_configure("best", background="#d9f2d9")
                self.tree.pack(fill="both", expand=True)
                self.note = ttk.Label(fr, justify="left"); self.note.pack(anchor="w", padx=6, pady=4)
            else:
                fig = Figure(figsize=(12, 7), dpi=100)
                cv = FigureCanvasTkAgg(fig, master=fr)
                NavigationToolbar2Tk(cv, fr)
                cv.get_tk_widget().pack(fill="both", expand=True)
                self.figs[key] = (fig, cv)
        self.after(100, self.poll)

    def start(self):
        self.models_used = list(dict.fromkeys([self.cb_a.get(), self.cb_b.get()]))
        self.base = int(self.cb_noise.get())
        n_runs = int(self.sp_runs.get())
        self.btn.state(["disabled"]); self.pb["value"] = 0; self.status["text"] = "Вычисления..."

        def work():
            try:
                rec, demo = run_experiment(self.models_used, n_runs, self.base,
                                           lambda d, t: self.q.put(("p", d, t)))
                self.q.put(("done", rec, demo))
            except Exception as e:
                self.q.put(("err", repr(e)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "p":
                    self.pb["value"] = 100 * msg[1] / msg[2]; self.status["text"] = f"{msg[1]}/{msg[2]}"
                elif msg[0] == "done":
                    self.records, self.demo = msg[1], msg[2]
                    self.btn.state(["!disabled"]); self.status["text"] = "Готово"; self.redraw()
                else:
                    self.btn.state(["!disabled"]); messagebox.showerror("Ошибка", msg[1])
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def sel(self, model, method, noise=None):
        return [r for r in self.records if r["model"] == model and r["method"] == method
                and (noise is None or r["noise"] == noise)]

    def redraw(self):
        self.draw_traj(); self.draw_err(); self.draw_mat(); self.draw_table(); self.draw_time(); self.draw_noise()

    def _axes(self, key, n):
        fig, cv = self.figs[key]; fig.clear()
        return fig, cv, fig.subplots(1, n, squeeze=False)[0]

    def draw_traj(self):
        fig, cv = self.figs["traj"]; fig.clear()
        nm = len(self.models_used); ncol = max(len(MODELS[m].obs) for m in self.models_used)
        for i, mn in enumerate(self.models_used):
            m, d = MODELS[mn], self.demo[mn]
            for j in range(len(m.obs)):
                ax = fig.add_subplot(nm, ncol, i * ncol + j + 1)
                ax.scatter(d["t"], d["noisy"][:, j], s=8, c="gray", alpha=.5, label="наблюдения")
                ax.plot(d["t"], d["clean"][:, j], "k:", lw=2.5, label="истина")
                for me in METHODS:
                    ax.plot(d["t"], d["fits"][me][:, j], color=M_COLORS[me], lw=1.6, label=me)
                ax.set_title(f"{mn}: {m.labels[j]}", fontsize=10); ax.grid(alpha=.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(f"Восстановленные траектории (шум {self.base} %)")
        fig.tight_layout(); cv.draw()

    def draw_err(self):
        fig, cv, axs = self._axes("err", len(self.models_used))
        for ax, mn in zip(axs, self.models_used):
            m = MODELS[mn]; k = len(m.pnames); w = 0.8 / k
            for pi, pn in enumerate(m.pnames):
                vals = [np.mean([r["rel"][pi] for r in self.sel(mn, me, self.base)]) for me in METHODS]
                ax.bar(np.arange(len(METHODS)) + pi * w - 0.4 + w / 2, np.maximum(vals, 1e-4), w, label=pn)
            ax.set_yscale("log"); ax.set_xticks(range(len(METHODS)))
            ax.set_xticklabels(METHODS, rotation=15, fontsize=8)
            ax.set_ylabel("Относительная ошибка, % (log)"); ax.axhline(SUCCESS_THR, color="k", ls="--", lw=.8)
            ax.set_title(f"{mn} (шум {self.base} %)"); ax.legend(); ax.grid(alpha=.3, which="both")
        fig.tight_layout(); cv.draw()

    def draw_mat(self):
        fig, cv, axs = self._axes("mat", len(self.models_used))
        for ax, mn in zip(axs, self.models_used):
            m = MODELS[mn]
            M = np.array([np.mean([r["rel"] for r in self.sel(mn, me, self.base)], axis=0) for me in METHODS])
            im = ax.imshow(np.log10(np.maximum(M, 1e-4)), cmap="RdYlGn_r", aspect="auto")
            ax.set_xticks(range(len(m.pnames))); ax.set_xticklabels(m.pnames)
            ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.2g}%", ha="center", va="center", fontsize=10)
            ax.set_title(f"Матрица ошибок: {mn}"); fig.colorbar(im, ax=ax, label="log10(отн. ошибка, %)")
        fig.tight_layout(); cv.draw()

    def stats(self, mn, me):
        rs = self.sel(mn, me, self.base)
        ys = [np.mean([r["mean_rel"] for r in self.sel(mn, me, l)]) for l in NOISE_LEVELS]
        return dict(err=np.mean([r["mean_rel"] for r in rs]), rmse=np.mean([r["rmse"] for r in rs]),
                    slope=np.polyfit(NOISE_LEVELS, ys, 1)[0],
                    succ=100 * np.mean([r["ok"] for r in rs]), time=np.mean([r["time"] for r in rs]))

    def draw_table(self):
        self.tree.delete(*self.tree.get_children()); notes = []
        for mn in self.models_used:
            st = {me: self.stats(mn, me) for me in METHODS}
            best_err = min(st, key=lambda k: st[k]["err"])
            robust = min(st, key=lambda k: st[k]["slope"])
            for me in METHODS:
                s = st[me]
                self.tree.insert("", "end", tags=("best",) if me == best_err else (), values=(
                    mn, me + (" ★" if me == robust else ""), f"{s['err']:.3g}", f"{s['rmse']:.3g}",
                    f"{s['slope']:.3g}", f"{s['succ']:.0f}", f"{s['time']:.2f}"))
            notes.append(f"{mn}: точнее всех — {best_err}; устойчивее всех к шуму (★) — {robust}")
        self.note["text"] = (f"Показатели при шуме {self.base} %. Деградация — наклон зависимости ошибки от "
                             f"уровня шума; успех — ошибка < {SUCCESS_THR:g} %. Зелёная строка — лучшая по точности.\n"
                             + "\n".join(notes))

    def draw_time(self):
        fig, cv, axs = self._axes("time", 1)
        ax = axs[0]; k = len(self.models_used); w = 0.8 / k
        for i, mn in enumerate(self.models_used):
            vals = [np.mean([r["time"] for r in self.sel(mn, me, self.base)]) for me in METHODS]
            ax.bar(np.arange(len(METHODS)) + i * w - 0.4 + w / 2, vals, w, label=mn)
        ax.set_yscale("log"); ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS)
        ax.set_ylabel("Время, с (log)"); ax.set_title("Время работы алгоритмов")
        ax.legend(); ax.grid(alpha=.3, which="both")
        fig.tight_layout(); cv.draw()

    def draw_noise(self):
        fig, cv, axs = self._axes("noise", len(self.models_used))
        for ax, mn in zip(axs, self.models_used):
            for me in METHODS:
                ys = [np.mean([r["mean_rel"] for r in self.sel(mn, me, l)]) for l in NOISE_LEVELS]
                ax.plot(NOISE_LEVELS, np.maximum(ys, 1e-4), "o-", color=M_COLORS[me], label=me)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks(NOISE_LEVELS); ax.set_xticklabels(NOISE_LEVELS)
            ax.set_xlabel("Уровень шума, % от СКО сигнала")
            ax.set_ylabel("Ср. отн. ошибка параметров, % (log)")
            ax.set_title(f"{mn}: влияние шума"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
        fig.tight_layout(); cv.draw()


if __name__ == "__main__":
    App().mainloop()
