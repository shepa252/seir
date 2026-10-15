"""
График разброса найденных параметров (самих β, α, γ) по запускам.

Запуск:  python -m seir.plot_spread
Читает results.csv (его создаёт experiment.py), открывает график в отдельном окне
(с панелью инструментов matplotlib внизу, как в main.py) и сохраняет fig_param_spread.png.

Строки: хороший и плохой старт. Столбцы: β, α, γ.
Каждая точка = один запуск. Ящик = середина 50% запусков, черта = медиана.
Красная пунктирная линия = истинное значение. Точки за пределами шкалы
прижаты к краю (красный треугольник) и подписаны.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# те же настройки оформления, что и в main.py
plt.rcParams.update({
    'font.size': 24, 'axes.labelsize': 24, 'axes.titlesize': 24,
    'xtick.labelsize': 22, 'ytick.labelsize': 22, 'legend.fontsize': 22, 'lines.linewidth': 3
})

METHODS = ["NM", "DE", "CMA", "BO"]
LABELS = {"NM": "Нелдер-\nМид", "DE": "Дифф.\nэволюция", "CMA": "CMA-ES", "BO": "Байес.\nопт."}
COLORS = {"NM": "red", "DE": "blue", "CMA": "purple", "BO": "green"}  # как на ваших графиках
PARAMS = [("beta", "β", 0.65, (-0.4, 1.1)),
          ("alpha", "α", 0.30, (-0.2, 1.1)),
          ("gamma", "γ", 0.10, (-0.5, 0.5))]
STARTS = [("good", "Идеальное приближение\nначальных параметров"),
          ("bad", "Плохое приближение\nначальных параметров")]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / "results.csv")
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
for i, (st, st_title) in enumerate(STARTS):
    for j, (col, name, true, (lo, hi)) in enumerate(PARAMS):
        ax = axes[i, j]
        for k, m in enumerate(METHODS, start=1):
            v = df[(df.start == st) & (df.method == m)][col].values
            ins = v[(v >= lo) & (v <= hi)]
            if len(ins):
                # ax.boxplot(ins, positions=[k], widths=0.5, showfliers=False,
                #            medianprops=dict(color="k", linewidth=2),
                #            boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                #            capprops=dict(linewidth=1.5))
                ax.scatter(k + rng.uniform(-.15, .15, len(ins)), ins, s=60, alpha=.7,
                           zorder=3, color=COLORS[m])
            hi_out, lo_out = v[v > hi], v[v < lo]
            if len(hi_out):
                ax.scatter([k], [hi - 0.04 * (hi - lo)], marker="^", color="k", s=150, zorder=4)
                ax.text(k, hi - 0.16 * (hi - lo), f"{len(hi_out)} шт.\nдо {hi_out.max():.0e}",
                        ha="center", fontsize=13, color="k")
            if len(lo_out):
                ax.scatter([k], [lo + 0.04 * (hi - lo)], marker="v", color="k", s=150, zorder=4)
                ax.text(k, lo + 0.1 * (hi - lo), f"{len(lo_out)} шт.", ha="center", fontsize=13)
        ax.axhline(true, color="r", ls="--", linewidth=2.5)
        # ax.annotate(f"{name} = {true:g}".replace(".", ","), xy=(0.45, true), xytext=(0, 8),
        #             textcoords="offset points", ha="left", va="bottom", color="r",
        #             fontsize=20, fontweight="bold")
        ax.set_ylim(lo, hi)
        ax.set_xlim(0.4, len(METHODS) + 0.6)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels([LABELS[m] for m in METHODS])
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.set_title(name)
        if j == 0:
            ax.set_ylabel(st_title)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "fig_param_spread.png", dpi=200)
plt.show()
