"""
Собирает компактную таблицу для тезисов из results.csv (его создаёт experiment.py).

Запуск:  python make_table.py            # старт "good"
         python make_table.py bad        # старт "bad"
Результат: table_<старт>.csv и вывод в консоль (можно копировать в Word).
"""
import sys
import numpy as np
import pandas as pd

TRUE = {"beta": 0.65, "alpha": 0.3, "gamma": 0.1}
NAMES = {"NM": "Нелдер-Мид", "NM_b": "Нелдер-Мид (границы)", "DE": "Дифф. эволюция",
         "CMA": "CMA-ES", "BO": "Байесовская опт. (TPE)"}
start = sys.argv[1] if len(sys.argv) > 1 else "good"

df = pd.read_csv("results.csv")
df = df[df.start == start].copy()
# сводная ошибка параметров по запуску: среднее относительных ошибок β, α, γ
for p, v in TRUE.items():
    df[f"e_{p}"] = 100 * (df[p] - v).abs() / v
df["mape_par"] = df[["e_beta", "e_alpha", "e_gamma"]].mean(axis=1)
df["fail"] = df[["e_beta", "e_alpha", "e_gamma"]].max(axis=1) > 50


def ms(x, d=3):
    return f"{x.mean():.{d}f} ± {x.std():.{d}f}".replace(".", ",")


rows = []
for m, name in NAMES.items():
    g = df[df.method == m]
    if g.empty:
        continue
    rows.append({
        "Метод": name,
        "β (ист. 0,65)": ms(g.beta),
        "α (ист. 0,30)": ms(g.alpha),
        "γ (ист. 0,10)": ms(g.gamma),
        "Ошибка параметров, % (MAPE)": ms(g.mape_par, 1),
        "Неудач, %": f"{100 * g.fail.mean():.0f}",
        "sMAPE кривой, %": ms(g.smape, 1),
        "RMSE кривой": ms(g.rmse, 1),
        "Время, с": ms(g.time, 1),
    })
t = pd.DataFrame(rows)
t.to_csv(f"table_{start}.csv", index=False, encoding="utf-8-sig")
print(f"Старт: {start}, запусков на метод: {df.groupby('method').size().min()}\n")
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(t.to_string(index=False))
