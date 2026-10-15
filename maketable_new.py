"""
Простая таблица для тезисов из results.csv (его создаёт experiment.py).

Запуск:  python make_table.py good      (или bad)
Результат: table_<старт>.csv (открывается в Excel) и вывод в консоль.

Столбцы:
  β, α, γ              найденные значения (типичный запуск = медиана по запускам)
  RMSE, MAE, sMAPE     насколько кривая модели совпала с данными (метрики качества подгонки)
  Время, с             типичное время одного запуска
"""
import sys
import pandas as pd

TRUE = {"beta": 0.65, "alpha": 0.3, "gamma": 0.1}
NAMES = {"NM": "Нелдер-Мид", "DE": "Дифференциальная эволюция",
         "CMA": "CMA-ES", "BO": "Байесовская оптимизация"}
start = sys.argv[1] if len(sys.argv) > 1 else "good"

df = pd.read_csv("results.csv")
df = df[df.start == start].copy()
for p, v in TRUE.items():
    df[f"e_{p}"] = 100 * (df[p] - v).abs() / v
df["err"] = df[["e_beta", "e_alpha", "e_gamma"]].mean(axis=1)
df["fail"] = df[["e_beta", "e_alpha", "e_gamma"]].fillna(1e6).max(axis=1) > 50


def num(v, d, cap):
    if abs(v) >= cap:
        return f">{cap:g}".replace(".", ",")
    return f"{v:.{d}f}".replace(".", ",")


rows = []
for m, name in NAMES.items():
    g = df[df.method == m]
    if g.empty:
        continue
    rows.append({
        "Метод": name,
        "β (истина 0,65)": num(g.beta.median(), 3, 100),
        "α (истина 0,30)": num(g.alpha.median(), 3, 100),
        "γ (истина 0,10)": num(g.gamma.median(), 3, 100),
        "RMSE": num(g.rmse.median(), 1, 1000),
        "MAE": num(g.mae.median(), 1, 1000),
        "sMAPE, %": num(g.smape.median(), 1, 1000),
        "Время, с": num(g.time.median(), 1, 1000),
    })
t = pd.DataFrame(rows)
t.to_csv(f"table_{start}.csv", index=False, encoding="utf-8-sig")
print(f"Старт: {start}\n")
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(t.to_string(index=False))
