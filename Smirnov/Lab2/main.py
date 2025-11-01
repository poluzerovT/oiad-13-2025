import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau, norm
import os

# === Подготовка папки для графиков ===
os.makedirs("graphics", exist_ok=True)

# === 0. Загрузка и выбор данных ===
df = pd.read_csv("../../datasets/students_simple.csv")
N = 9

col1 = N % 5 #4
col2 = (N**2 % 5) + 5 #6

x = df.iloc[:, col1]
y = df.iloc[:, col2]

# === 1. Корреляции ===
def fechner_corr(x, y):
    signs = np.sign(x - np.mean(x)) == np.sign(y - np.mean(y))
    return (2 * np.sum(signs) - len(x)) / len(x)

fechner = fechner_corr(x, y)
pearson, _ = pearsonr(x, y)
spearman, _ = spearmanr(x, y)
kendall, _ = kendalltau(x, y)

r_z = np.arctanh(pearson)
se = 1 / np.sqrt(len(x) - 3)
z = norm.ppf(0.975)
ci_low, ci_high = np.tanh([r_z - z * se, r_z + z * se])

print("Корреляции:")
print(f"Фехнер: {fechner:.3f} <- обратная умеренная" )
print(f"Пирсон: {pearson:.3f}, 95% CI: [{ci_low:.3f}, {ci_high:.3f}] <- практически полностью независимы")
print(f"Спирмен: {spearman:.3f} <- отрицательная монотонная связь")
print(f"Кенделл: {kendall:.3f} <- слабая отрицательная монотонная связь")

plt.figure(figsize=(10, 8))

# Общие бины
bins = np.histogram_bin_edges(np.concatenate([x, y]), bins='auto')

# Гистограмма X
plt.subplot(2, 2, 1)
plt.hist(x, bins=bins, density=True, color='steelblue', alpha=0.7)
plt.title("Гистограмма X")
plt.xlabel("Значения")
plt.ylabel("Плотность")

# График рассеяния — на всю ширину
plt.subplot(2, 1, 2)
plt.scatter(x, y, color='black', alpha=0.7)
plt.title("График рассеяния")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.savefig("graphics/histograms_and_scatter_vertical.png")
plt.close()

# === 3. Регрессии ===
x_vals = np.linspace(min(x), max(x), 100)

# Линейная
X_lin = np.vstack([x, np.ones(len(x))]).T
w1_lin, w0_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]
y_lin = w1_lin * x_vals + w0_lin

# Квадратичная
X_quad = np.vstack([x**2, x, np.ones(len(x))]).T
w2_quad, w1_quad, w0_quad = np.linalg.lstsq(X_quad, y, rcond=None)[0]
y_quad = w2_quad * x_vals**2 + w1_quad * x_vals + w0_quad

# Гиперболическая
X_hyp = np.vstack([1/x, np.ones(len(x))]).T
w1_hyp, w0_hyp = np.linalg.lstsq(X_hyp, y, rcond=None)[0]
y_hyp = w1_hyp / x_vals + w0_hyp

# Показательная
log_y = np.log(y)
X_exp = np.vstack([x, np.ones(len(x))]).T
log_w1, log_w0 = np.linalg.lstsq(X_exp, log_y, rcond=None)[0]
w1_exp = np.exp(log_w1)
w0_exp = np.exp(log_w0)
y_exp = w0_exp * w1_exp**x_vals

# Визуализация моделей
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Данные', color='black')
plt.plot(x_vals, y_lin, label='Линейная')
plt.plot(x_vals, y_quad, label='Квадратичная')
plt.plot(x_vals, y_hyp, label='Гиперболическая')
plt.plot(x_vals, y_exp, label='Показательная')
plt.title("Модели регрессии")
plt.legend()
plt.savefig("graphics/regression_models.png")
plt.close()

# === 4. Критерий Фишера ===
def fisher_test(y_true, y_pred, k):
    n = len(y_true)
    RSS = np.sum((y_true - y_pred)**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    return ((TSS - RSS) / k) / (RSS / (n - k - 1))

models = {
    "Линейная": w1_lin * x + w0_lin,
    "Квадратичная": w2_quad * x**2 + w1_quad * x + w0_quad,
    "Гиперболическая": w1_hyp / x + w0_hyp,
    "Показательная": w0_exp * w1_exp**x
}

f_stats = {name: fisher_test(y, pred, k=(1 if name == "Линейная" else 2)) for name, pred in models.items()}
best_model = max(f_stats, key=f_stats.get)
worst_model = min(f_stats, key=f_stats.get)

print("\nКритерий Фишера:")
for name, f in f_stats.items():
    print(f"{name}: F = {f:.2f}")
print(f"\nЛучшая модель: {best_model}")
print(f"Худшая модель: {worst_model}")


# === 3.1 Уравнения регрессии ===
print("\nУравнения регрессии:")
print(f"Линейная: y = {w1_lin:.3f} * x + {w0_lin:.3f}")
print(f"Квадратичная: y = {w2_quad:.3f} * x^2 + {w1_quad:.3f} * x + {w0_quad:.3f}")
print(f"Гиперболическая: y = {w1_hyp:.3f} / x + {w0_hyp:.3f}")
print(f"Показательная: y = {w0_exp:.3f} * {w1_exp:.3f}^x")

