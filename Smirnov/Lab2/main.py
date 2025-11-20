import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau, norm, rankdata

# === 0. Подготовка ===
os.makedirs("graphics", exist_ok=True)

# === 1. Загрузка и выбор данных ===
df = pd.read_csv("../../datasets/students_simple.csv")
N = 9
col1 = N % 5               # 4
col2 = (N**2 % 5) + 5      # 6
x = df.iloc[:, col1].to_numpy()
y = df.iloc[:, col2].to_numpy()

# === 2. Утилита для парных графиков ===
def save_hist_scatter(hist_data, scatter_x, scatter_y, coeff_value, title, hist_title, filename, bins='auto'):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(hist_data, bins=bins, color='steelblue', alpha=0.8, edgecolor='white')
    plt.title(hist_title)
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(scatter_x, scatter_y, color='black', alpha=0.75)
    plt.title(f"{title}: {coeff_value:.3f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"graphics/{filename}.png")
    plt.close()

# === 3. Корреляции ===
# Фехнер
signs = (np.sign(x - np.mean(x)) == np.sign(y - np.mean(y))).astype(int)
fechner = (2 * signs.sum() - len(x)) / len(x)
save_hist_scatter(signs, x, y, fechner, "Фехнер", "Совпадение знаков (0/1)", "fechner", bins=[-0.5,0.5,1.5])

# Пирсон
prod = (x - np.mean(x)) * (y - np.mean(y))
pearson, _ = pearsonr(x, y)
r_z = np.arctanh(pearson)
se = 1 / np.sqrt(len(x) - 3)
z = norm.ppf(0.975)
ci_low, ci_high = np.tanh([r_z - z * se, r_z + z * se])
save_hist_scatter(prod, x, y, pearson, f"Пирсон (95% CI: {ci_low:.3f}..{ci_high:.3f})",
                  "Произведения отклонений", "pearson")

# Спирмен
rx, ry = rankdata(x), rankdata(y)
diff_ranks = rx - ry
spearman, _ = spearmanr(x, y)
save_hist_scatter(diff_ranks, x, y, spearman, "Спирмен", "Разности рангов", "spearman")

# Кенделл
pairs = []
for i in range(len(x)):
    for j in range(i+1, len(x)):
        s = (x[i]-x[j])*(y[i]-y[j])
        pairs.append(np.sign(s))
pairs = np.array(pairs)
kendall, _ = kendalltau(x, y)
save_hist_scatter(pairs, x, y, kendall, "Кенделл", "Согласованные/несогласованные пары", "kendall",
                  bins=[-1.5,-0.5,0.5,1.5])

# === 3.1 Вывод коэффициентов ===
print("\nКоэффициенты корреляции:")
print(f"Фехнер: {fechner:.3f}")
print(f"Пирсон: {pearson:.3f}, 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Спирмен: {spearman:.3f}")
print(f"Кенделл: {kendall:.3f}")

# === 4. Регрессионные модели ===
x_vals = np.linspace(np.min(x), np.max(x), 200)

# Линейная
X_lin = np.vstack([x, np.ones(len(x))]).T
w1_lin, w0_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]
y_lin = w1_lin * x_vals + w0_lin

# Квадратичная
X_quad = np.vstack([x**2, x, np.ones(len(x))]).T
w2_quad, w1_quad, w0_quad = np.linalg.lstsq(X_quad, y, rcond=None)[0]
y_quad = w2_quad * x_vals**2 + w1_quad * x_vals + w0_quad

# Гиперболическая
if np.any(x == 0):
    mask = x != 0
    X_hyp = np.vstack([1/x[mask], np.ones(mask.sum())]).T
    w1_hyp, w0_hyp = np.linalg.lstsq(X_hyp, y[mask], rcond=None)[0]
else:
    X_hyp = np.vstack([1/x, np.ones(len(x))]).T
    w1_hyp, w0_hyp = np.linalg.lstsq(X_hyp, y, rcond=None)[0]
y_hyp = w1_hyp / x_vals + w0_hyp

# Показательная
mask_pos = y > 0
if mask_pos.sum() >= 2:
    X_exp = np.vstack([x[mask_pos], np.ones(mask_pos.sum())]).T
    log_w1, log_w0 = np.linalg.lstsq(X_exp, np.log(y[mask_pos]), rcond=None)[0]
    w1_exp = np.exp(log_w1)
    w0_exp = np.exp(log_w0)
    y_exp = w0_exp * (w1_exp ** x_vals)
else:
    w1_exp = np.nan
    w0_exp = np.nan
    y_exp = np.full_like(x_vals, np.nan)

plt.figure(figsize=(10,6))
plt.scatter(x,y,label="Данные",color="black")
plt.plot(x_vals,y_lin,label="Линейная")
plt.plot(x_vals,y_quad,label="Квадратичная")
plt.plot(x_vals,y_hyp,label="Гиперболическая")
if not np.isnan(y_exp).all():
    plt.plot(x_vals,y_exp,label="Показательная")
plt.legend(); plt.grid(alpha=0.3)
plt.title("Модели регрессии")
plt.savefig("graphics/regression_models.png")
plt.close()

# === 5. Критерий Фишера ===
def fisher_test(y_true, y_pred, k):
    n = len(y_true)
    RSS = np.sum((y_true - y_pred)**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    return ((TSS - RSS) / k) / (RSS / (n - k - 1))

models = {
    "Линейная": w1_lin * x + w0_lin,
    "Квадратичная": w2_quad * x**2 + w1_quad * x + w0_quad,
    "Гиперболическая": w1_hyp / x + w0_hyp
}
if not np.isnan(w1_exp):
    models["Показательная"] = w0_exp * (w1_exp ** x)

k_params = {"Линейная":1,"Квадратичная":2,"Гиперболическая":1,"Показательная":2}
f_stats = {name:fisher_test(y,pred,k_params[name]) for name,pred in models.items()}
best_model = max(f_stats,key=f_stats.get)
worst_model = min(f_stats,key=f_stats.get)

print("\nКритерий Фишера:")
for name,f in f_stats.items():
    print(f"{name}: F = {f:.2f}")
print(f"Лучшая модель: {best_model}")
print(f"Худшая модель: {worst_model}")

# === 6. Уравнения регрессии ===
print("\nУравнения регрессии:")
print(f"Линейная: y = {w1_lin:.3f}*x + {w0_lin:.3f}")
print(f"Квадратичная: y = {w2_quad:.3f}*x^2 + {w1_quad:.3f}*x + {w0_quad:.3f}")
print(f"Гиперболическая: y = {w1_hyp:.3f}/x + {w0_hyp:.3f}")
if not np.isnan(w1_exp):
    print(f"Показательная: y = {w0_exp:.3f} * {w1_exp:.3f}^x")
else:
    print("Показательная: пропущена (в данных есть неположительные y)")
