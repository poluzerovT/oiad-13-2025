import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('../../datasets/teen_phone_addiction_dataset.csv')

group_number = 9
column_index = (group_number % 7) +5 # 9 % 7 = 2
column_name = df.columns[column_index]

data = df[column_name].dropna()

# === I. Описательная статистика ===
print("Среднее значение:", np.mean(data))
print("Дисперсия:", np.var(data, ddof=1))
print("Мода:", stats.mode(data, keepdims=True)[0][0])
print("Медиана:", np.median(data))
print("Квантили (25%, 50%, 75%):", np.quantile(data, [0.25, 0.5, 0.75]))
print("Эксцесс (куртозис):", stats.kurtosis(data))
print("Асимметрия:", stats.skew(data))
print("Межквартильный размах (IQR):", stats.iqr(data))

"""
counts = data.value_counts().sort_index()
for value, count in counts.items():
    print(f"{value}: {count} раз(а)")
"""

# === Графики: гистограмма и ЭФР ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data, kde=False, bins=20)
plt.title('Histogram')
plt.subplot(1, 2, 2)
sns.ecdfplot(data)
plt.title('Empirical CDF')
plt.tight_layout()
plt.savefig("graphics/output_descriptive_graphs.png")

print("\n— Промежуточный вывод I —")
print("Данные лежат в диапазоне от", data.min(), "до", data.max())
print("Центр распределения — около", np.median(data), ", но среднее немного смещено:", np.mean(data))
print("Асимметрия =", stats.skew(data), "→", "распределение", "вправо" if stats.skew(data) > 0 else "влево" if stats.skew(data) < 0 else "симметрично")
print("Эксцесс =", stats.kurtosis(data), "→", "распределение", "острое" if stats.kurtosis(data) > 0 else "плоское" if stats.kurtosis(data) < 0 else "нормальное")


# === II. Проверка на нормальность ===
def chi_square_test(data, bins=5):
    observed, bin_edges = np.histogram(data, bins=bins)
    expected = len(data) * np.diff(stats.norm.cdf(bin_edges, np.mean(data), np.std(data)))
    chi2_stat = ((observed - expected) ** 2 / expected).sum()
    df_chi = bins - 1 - 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, df_chi)


    return chi2_stat, p_value

chi2_stat, p_val = chi_square_test(data)
print("Критерий хи-квадрат: статистика =", chi2_stat, ", p-значение =", p_val)

# ВАЖНО: значения в выборке лежат в диапазоне от 13 до 19 — это дискретные и ограниченные данные.
# Поэтому хи-квадрат может показывать сильное отклонение даже при "нормальном" распределении.
# Для таких данных лучше использовать визуальный анализ (Q-Q график) или тест Шапиро-Уилка.

print("Тест на асимметрию: p-значение =", stats.skewtest(data).pvalue)
print("Тест на эксцесс: p-значение =", stats.kurtosistest(data).pvalue)

print("\n— Промежуточный вывод II —")
if p_val < 0.05:
    print("Хи-квадрат показывает значительное отклонение от нормального распределения (p =", p_val, ")")
else:
    print("Хи-квадрат не выявил значимого отклонения (p =", p_val, ")")

if stats.skewtest(data).pvalue < 0.05:
    print("Асимметрия статистически значима → распределение несимметрично")
else:
    print("Асимметрия незначима → распределение может быть симметричным")

if stats.kurtosistest(data).pvalue < 0.05:
    print("Эксцесс статистически значим → распределение отличается по пиковости")
else:
    print("Эксцесс незначим → пиковость близка к нормальной")


# Q-Q plot
plt.figure()
stats.probplot(data, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.savefig("graphics/output_qq_plot.png")

# === III. Обработка данных ===
z_scores = np.abs(stats.zscore(data))
data_clean = data[z_scores < 3]
data_log = np.log1p(data_clean)
data_std = (data_log - np.mean(data_log)) / np.std(data_log)

# Повторная проверка
print("\nПосле преобразования:")
print("Среднее значение:", np.mean(data_std))
print("Дисперсия:", np.var(data_std, ddof=1))
print("Асимметрия:", stats.skew(data_std))
print("Эксцесс (куртозис):", stats.kurtosis(data_std))

# Q-Q plot после обработки
plt.figure()
stats.probplot(data_std, dist="norm", plot=plt)
plt.title("Q-Q Plot (Transformed)")
plt.savefig("graphics/output_qq_plot_transformed.png")

print("\n— Промежуточный вывод III —")
print("После удаления выбросов и логарифмического преобразования:")
print("Асимметрия =", stats.skew(data_std), "→", "распределение", "вправо" if stats.skew(data_std) > 0 else "влево" if stats.skew(data_std) < 0 else "симметрично")
print("Эксцесс =", stats.kurtosis(data_std), "→", "распределение", "острое" if stats.kurtosis(data_std) > 0 else "плоское" if stats.kurtosis(data_std) < 0 else "нормальное")


# === IV. Группировка по School_Grade ===
grouped = df.groupby('School_Grade')[column_name]

plt.figure(figsize=(10, 6))
for name, group in grouped:
    sns.histplot(group.dropna(), label=f'Grade {name}', kde=False, bins=15, alpha=0.5)
plt.title('Histograms by School Grade')
plt.legend()
plt.tight_layout()
plt.savefig("graphics/output_school_grade_histograms.png")

print("\n— Промежуточный вывод IV —")
print("Распределение по классам показывает различия в среднем значении и разбросе:")
for name, group in grouped:
    print(f"Класс {name}: Среднее = {group.mean():.2f}, Дисперсия = {group.var(ddof=1):.2f}")








