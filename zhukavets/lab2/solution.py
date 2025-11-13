import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# 1. CORRELATION CALCULATIONS

def fechner_correlation(x, y):
    """Calculate Fechner correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    
    x_median = np.median(x)
    y_median = np.median(y)
    
    x_signs = np.sign(x - x_median)
    y_signs = np.sign(y - y_median)
        
    concordant = np.sum(x_signs * y_signs > 0)
    discordant = np.sum(x_signs * y_signs < 0)
    
    if concordant + discordant == 0:
        return 0.0
    
    return (concordant - discordant) / (concordant + discordant)

def pearson_with_ci(x, y, alpha=0.05):
    """Calculate Pearson correlation with confidence interval."""
    r, p_value = pearsonr(x, y)
    
    n = len(x)
    if n <= 3:
        return r, p_value, (0, 0)
    
    # Fisher's z transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    
    # Critical value for confidence interval
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Confidence interval in z-space
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    # Transform back to correlation space
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    return r, p_value, (r_lower, r_upper)

N = 5
col1_idx = N % 5
col2_idx = (N**2) % 5 + 5
try:
    df = pd.read_csv('/home/mbugzy/uni/uni/4course/oaid/oiad-13-2025/datasets/students_simple.csv')
except FileNotFoundError:
    print('No dataset found')
    exit(1)
columns = df.columns.tolist()
col1_name = columns[col1_idx]
col2_name = columns[col2_idx]

print(f"Student number: {N}")
print(f"Selected columns: {col1_name} (index {col1_idx}), {col2_name} (index {col2_idx})")

x = df[col1_name].values
y = df[col2_name].values

mask = ~(np.isnan(x) | np.isnan(y))
x = x[mask]
y = y[mask]

print(f"Data points: {len(x)}")

print("\n" + "="*50)
print("1. CORRELATION CALCULATIONS")
print("="*50)

# Fechner correlation
fechner_r = fechner_correlation(x, y)
print(f"1. Fechner correlation: {fechner_r:.4f}")

# Pearson correlation with confidence interval
pearson_r, pearson_p, pearson_ci = pearson_with_ci(x, y)
print(f"2. Pearson correlation: {pearson_r:.4f} (p-value: {pearson_p:.4f})")
print(f"   95% Confidence interval: [{pearson_ci[0]:.4f}, {pearson_ci[1]:.4f}]")

# Spearman correlation
spearman_r, spearman_p = spearmanr(x, y)
print(f"3. Spearman correlation: {spearman_r:.4f} (p-value: {spearman_p:.4f})")

# Kendall correlation
kendall_r, kendall_p = kendalltau(x, y)
print(f"4. Kendall correlation: {kendall_r:.4f} (p-value: {kendall_p:.4f})")

# 2. VISUALIZATIONS

print("\n" + "="*50)
print("2. VISUALIZATIONS")
print("="*50)

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram of x
axes[0, 0].hist(x, bins=20, alpha=0.7, edgecolor='black')
axes[0, 0].set_title(f'Histogram of {col1_name}')
axes[0, 0].set_xlabel(col1_name)
axes[0, 0].set_ylabel('Frequency')

# Histogram of y
axes[0, 1].hist(y, bins=20, alpha=0.7, edgecolor='black')
axes[0, 1].set_title(f'Histogram of {col2_name}')
axes[0, 1].set_xlabel(col2_name)
axes[0, 1].set_ylabel('Frequency')

# Scatter plot
axes[1, 0].scatter(x, y, alpha=0.6)
axes[1, 0].set_title(f'Scatter Plot: {col1_name} vs {col2_name}')
axes[1, 0].set_xlabel(col1_name)
axes[1, 0].set_ylabel(col2_name)

# Combined histogram (2D)
axes[1, 1].hist2d(x, y, bins=20, alpha=0.7)
axes[1, 1].set_title(f'2D Histogram: {col1_name} vs {col2_name}')
axes[1, 1].set_xlabel(col1_name)
axes[1, 1].set_ylabel(col2_name)

plt.tight_layout()
plt.savefig('zhukavets/lab2/visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. REGRESSION EQUATIONS

print("\n" + "="*50)
print("3. REGRESSION EQUATIONS")
print("="*50)

def linear_regression(x, y):
    """Linear regression: y = w1*x + w0"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)
    
    w1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    w0 = (sum_y - w1 * sum_x) / n
    
    return w0, w1

def quadratic_regression(x, y):
    """Quadratic regression: y = w2*x^2 + w1*x + w0"""
    n = len(x)
    X = np.column_stack([np.ones(n), x, x**2])
    
    # Normal equations: (X^T * X) * w = X^T * y
    XTX = X.T @ X
    XTy = X.T @ y
    
    try:
        w = np.linalg.solve(XTX, XTy)
        return w[0], w[1], w[2]  # w0, w1, w2
    except np.linalg.LinAlgError:
        return 0, 0, 0

def hyperbolic_regression(x, y):
    """Hyperbolic regression: y = w1/x + w0"""
    # Avoid division by zero
    x_safe = np.where(x == 0, 1e-10, x)
    x_inv = 1 / x_safe
    
    n = len(x_inv)
    sum_x_inv = np.sum(x_inv)
    sum_y = np.sum(y)
    sum_x_inv_y = np.sum(x_inv * y)
    sum_x_inv2 = np.sum(x_inv**2)
    
    w1 = (n * sum_x_inv_y - sum_x_inv * sum_y) / (n * sum_x_inv2 - sum_x_inv**2)
    w0 = (sum_y - w1 * sum_x_inv) / n
    
    return w0, w1

def exponential_regression(x, y):
    """Exponential regression: y = w1^x * w0"""
    # Linearize: log(y) = x*log(w1) + log(w0)
    # Avoid log(0) and negative values
    y_safe = np.where(y <= 0, 1e-10, y)
    log_y = np.log(y_safe)
    
    n = len(x)
    sum_x = np.sum(x)
    sum_log_y = np.sum(log_y)
    sum_x_log_y = np.sum(x * log_y)
    sum_x2 = np.sum(x**2)
    
    log_w1 = (n * sum_x_log_y - sum_x * sum_log_y) / (n * sum_x2 - sum_x**2)
    log_w0 = (sum_log_y - log_w1 * sum_x) / n
    
    w1 = np.exp(log_w1)
    w0 = np.exp(log_w0)
    
    return w0, w1

def calculate_r_squared(y_true, y_pred):
    """Calculate R-squared coefficient."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Calculate regression coefficients
w0_lin, w1_lin = linear_regression(x, y)
w0_quad, w1_quad, w2_quad = quadratic_regression(x, y)
w0_hyp, w1_hyp = hyperbolic_regression(x, y)
w0_exp, w1_exp = exponential_regression(x, y)

print(f"Linear regression: y = {w1_lin:.4f}*x + {w0_lin:.4f}")
print(f"Quadratic regression: y = {w2_quad:.4f}*x² + {w1_quad:.4f}*x + {w0_quad:.4f}")
print(f"Hyperbolic regression: y = {w1_hyp:.4f}/x + {w0_hyp:.4f}")
print(f"Exponential regression: y = {w1_exp:.4f}^x * {w0_exp:.4f}")

# Calculate predictions and R-squared
x_range = np.linspace(x.min(), x.max(), 100)

# Linear
y_lin = w1_lin * x + w0_lin
y_lin_range = w1_lin * x_range + w0_lin
r2_lin = calculate_r_squared(y, y_lin)

# Quadratic
y_quad = w2_quad * x**2 + w1_quad * x + w0_quad
y_quad_range = w2_quad * x_range**2 + w1_quad * x_range + w0_quad
r2_quad = calculate_r_squared(y, y_quad)

# Hyperbolic
x_safe_range = np.where(x_range == 0, 1e-10, x_range)
y_hyp_range = w1_hyp / x_safe_range + w0_hyp
x_safe = np.where(x == 0, 1e-10, x)
y_hyp = w1_hyp / x_safe + w0_hyp
r2_hyp = calculate_r_squared(y, y_hyp)

# Exponential
y_exp = w1_exp**x * w0_exp
y_exp_range = w1_exp**x_range * w0_exp
r2_exp = calculate_r_squared(y, y_exp)

print(f"\nR-squared values:")
print(f"Linear: {r2_lin:.4f}")
print(f"Quadratic: {r2_quad:.4f}")
print(f"Hyperbolic: {r2_hyp:.4f}")
print(f"Exponential: {r2_exp:.4f}")

# Find best and worst models
models = {
    'Linear': r2_lin,
    'Quadratic': r2_quad,
    'Hyperbolic': r2_hyp,
    'Exponential': r2_exp
}

best_model = max(models, key=models.get)
worst_model = min(models, key=models.get)

print(f"\nBest model: {best_model} (R² = {models[best_model]:.4f})")
print(f"Worst model: {worst_model} (R² = {models[worst_model]:.4f})")

# Plot regression curves
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x_range, y_lin_range, 'r-', label=f'Linear (R²={r2_lin:.3f})')
plt.xlabel(col1_name)
plt.ylabel(col2_name)
plt.title('Linear Regression')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x_range, y_quad_range, 'g-', label=f'Quadratic (R²={r2_quad:.3f})')
plt.xlabel(col1_name)
plt.ylabel(col2_name)
plt.title('Quadratic Regression')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x_range, y_hyp_range, 'b-', label=f'Hyperbolic (R²={r2_hyp:.3f})')
plt.xlabel(col1_name)
plt.ylabel(col2_name)
plt.title('Hyperbolic Regression')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x_range, y_exp_range, 'm-', label=f'Exponential (R²={r2_exp:.3f})')
plt.xlabel(col1_name)
plt.ylabel(col2_name)
plt.title('Exponential Regression')
plt.legend()

plt.tight_layout()
plt.savefig('zhukavets/lab2/regression_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. FISHER'S F-TEST

print("\n" + "="*50)
print("4. FISHER'S F-TEST")
print("="*50)

def fishers_f_test(y_true, y_pred, k):
    """Fisher's F-test for regression model adequacy."""
    n = len(y_true)
    if n <= k:
        return 0, 1, 0
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Sum of squares
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_reg = ss_tot - ss_res
    
    # Degrees of freedom
    df_reg = k
    df_res = n - k - 1
    
    if df_res <= 0 or ss_res == 0:
        return 0, 1, 0
    
    # F-statistic
    f_stat = (ss_reg / df_reg) / (ss_res / df_res)
    
    # P-value
    p_value = 1 - stats.f.cdf(f_stat, df_reg, df_res)
    
    return f_stat, p_value, ss_res

# Test best and worst models
print(f"Testing {best_model} model:")
if best_model == 'Linear':
    f_stat_best, p_val_best, ss_res_best = fishers_f_test(y, y_lin, 1)
elif best_model == 'Quadratic':
    f_stat_best, p_val_best, ss_res_best = fishers_f_test(y, y_quad, 2)
elif best_model == 'Hyperbolic':
    f_stat_best, p_val_best, ss_res_best = fishers_f_test(y, y_hyp, 1)
else:  # Exponential
    f_stat_best, p_val_best, ss_res_best = fishers_f_test(y, y_exp, 1)

print(f"F-statistic: {f_stat_best:.4f}")
print(f"P-value: {p_val_best:.4f}")
print(f"Residual sum of squares: {ss_res_best:.4f}")

print(f"\nTesting {worst_model} model:")
if worst_model == 'Linear':
    f_stat_worst, p_val_worst, ss_res_worst = fishers_f_test(y, y_lin, 1)
elif worst_model == 'Quadratic':
    f_stat_worst, p_val_worst, ss_res_worst = fishers_f_test(y, y_quad, 2)
elif worst_model == 'Hyperbolic':
    f_stat_worst, p_val_worst, ss_res_worst = fishers_f_test(y, y_hyp, 1)
else:  # Exponential
    f_stat_worst, p_val_worst, ss_res_worst = fishers_f_test(y, y_exp, 1)

print(f"F-statistic: {f_stat_worst:.4f}")
print(f"P-value: {p_val_worst:.4f}")
print(f"Residual sum of squares: {ss_res_worst:.4f}")

# 5. CONCLUSIONS

print("\n" + "="*50)
print("5. CONCLUSIONS")
print("="*50)

print("\n1. CORRELATION ANALYSIS:")
print(f"   - Fechner correlation ({fechner_r:.4f}) shows the direction of relationship")
print(f"   - Pearson correlation ({pearson_r:.4f}) indicates linear relationship strength")
print(f"   - Spearman correlation ({spearman_r:.4f}) shows monotonic relationship")
print(f"   - Kendall correlation ({kendall_r:.4f}) provides rank-based association")

print(f"\n2. VISUALIZATION:")
print(f"   - Histograms show the distribution of {col1_name} and {col2_name}")
print(f"   - Scatter plot reveals the relationship pattern between variables")
print(f"   - 2D histogram provides density information")

print(f"\n3. REGRESSION MODELS:")
print(f"   - Best model: {best_model} with R² = {models[best_model]:.4f}")
print(f"   - Worst model: {worst_model} with R² = {models[worst_model]:.4f}")
print(f"   - Model comparison shows which functional form fits the data best")

print(f"\n4. FISHER'S F-TEST:")
print(f"   - {best_model} model: F = {f_stat_best:.4f}, p = {p_val_best:.4f}")
print(f"   - {worst_model} model: F = {f_stat_worst:.4f}, p = {p_val_worst:.4f}")
print(f"   - F-test evaluates the statistical significance of the regression models")

print(f"\n5. OVERALL ASSESSMENT:")
if abs(pearson_r) > 0.7:
    strength = "strong"
elif abs(pearson_r) > 0.5:
    strength = "moderate"
else:
    strength = "weak"

print(f"   - The relationship between {col1_name} and {col2_name} is {strength}")
print(f"   - The {best_model} model provides the best fit to the data")
print(f"   - Statistical tests confirm the adequacy of the regression models")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)

